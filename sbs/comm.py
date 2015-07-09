#!/usr/bin/env python
# encoding: utf-8
"""
    Implemented using regular sockets.
"""

import socket as skt
import numpy as np
import sys
import subprocess as sp
import os.path as osp
import cPickle as pkl
import os
import atexit
import functools as ft

from .logcfg import log
from . import utils

BUFLEN = 4096

# send object as pickle over a socket
def send_object(socket, obj):
    obj_str = pkl.dumps(obj, protocol=-1)
    # first, send the length
    obj_len = len(obj_str)
    log.debug("Object length: {}".format(obj_len))
    socket.send(str(obj_len))

    ack = socket.recv(BUFLEN)
    assert ack == "ACK"

    send_counter = 0
    while send_counter < obj_len:
        chunk_size = socket.send(obj_str[send_counter:])

        if chunk_size == 0:
            raise RuntimeError("Socket connection lost.")

        send_counter += chunk_size
    ack = socket.recv(BUFLEN)
    assert ack == "ACK"


def recv_object(socket):
    obj_len = int(socket.recv(BUFLEN))
    socket.send("ACK")
    recv_counter = 0
    chunks = []
    while recv_counter < obj_len:
        chunk = socket.recv(BUFLEN)
        if chunk == "":
            raise RuntimeError("Socket connection lost.")

        recv_counter += len(chunk)
        chunks.append(chunk)

    obj = pkl.loads("".join(chunks))
    socket.send("ACK")
    return obj


# run a function in a subprocess in subprocess with a single decorator
# the sub function should not have any data dependencies on the rest
# of the program as a new instance of the program will be created
#
# NOTE: This decorator REQUIRES you to encapsulate your python scripts with a
# check for __main__ in __name__, otherwise code might be executed twice if you
# decorate a function from the main file.
#
# NOTE: When dealing with numpy arrays, make sure to pass them in args or
# kwargs directly, otherwise they will be needlessly pickled (=copied).
class RunInSubprocess(object):
    """
        A functor that replaces the original function.
    """
    def __init__(self, func):
        self._func = func
        self.__doc__ = getattr(func, "__doc__", "")

        self._func_name = func.func_name
        self._func_module = func.__module__
        try:
            self._func_dir = self._get_func_dir(self._func_module)
        except AttributeError:
            log.error("{} was not defined in a static module, cannot run in "
                    "subprocess!".format(self._func_name))

    def __call__(self, *args, **kwargs):
        return self._host(*args, **kwargs)

    def _host(self, *args, **kwargs):
        script_filename = None
        return_values = None
        process = None
        try:
            socket, address, port = self._setup_socket_host()
            script_filename = self._setup_script_file(address, port)
            atexit.register(ft.partial(_delete_script_file,
                script_filename, warn=False))

            socket.listen(1)

            process = self._spawn_process(script_filename)

            conn, client_address = socket.accept()

            self._send_arguments(conn, args, kwargs)
            return_values = self._recv_returnvalue(conn)

            process.wait()
        except:
            raise
        finally:
            if process is not None and process.poll() is None:
                process.kill()
            if script_filename is not None:
                _delete_script_file(script_filename)

        return return_values

    def _client(self, address_tpl):
        socket = self._setup_socket_client(address_tpl)
        args, kwargs = self._recv_arguments(socket)

        return_value = None
        try:
            return_value = self._func(*args, **kwargs)
        finally:
            self._send_returnvalue(socket, return_value)


    def _spawn_process(self, script_filename):
        log.debug("Spawning subprocess..")
        return sp.Popen([sys.executable, script_filename],
                cwd=self._func_dir)

    def _setup_socket_host(self):
        log.debug("Setting up host socket..")

        socket = skt.socket(skt.AF_INET, skt.SOCK_STREAM)
        socket.bind(("localhost", 0))
        address, port = socket.getsockname()

        return socket, address, port

    def _setup_socket_client(self, address_tpl):
        log.debug("Setting up client socket..")

        socket = skt.socket(skt.AF_INET, skt.SOCK_STREAM)
        socket.connect(address_tpl)

        return socket

    def _get_func_dir(self, module_name):
        """
            Get the toplevel-directory of the module so that the import works
            in the submodule.
        """
        module = sys.modules[module_name]
        module_path = osp.abspath(module.__file__)

        if module_name == "__main__":
            return osp.dirname(module_path)

        base_module_name = module_name.split(".")[0]
        module_path_split = module_path.split(osp.sep)

        # adjust so osp.join creates an absolute path
        module_path_split[0] = "/"

        # find out where the base_module_folder is residing
        for i_end, folder in enumerate(module_path_split):
            if folder == base_module_name:
                break
        func_dir = osp.join(*module_path_split[:i_end])
        if not osp.isdir(func_dir):
            log.debug("func_dir: {} is no directory, shortening…".format(
                func_dir))
            func_dir = osp.dirname(func_dir)
        log.debug("func_dir: {}".format(func_dir))
        return func_dir

    def _get_transmitter(self, obj):
        if isinstance(obj, np.ndarray) and obj.flags.c_contiguous:
            return "array"
        else:
            return "zipped_pkl"

    def _get_transmitter_iter(self, obj):
        return [self._get_transmitter(o) for o in obj]

    def _get_transmitter_dict(self, obj):
        return {k: self._get_transmitter(v) for k,v in obj.iteritems()}

    def _send_arguments(self, socket, args, kwargs):
        log.debug("Sending arguments.")
        send_object(socket, (args, kwargs))

    def _recv_arguments(self, socket):
        log.debug("Receiving arguments.")

        args, kwargs = recv_object(socket)

        return args, kwargs

    def _send_returnvalue(self, socket, retval):
        log.debug("Sending return value.")
        send_object(socket, retval)

    def _recv_returnvalue(self, socket):
        log.debug("Receiving return value.")
        return recv_object(socket)

    def _get_module_import_name(self):
        if self._func_module != "__main__":
            return self._func_module
        else:
            module_path = osp.abspath(sys.modules[self._func_module].__file__)
            module_path = osp.basename(module_path)
            return osp.splitext(module_path)[0]

    def _setup_script_file(self, address, port):
        # NOTE: Currently we are using shared memory because the process is
        # always started on the same machine, might change in the future.

        log.debug("Setting up script file.")
        filename = "/dev/shm/{}_{}.py".format(self._func_module.split(".")[0],
                utils.get_random_string())
        log.debug("Script filename {}.".format(filename))
        script = open(filename, "w")

        # write preamble
        script.write("#!{}\n".format(sys.executable))
        script.write("import sys, os\n")
        script.write("sys.path.append(os.getcwd())\n")

        # import the needed module
        script.write("import {} as target_module\n".format(
            self._get_module_import_name()))

        # execute the client subfunction with the passed address
        script.write("target_module.{}._client((\"{}\", {}))\n".format(
            self._func_name, address, port))

        script.close()

        return filename


# utility functions

def _delete_script_file(script_filename, warn=True):
    try:
        os.remove(script_filename)
        log.debug("Deleted script file.")
    except OSError:
        if warn:
            log.warn(
                "Could not delete temporary script file for subprocess.")

