#!/usr/bin/env python
# encoding: utf-8

import zmq
import numpy as np
import sys
import subprocess as sp
import os.path as osp
import zlib
import cPickle as pkl
import peewee as pw
import itertools as it
import os

from .logcfg import log
from . import utils

# taken from pyzmq examples
class SerializingSocket(zmq.Socket):
    """
        A class with some extra serialization methods send_zipped_pkl is
        just like send_pyobj, but uses zlib to compress the stream before
        sending. send_array sends numpy arrays with metadata necessary for
        reconstructing the array on the other side (dtype,shape).
    """

    def send_zipped_pkl(self, obj, flags=0, protocol=-1):
        """
            pack and compress an object with pkl and zlib.
        """
        pobj = pkl.dumps(obj, protocol)
        zobj = zlib.compress(pobj)
        log.debug("Zipped pickle is {:d} bytes.".format(len(zobj)))
        return self.send(zobj, flags=flags)

    def recv_zipped_pkl(self, flags=0):
        """
            reconstruct a Python object sent with zipped_pkl
        """
        zobj = self.recv(flags)
        pobj = zlib.decompress(zobj)
        return pkl.loads(pobj)

    def send_array(self, A, flags=0, copy=True, track=False):
        """
            send a numpy array with metadata
        """
        md = dict(
            dtype = str(A.dtype),
            shape = A.shape,
        )
        self.send_json(md, flags|zmq.SNDMORE)
        return self.send(A, flags, copy=copy, track=track)

    def recv_array(self, flags=0, copy=True, track=False):
        """
            recv a numpy array
        """
        md = self.recv_json(flags=flags)
        msg = self.recv(flags=flags, copy=copy, track=track)
        buf = buffer(msg)
        A = np.frombuffer(buf, dtype=md['dtype'])
        return A.reshape(md['shape'])

    # does not work because storage_fields need db-access!!!
    # ignore for now
    #
    # def send_model(self, model, flags=0):
        # meta = {
                # "name" : model.__class__.__name__,
                # "module" : model.__class__.__module__
            # }

        # model_fields = { k: getattr(model, k)\
                # for k in dir(model)\
                # if isinstance(getattr(model.__class__, k), pw.Field)
            # }

        # storage_field_names = getattr(model, "_storage_fields", [])
        # storage_fields = { k: getattr(model, k)\
                # for k in storage_field_names
            # }

        # self.send_json(meta, flags=zmq.SNDMORE)
        # self.send_json(model_fields, flags=zmq.SNDMORE)
        # self.send_json(storage_field_names,
                # flags=(zmq.SNDMORE * (len(storage_fields) > 0)))

        # for i, sf in enumerate(storage_field_names):
            # self.send_array(storage_fields[sf],
                    # flags=zmq.SNDMORE * (i < (len(storage_fields)-1)))

    # def recv_model(self, flags=0):
        # meta = self.recv_json()

        # exec("import {} as model_module".format(meta["module"]))
        # model_t = getattr(model_module, meta["name"])

        # model_fields = self.recv_json()


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
            script_filename = self._setup_script_file()
            socket, address = self._setup_socket_host()

            process = self._spawn_process(script_filename, address)

            self._send_arguments(socket, args, kwargs)
            return_values = self._recv_returnvalue(socket)

            process.wait()
        except:
            raise
        finally:
            if process is not None and process.poll() is None:
                process.kill()
            if script_filename is not None:
                self._delete_script_file(script_filename)

        return return_values

    def _client(self, address):
        socket = self._setup_socket_client(address)
        args, kwargs = self._recv_arguments(socket)

        return_value = None
        try:
            return_value = self._func(*args, **kwargs)
        finally:
            self._send_returnvalue(socket, return_value)


    def _spawn_process(self, script_filename, address):
        log.debug("Spawning subprocess..")
        return sp.Popen([sys.executable, script_filename, address],
                cwd=self._func_dir)

    def _setup_socket_host(self):
        log.debug("Setting up host socket..")
        ctx = zmq.Context.instance()

        socket = SerializingSocket(ctx, zmq.REQ)
        socket.bind("ipc://*")
        address = socket.getsockopt(zmq.LAST_ENDPOINT)

        return socket, address

    def _setup_socket_client(self, address):
        log.debug("Setting up client socket..")
        ctx = zmq.Context.instance()

        socket = SerializingSocket(ctx, zmq.REP)
        socket.connect(address)

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
        args_transmitter = self._get_transmitter_iter(args)
        kwargs_transmitter = self._get_transmitter_dict(kwargs)

        socket.send_json(args_transmitter, flags=zmq.SNDMORE)
        socket.send_json(kwargs_transmitter, flags=zmq.SNDMORE)

        for arg, trans in it.izip(args, args_transmitter):
            getattr(socket, "send_"+trans)(arg, flags=zmq.SNDMORE)

        for k in kwargs.iterkeys():
            v = kwargs[k]
            trans = kwargs_transmitter[k]

            socket.send(trans, flags=zmq.SNDMORE)
            socket.send(k, flags=zmq.SNDMORE)
            getattr(socket, "send_"+trans)(v, flags=zmq.SNDMORE)

        # send an empty string to indicate we are finished
        socket.send("")

    def _recv_arguments(self, socket):
        log.debug("Receiving arguments.")
        args_transmitter = socket.recv_json()
        kwargs_transmitter = socket.recv_json()

        args = []
        for trans in args_transmitter:
            args.append(getattr(socket, "recv_"+trans)())

        kwargs = {}
        while True:
            trans = socket.recv()
            if trans == "":
                break
            k = socket.recv()
            v = getattr(socket, "recv_"+trans)()

            kwargs[k] = v

        return args, kwargs

    def _send_returnvalue(self, socket, retval):
        log.debug("Sending return value.")
        if isinstance(retval, tuple):
            retval_was_tuple = True
        else:
            retval = (retval,)
            retval_was_tuple = False

        retval_transmitters = self._get_transmitter_iter(retval)
        socket.send_json(retval_transmitters, flags=zmq.SNDMORE)

        for i, (rv, trans) in enumerate(it.izip(retval, retval_transmitters)):
            getattr(socket, "send_"+trans)(rv, flags=zmq.SNDMORE)

        socket.send_zipped_pkl(retval_was_tuple)

    def _recv_returnvalue(self, socket):
        log.debug("Receiving return value.")
        retval_transmitters = socket.recv_json()

        retval = []
        for trans in retval_transmitters:
            retval.append(getattr(socket, "recv_"+trans)())

        retval_was_tuple = socket.recv_zipped_pkl()

        if retval_was_tuple:
            retval = tuple(retval)
        else:
            retval = retval[0]

        return retval

    def _get_module_import_name(self):
        if self._func_module != "__main__":
            return self._func_module
        else:
            module_path = osp.abspath(sys.modules[self._func_module].__file__)
            module_path = osp.basename(module_path)
            return osp.splitext(module_path)[0]

    def _setup_script_file(self):
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
        script.write("target_module.{}._client(sys.argv[1])\n".format(
            self._func_name))

        script.close()

        return filename

    def _delete_script_file(self, script_filename):
        try:
            os.remove(script_filename)
            log.debug("Deleted script file.")
        except OSError:
            log.warn("Could not delete temporary script file for subprocess.")


# communication methods (pyzmq example)
# TODO: Delme once all references are deleted (now in serializing context)
def send_array(socket, arr, flags=0, copy=True, track=False):
    """send a numpy array with metadata"""
    md = dict(
        dtype = str(arr.dtype),
        shape = arr.shape,
    )
    socket.send_json(md, flags|zmq.SNDMORE)
    return socket.send(arr, flags, copy=copy, track=track)

def recv_array(socket, flags=0, copy=True, track=False):
    """recv a numpy array"""
    md = socket.recv_json(flags=flags)
    msg = socket.recv(flags=flags, copy=copy, track=track)
    buf = buffer(msg)
    arr = np.frombuffer(buf, dtype=md['dtype'])
    return arr.reshape(md['shape'])


def send_src_cfg(socket, db_sources, flags=0):
    source_cfg = [{"rate": src.rate, "weight": src.weight,
        "is_exc": src.is_exc, "has_spikes": src.has_spikes}\
                for src in db_sources]

    # send spike arrays for all sources without rate
    sources_with_spike_times = filter(lambda x: x.has_spikes, db_sources)
    num_sources_with_spike_times = len(sources_with_spike_times)

    # first send the the source_cfg
    socket.send_json(source_cfg,
            flags=flags|(zmq.SNDMORE * (num_sources_with_spike_times > 0)))

    for i, src in enumerate(sources_with_spike_times):
        send_array(socket, np.array(src.spike_times),
                flags=zmq.SNDMORE*(i<(num_sources_with_spike_times-1)),
                copy=True)

def recv_src_cfg(socket, flags=0):
    source_cfg = socket.recv_json()

    for src in source_cfg:
        if src["has_spikes"]:
            src["spike_times"] = recv_array(socket)

    return source_cfg


