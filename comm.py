#!/usr/bin/env python
# encoding: utf-8

import zmq
import numpy as np

# communication methods (pyzmq example)
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


