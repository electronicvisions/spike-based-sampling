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
