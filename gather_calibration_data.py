#!/usr/bin/env python
# encoding: utf-8

"""
    Gather calibration data in another process.
"""

import sys
import os.path as osp
import functools as ft
import zmq
import numpy as np
import subprocess as sp
import itertools as it

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

# NOTE: No relative imports here because the file will also be executed as
#       script.
from sbs import comm
from sbs.logcfg import log
from sbs import buildingblocks as bb


context = zmq.Context()

def gather_calibration_data(db_neuron_params, db_partial_calibration,
        db_sources, sim_name="pyNN.nest"):
    """
        db_neuron_params:
            NeuronParameters instance that contains the parameters to use.

        db_partial_calibration:
            Calibration instance holding configuration parameters
            (duration, num_samples, etc).

        db_sources:
            List of sources to use for calibration.
    """
    dbpc = db_partial_calibration

    log.info("Preparing calibration..")
    log.debug("Setting up socket..")

    socket = context.socket(zmq.REQ)
    socket.bind("ipc://*")

    address = socket.getsockopt(zmq.LAST_ENDPOINT)

    log.debug("Spawning calibration process..")
    proc = sp.Popen([
        sys.executable, osp.abspath(__file__), address,
        ], cwd=osp.dirname(osp.abspath(__file__)))

    # prepare data to send
    cfg_data = {
            "sim_name" : sim_name,
            "calib_cfg" : {k: getattr(dbpc, k) for k in [
                "duration",
                "num_samples",
                "std_range",
                "mean",
                "std",
                "burn_in_time",
                "dt",
            ]},
            "neuron_model" : db_neuron_params.pynn_model,
            "neuron_params" : db_neuron_params.get_pynn_parameters(),
        }

    socket.send_json(cfg_data, flags=zmq.SNDMORE)
    comm.send_src_cfg(socket, db_sources)

    dbpc.samples_v_rest = comm.recv_array(socket)
    dbpc.samples_p_on = comm.recv_array(socket)


def _client_calibration_gather_data(address):
    """
        This function is meant to be run by the spawned calibration process.
    """
    socket = context.socket(zmq.REP)
    socket.connect(address)

    cfg_data = socket.recv_json()
    src_cfg = comm.recv_src_cfg(socket)

    samples_v_rest, samples_p_on =\
            standalone_gather_calibration_data(sources_cfg=src_cfg, **cfg_data)

    comm.send_array(socket, samples_v_rest, flags=zmq.SNDMORE)
    comm.send_array(socket, samples_p_on, flags=0)


def standalone_gather_calibration_data(sim_name, calib_cfg, neuron_model,
        neuron_params, sources_cfg):
    """
        This function performs a single calibration run and should normally run
        in a seperate subprocess (which it is when called from LIFsampler).

        It does not fit the sigmoid.

        calib_cfg: (dict with keys) duration, num_samples, mean, std, std_range
        sources_cfg: (list of dicts with keys) rate, weight, is_exc
    """
    log.info("Calibration started.")
    log.info("Preparing network.")
    exec("import {} as sim".format(sim_name))

    spread = calib_cfg["std_range"] * calib_cfg["std"]
    mean = calib_cfg["mean"]

    num = calib_cfg["num_samples"]

    burn_in_time = calib_cfg["burn_in_time"]
    duration = calib_cfg["duration"]
    total_duration = burn_in_time + duration

    samples_v_rest = np.linspace(mean-spread, mean+spread, num)

    # TODO maybe implement a seed here
    sim.setup(time_step=calib_cfg["dt"])

    # create sources
    sources = bb.create_sources(sim, sources_cfg, total_duration)

    log.info("Setting up {} samplers.".format(num))
    samplers = sim.Population(num, getattr(sim, neuron_model)(**neuron_params))
    samplers.record("spikes")
    samplers.initialize(v=samples_v_rest)
    samplers.set(v_rest=samples_v_rest)

    # connect the two
    projections = bb.connect_sources(sim, sources_cfg, sources, samplers)

    steps = 10
    increment = duration / steps

    # bring samplers into high conductance state
    log.info("Burning in samplers for {} ms".format(burn_in_time))
    sim.run(burn_in_time)
    log.info("Generating calibration data..")
    callbacks = [ft.partial(log_time,
         increment=increment, duration=calib_cfg["duration"],
         offset=burn_in_time)]

    if sim.__name__.split(".")[-1] == "neuron":
        # neuron does not wörk with callbacks
        callbacks = []
    sim.run(duration, callbacks=callbacks)

    log.info("Reading spikes.")
    spiketrains = samplers.get_data("spikes").segments[0].spiketrains
    num_spikes = np.array([(s > burn_in_time).sum() for s in spiketrains],
            dtype=int)

    samples_p_on = num_spikes * neuron_params["tau_refrac"] / duration

    return samples_v_rest, samples_p_on


def log_time(time, increment, duration, offset):
    log.info("{} ms / {} ms".format(time-offset, duration))
    return time + increment


if __name__ == "__main__":
    _client_calibration_gather_data(sys.argv[1])

