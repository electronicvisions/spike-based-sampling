#!/usr/bin/env python
# encoding: utf-8

"""
    The whole calibration done in async fashion.

"""

# NOTE: No relative imports here because the file will also be executed as
#       script.
from sbs import comm
from sbs.logcfg import log

import zmq
import numpy as np
import subprocess as sp
import itertools as it
import os.path as osp
import sys

context = zmq.Context()

def calibrate_sampler(db_neuron_params, db_partial_calibration, db_sources,
        sim_name="pyNN.nest"):
    """
        db_neuron_params:
            NeuronParameters instance that contains the parameters to use.

        db_partial_calibration:
            Calibration instance holding configuration parameters
            (duration, num_samples).

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
            "calib_cfg" : {k: getattr(dbpc, k) for k in\
                    ["duration", "num_samples", "std_range", "mean", "std"]},
            "neuron_model" : db_neuron_params.pynn_model,
            "neuron_params" : db_neuron_params.get_pynn_parameters(),
            "sources_cfg" : [{"rate": src.rate, "weight": src.weight,
                "is_exc": src.is_exc} for src in db_sources]
        }

    socket.send_json(cfg_data)

    dbpc.samples_v_rest = comm.recv_array(socket)
    dbpc.samples_p_on = comm.recv_array(socket)


def standalone_calibration(sim_name, calib_cfg, neuron_model, neuron_params,
        sources_cfg):
    """
        This function performs a single calibration run and should normally run
        in a seperate subprocess (which it is when called from LIFsampler).
    """
    log.info("Calibration started.")
    log.info("Preparing network.")
    exec("import {} as sim".format(sim_name))

    spread = calib_cfg["std_range"] * calib_cfg["std"]
    mean = calib_cfg["mean"]

    num = calib_cfg["num_samples"]

    samples_v_rest = np.linspace(mean-spread, mean+spead, num)

    # TODO maybe implement a seed here
    sim.setup()

    samplers = sim.Population(num, getattr(sim, neuron_model)(**neuron_params))
    samplers.record("spikes")

    for sampler, v_rest in it.izip(samplers, samples_v_rest):
        sampler.set(v_rest=v_rest)

    log.info("Running simulation..")
    sim.run(calib_cfg["duration"])

    log.info("Reading spikes.")
    spikes = samplers.get_data("spikes")
    spikes = np.array([len(s) for s in spikes.segments[0].spiketrains],
            dtype=int)

    samples_p_on = spikes * neuron_params["tau_refrac"] / calib_data["duration"]

    return samples_v_rest, samples_p_on


def _client_calibration(self, address):
    """
        This function is meant to be run by the spawned calibration process.
    """
    socket = context.socket(zmq.REP)
    socket.connect(address)

    cfg_data = socket.get_json()

    samples_v_rest, samples_p_on = standalone_calibration(**cfg_data)

    comm.send_array(socket, samples_v_rest, flags=zmq.SNDMORE)
    comm.send_array(socket, samples_p_on, flags=0)

if __name__ == "__main__":
    _client_calibration(sys.argv[1])
