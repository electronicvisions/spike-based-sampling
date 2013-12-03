#!/usr/bin/env python
# encoding: utf-8

"""
    Gather calibration data in another process.
"""

import sys
import os.path as osp
import functools as ft

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

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

def gather_calibration_data(db_neuron_params, db_partial_calibration, db_sources,
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
            "sources_cfg" : [{"rate": src.rate, "weight": src.weight,
                "is_exc": src.is_exc} for src in db_sources]
        }

    socket.send_json(cfg_data)

    dbpc.samples_v_rest = comm.recv_array(socket)
    dbpc.samples_p_on = comm.recv_array(socket)

    proc.wait()


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

    samples_v_rest = np.linspace(mean-spread, mean+spread, num)

    rates = np.array([src_cfg["rate"] for src_cfg in sources_cfg])

    # TODO maybe implement a seed here
    sim.setup(time_step=calib_cfg["dt"])

    # create sources
    if hasattr(sim, "nest"):
        source_t = sim.native_cell_type("poisson_generator")
    else:
        source_t = sim.SpikePoissonGenerator

    sources = sim.Population(len(rates), source_t(
        stop=calib_cfg["duration"], start=0.))

    for src, rate in it.izip(sources, rates):
        src.rate = rate

    log.info("Setting up {} samplers.".format(num))
    samplers = sim.Population(num, getattr(sim, neuron_model)(**neuron_params))
    samplers.record("spikes")
    samplers.initialize(v=samples_v_rest)
    samplers.set(v_rest=samples_v_rest)

    # connect the two
    projections = []

    for i, src_cfg in enumerate(sources_cfg):
        src = sources[i:i+1] # get a population view because only those can
                               # be connected
        projections.append(sim.Projection(src, samplers,
            sim.AllToAllConnector(),
            synapse_type=sim.StaticSynapse(weight=src_cfg["weight"]),
            receptor_type=["inhibitory", "excitatory"][src_cfg["is_exc"]]))

    steps = 10
    increment = duration / steps

    # bring samplers into high conductance state
    log.info("Burning in samplers for {} ms".format(burn_in_time))
    sim.run(burn_in_time)
    log.info("Generating calibration data..")
    sim.run(duration,
            callbacks=[ft.partial(log_time,
                increment=increment, duration=calib_cfg["duration"],
                offset=burn_in_time)])

    log.info("Reading spikes.")
    spiketrains = samplers.get_data("spikes").segments[0].spiketrains
    num_spikes = np.array([(s > burn_in_time).sum() for s in spiketrains],
            dtype=int)

    samples_p_on = num_spikes * neuron_params["tau_refrac"] / duration

    return samples_v_rest, samples_p_on


def log_time(time, increment, duration, offset):
    log.info("{} ms / {} ms".format(time-offset, duration))
    return time + increment


def _client_calibration_gather_data(address):
    """
        This function is meant to be run by the spawned calibration process.
    """
    socket = context.socket(zmq.REP)
    socket.connect(address)

    cfg_data = socket.recv_json()

    samples_v_rest, samples_p_on =\
            standalone_gather_calibration_data(**cfg_data)

    comm.send_array(socket, samples_v_rest, flags=zmq.SNDMORE)
    comm.send_array(socket, samples_p_on, flags=0)


if __name__ == "__main__":
    _client_calibration_gather_data(sys.argv[1])

