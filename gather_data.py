#!/usr/bin/env python
# encoding: utf-8

"""
    Gather calibration data in another process.
"""

import sys
import os.path as osp
import functools as ft
import numpy as np
import subprocess as sp
import itertools as it

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

# NOTE: No relative imports here because the file will also be executed as
#       script.
from . import comm
from .logcfg import log
from . import buildingblocks as bb


def log_time(time, duration, num_steps=10, offset=0):
    log.info("{} ms / {} ms".format(time-offset, duration))
    return time + duration/num_steps


def get_callbacks(sim, log_time_params):

    if sim.__name__.split(".")[-1] == "neuron":
        # neuron does not wörk with callbacks
        callbacks = []
    else:
        callbacks = [ft.partial(log_time, **log_time_params)]

    return callbacks

@comm.RunInSubprocess
def gather_calibration_data(sim_name, calib_cfg, neuron_model,
        neuron_params, sources_cfg):
    """
        This function performs a single calibration run and should normally run
        in a seperate subprocess (which it is when called from LIFsampler).

        It does not fit the sigmoid.

        sim_name: name of the simulator module
        calib_cfg: All non-None keys in Calibration-Model (duration, dt etc.)
        neuron_model: name of the used neuron model
        neuron_params: name of the parameters for the neuron
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

    callbacks = get_callbacks(sim, {
            "duration" : calib_cfg["duration"],
            "offset" : burn_in_time,
        })

    # bring samplers into high conductance state
    log.info("Burning in samplers for {} ms".format(burn_in_time))
    sim.run(burn_in_time)
    log.info("Generating calibration data..")
    sim.run(duration, callbacks=callbacks)

    log.info("Reading spikes.")
    spiketrains = samplers.get_data("spikes").segments[0].spiketrains
    num_spikes = np.array([(s > burn_in_time).sum() for s in spiketrains],
            dtype=int)

    samples_p_on = num_spikes * neuron_params["tau_refrac"] / duration

    return samples_v_rest, samples_p_on


@comm.RunInSubprocess
def gather_free_vmem_trace(distribution_params, neuron_model,
                neuron_params, sources_cfg, sim_name, adjusted_v_thresh=50.):
    """
        Records a voltage trace of the free membrane potential of the given
        neuron model with the given parameters.

        adjusted_v_tresh is the value the neuron-threshold will be set to
        to avoid spiking.
    """
    dp = distribution_params
    log.info("Preparing to take free Vmem distribution")
    exec("import {} as sim".format(sim_name))

    sim.setup(time_step=dp["dt"])

    sources = bb.create_sources(sim, sources_cfg, dp["duration"])

    population = sim.Population(1, getattr(sim, neuron_model)(**neuron_params))
    population.record("v")
    population.initialize(v=neuron_params["v_rest"])
    population.set(v_thresh=adjusted_v_thresh)

    projections = bb.connect_sources(sim, sources_cfg, sources, population)

    callbacks = get_callbacks(sim, {
            "duration" : dp["duration"],
        })

    log.info("Starting data gathering run.")
    sim.run(dp["duration"], callbacks=callbacks)

    data = population.get_data("v")

    voltage_trace = np.array(data.segments[0].analogsignalarrays[0])[:, 0]

    return voltage_trace


