#!/usr/bin/env python
# encoding: utf-8

from .logcfg import log

import functools as ft
import itertools as it
import numpy as np


def create_sources(sim, sources_cfg, duration):
    """
        Create sources from a list of source specifications
    """
    sources_poisson = None
    sources_array  = None

    sources_poisson_cfg = filter(lambda s: not s["has_spikes"], sources_cfg)
    sources_array_cfg = filter(lambda s: s["has_spikes"], sources_cfg)

    # Note: poisson_generator takes "stop", spike source poisson takes
    # "duration"!
    if hasattr(sim, "nest"):
        source_t = ft.partial(sim.native_cell_type("poisson_generator"),
                stop=duration)
    else:
        source_t = ft.partial(sim.SpikeSourcePoisson, duration=duration)

    if len(sources_poisson_cfg) > 0:
        log.info("Setting up Poisson sources.")
        rates = np.array([src_cfg["rate"] for src_cfg in sources_array_cfg])
        sources_poisson = sim.Population(len(sources_poisson_cfg),
                source_t(start=0.))

        for src, rate in it.izip(sources_poisson, rates):
            src.rate = rate

    if len(sources_array_cfg) > 0:
        log.info("Setting up spike array sources.")
        sources_array = sim.Population(len(sources_array_cfg),
                sim.SpikeSourceArray())
        for src, src_cfg in it.izip(sources_array, sources_array_cfg):
            src.spike_times = src_cfg["spike_times"]

    num_sources = 0
    if sources_poisson is not None:
        num_sources += len(sources_poisson)
    if sources_array is not None:
        num_sources += len(sources_array)
    log.info("Created {} sources.".format(num_sources))

    return {"poisson": sources_poisson, "array": sources_array}


def connect_sources(sim, sources_cfg, sources, target):
    """
        Connect the `sources` created from `sources_cfg` to target.
    """
    sources_poisson_cfg = filter(lambda s: not s["has_spikes"], sources_cfg)
    sources_array_cfg = filter(lambda s: s["has_spikes"], sources_cfg)

    projections_poisson = []
    projections_array = []

    if len(sources_poisson_cfg) > 0:
        log.info("Connecting samplers to Poisson sources.")
        for i, src_cfg in enumerate(sources_poisson_cfg):
            # get a population view because only those can be connected
            src = sources["poisson"][i:i+1]
            projections_poisson.append(sim.Projection(src, target,
                sim.AllToAllConnector(),
                synapse_type=sim.StaticSynapse(weight=src_cfg["weight"]),
                receptor_type=["inhibitory", "excitatory"][src_cfg["is_exc"]]))

    if len(sources_array_cfg) > 0:
        log.info("Connecting samplers to spike array sources.")
        for i, src_cfg in enumerate(sources_array_cfg):
            # get a population view because only those can be connected
            src = sources["array"][i:i+1]
            projections_array.append(sim.Projection(src, target,
                sim.AllToAllConnector(),
                synapse_type=sim.StaticSynapse(weight=src_cfg["weight"]),
                receptor_type=["inhibitory", "excitatory"][src_cfg["is_exc"]]))

    num_synapses = sum((len(proj) for proj in\
            projections_poisson+projections_array))

    return {"poisson": projections_poisson, "array": projections_array}

