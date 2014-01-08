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
    source_params = {"start" : 0.}
    if hasattr(sim, "nest"):
        source_t = sim.native_cell_type("poisson_generator")
        source_params["stop"] = duration
    else:
        source_t = sim.SpikeSourcePoisson
        source_params["duration"] = duration

    if len(sources_poisson_cfg) > 0:
        log.info("Setting up Poisson sources.")
        rates = np.array([src_cfg["rate"] for src_cfg in sources_poisson_cfg])
        sources_poisson = sim.Population(len(sources_poisson_cfg),
                source_t(**source_params))

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

    projections = {}

    src_types = []
    if len(sources_poisson_cfg) > 0:
        src_types.append("poisson")
    if len(sources_array_cfg) > 0:
        src_types.append("array")

    for st in src_types:
        log.info("Connecting samplers to {} sources.".format(st))

        local_projections = projections.setdefault(st, [])
        for i, src_cfg in enumerate(sources_poisson_cfg):
            # get a population view because only those can be connected
            src = sources[st][i:i+1]
            local_projections.append(sim.Projection(src, target,
                sim.AllToAllConnector(),
                synapse_type=sim.StaticSynapse(weight=src_cfg["weight"]),
                receptor_type=["inhibitory", "excitatory"][src_cfg["is_exc"]]))

    num_synapses = sum((len(proj) for proj in projections.itervalues()))
    log.info("Sources -> target synapse count: {}".format(num_synapses))

    return projections

