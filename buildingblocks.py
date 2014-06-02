#!/usr/bin/env python
# encoding: utf-8

from .logcfg import log

import functools as ft
import itertools as it
import numpy as np

from pprint import pformat as pf


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


def create_nest_optimized_sources(sim, samplers, population, duration,
        source_model=None, source_model_kwargs=None):
    """
        Create the least number of poisson_generator type specific to python
        and connecte them to the samplers.

        Afterwards, all other sources are created with `create_sources` and
        `connect_sources`.

        source_(model/kwargs) can be used to specify a diffferent source model.
    """
    log.info("Applying NEST-specific optimization in source creation.")

    if source_model is None:
        source_model = "poisson_generator"

    if source_model_kwargs is None:
        source_model_kwargs = {}

    sources = {}
    projections = {}
    # _ps = _per_sampler
    all_source_cfgs_ps = [sampler.get_sources_cfg_lod() for sampler in samplers]

    # create poisson sources
    has_no_spikes = lambda x: not x["has_spikes"]
    poisson_cfgs_ps = [filter(has_no_spikes, srcs)
            for srcs in all_source_cfgs_ps]

    num_poisson_ps = np.array([len(srcs) for srcs in poisson_cfgs_ps])

    if num_poisson_ps.sum() > 0:
        # we want a mapping from each samplers sources into a large flattened
        # array
        offset_ps = np.cumsum(num_poisson_ps) - num_poisson_ps[0]
        rates = [src["rate"] for srcs in poisson_cfgs_ps for src in srcs]
        rates = np.array(rates)

        uniq_rates, idx_to_src_id = np.unique(rates, return_inverse=True)

        log.info("Creating {} different poisson sources.".format(
            uniq_rates.size))
        poisson_gen_t = sim.native_cell_type(source_model)
        sources["poisson"] = sim.Population(uniq_rates.size,
                poisson_gen_t(start=0., stop=duration, **source_model_kwargs))

        sources["poisson"].set(rate=uniq_rates)

        log.info("Connecting poisson sources to samplers.")
        # manage which source is connected to what sampler
        connections = {"exc": [], "inh": []}
        conn_type = ["inh", "exc"]

        for i, source_cfgs in enumerate(poisson_cfgs_ps):
            for j, src_cfg in enumerate(source_cfgs):
                connections[conn_type[src_cfg["is_exc"]]].append(
                        (idx_to_src_id[offset_ps[i]+j], i, src_cfg["weight"]))

        projections["poisson"] = poiss_proj = {}
        for ct in conn_type:
            poiss_proj[ct] = sim.Projection(
                    sources["poisson"], population,
                    receptor_type={"exc":"excitatory", "inh":"inhibitory"}[ct],
                    synapse_type=sim.StaticSynapse(),
                    connector=sim.FromListConnector(connections[ct],
                        column_names=["weight"]))

    # create the other sources
    has_spikes = lambda x: x["has_spikes"]
    array_cfgs_ps = [filter(has_spikes, srcs) for srcs in all_source_cfgs_ps]

    num_array_ps = [len(srcs) for srcs in poisson_cfgs_ps]

    if sum(num_array_ps) > 0:
        log.info("Creating array sources for each sampler.")

        sources["array_per_sampler"] = []
        projections["array_per_sampler"] = []

        for i, sources_cfg in enumerate(array_cfgs_ps):
            if len(sources_cfg) > 0:
                local_pop = population[i:i+1]
                local_src = create_sources(sim, sources_cfg, duration)
                local_proj = connect_sources(
                        sim, sources_cfg, local_src, local_pop)

            else:
                local_src = {}
                local_proj = {}

            sources["array_per_sampler"].append(local_src)
            projections["array_per_sampler"].append(local_proj)

    return sources, projections

