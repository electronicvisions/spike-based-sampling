#!/usr/bin/env python
# encoding: utf-8

from .logcfg import log

from . import db

import itertools as it
import numpy as np


def create_sources(sim, source_config, duration):
    """
        DEPRECATED!

        Create sources from a list of source specifications
    """
    sources = None
    # Note: poisson_generator takes "stop", spike source poisson takes
    # "duration"!
    if isinstance(source_config, db.PoissonSourceConfiguration):
        source_params = {"start": 0.}
        source_t = sim.SpikeSourcePoisson
        source_params["duration"] = duration

        log.info("Setting up Poisson sources.")
        rates = source_config.rates
        sources = sim.Population(len(rates), source_t(**source_params))

        for src, rate in it.izip(sources, rates):
            src.rate = rate

        num_sources = len(rates)
        log.info("Created {} sources.".format(num_sources))

    elif isinstance(source_config, db.FixedSpikeTrainConfiguration):
        # TODO: test me
        log.info("Setting up fixed spike train sources.")
        weights = source_config.weights
        spike_times = source_config.spike_times
        spike_ids = source_config.spike_ids
        rates = source_config.rates

        num_sources = len(weights)
        sources = sim.Population(num_sources, sim.SpikeSourceArray())
        for i, src in enumerate(sources):
            local_spike_times = spike_times[spike_ids == i]
            src.spike_times = local_spike_times
            src.rate = rates[i]

        log.info("Created {} fixed spike train sources.".format(num_sources))

    else:
        log.error("Source configuration of type {} unkown.".format(
            source_config.__class__.__name__))

    return sources


def connect_sources(sim, source_config, sources, target):
    """
        Connect the `sources` created from `sources_cfg` to target.
    """
    projections = None

    if isinstance(source_config, db.PoissonSourceConfiguration):
        projections = {}

        column_names = ["weight"]

        is_exc = np.array(source_config.weights > 0., dtype=int)

        receptor_types = ["inhibitory", "excitatory"]

        for i_r, rectype in enumerate(receptor_types):
            conn_list = []
            idx = is_exc == i_r

            for i, weight in it.izip(np.where(idx)[0],
                                     source_config.weights[idx]):
                for j in xrange(len(target)):
                    conn_list.append((i, j, np.abs(weight)))

            projections[rectype] = sim.Projection(
                sources, target,
                sim.FromListConnector(conn_list, column_names=column_names),
                synapse_type=sim.StaticSynapse(),
                receptor_type=rectype)

    elif isinstance(source_config, db.FixedSpikeTrainConfiguration):
        # TODO: test me
        projections = {}

        column_names = ["weight"]

        is_exc = np.array(source_config.weights > 0., dtype=int)

        receptor_types = ["inhibitory", "excitatory"]

        for i_r, rectype in enumerate(receptor_types):
            conn_list = []
            idx = is_exc == i_r

            for i, weight in it.izip(np.where(idx)[0],
                                     source_config.weights[idx]):
                for j in xrange(len(target)):
                    conn_list.append((i, j, np.abs(weight)))

            projections[rectype] = sim.Projection(
                sources, target,
                sim.FromListConnector(conn_list, column_names=column_names),
                synapse_type=sim.StaticSynapse(),
                receptor_type=rectype)

    else:
        log.error("Source configuration of type {} unkown.".format(
            source_config.__class__.__name__))

    return projections


def create_nest_optimized_sources(
        sim, samplers, population, duration,
        source_model=None, source_model_kwargs=None):
    """
        DEPRECATED!

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
    for sampler in samplers:
        assert isinstance(
                sampler.calibration.source_config,
                db.PoissonSourceConfiguration)

    num_sources_per_sampler = np.array(
            (len(s.calibration.source_config.rates) for s in samplers))

    # we want a mapping from each samplers sources into a large flattened
    # array
    offset_per_sampler = np.r_[np.cumsum(num_sources_per_sampler)]

    def id_to_sampler(idx):
        sampler = 0
        idx -= offset_per_sampler[sampler]
        while idx > 0:
            sampler += 1
            idx -= offset_per_sampler[sampler]
        return sampler

    rates = np.hstack((s.calibration.source_config.rates for s in samplers))

    uniq_rates, idx_to_src_id = np.unique(rates, return_inverse=True)

    log.info("Creating {} different poisson sources.".format(
        uniq_rates.size))
    poisson_gen_t = sim.native_cell_type(source_model)
    sources = sim.Population(
            uniq_rates.size,
            poisson_gen_t(start=0., stop=duration, **source_model_kwargs))

    sources.set(rate=uniq_rates)

    log.info("Connecting poisson sources to samplers.")
    # manage which source is connected to what sampler
    connections = {"exc": [], "inh": []}
    conn_type = ["inh", "exc"]

    cur_source = 0
    for i, sampler in enumerate(samplers):
        for j, weight in enumerate(sampler.calibration.source_config.weights):
            connections[conn_type[weight > 0]].append(
                    (idx_to_src_id[cur_source], i, np.abs(weight)))
            cur_source += 1

    projections = {}
    for ct in conn_type:
        projections[ct] = sim.Projection(
                sources, population,
                receptor_type={"exc": "excitatory", "inh": "inhibitory"}[ct],
                synapse_type=sim.StaticSynapse(),
                connector=sim.FromListConnector(
                    connections[ct],
                    column_names=["weight"]))

    return sources, projections
