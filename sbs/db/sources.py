#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import itertools as it
import collections as c

from ..logcfg import log
from .core import Data

__all__ = [
        "sources_create_connect",
        "SourceConfiguration",
        "PoissonSourceConfiguration",
        "FixedSpikeTrainConfiguration",
    ]



def sources_create_connect(sim, samplers, duration, **kwargs):
    """
        Creates sources and connects them to the samplers.

        It seperates the samplers into subgroups that have the same type
        of source config.

        Returns a list of created sources.
    """
    sampler_same_src_cfg = [
            [samplers[0]]
        ]

    for s in samplers[1:]:
        if isinstance(s.source_config,
                sampler_same_src_cfg[-1][0].source_config.__class__):
            sampler_same_src_cfg[-1].append(s)
        else:
            sampler_same_src_cfg.append([s])

    results = []
    for l_samplers in sampler_same_src_cfg:
        results.append(l_samplers[0].source_config.create_connect(
            sim, l_samplers, duration, **kwargs))

    #  if len(results) == 1:
        #  return results[0]
    #  else:
    return results


class SourceConfiguration(Data):

    def create_connect(self, sim, samplers, **kwargs):
        """
            Shall create and connect the sources to the samplers.

            Should return the tuple (sources, projections).

            The type of projections should be a dictionary.

            Both sources and projections can be None if they
            are created internally in the simulator.
        """
        raise NotImplementedError

# helper functions 
def connect_one_to_all(sim, sources, samplers, weights):
    """
        BROKEN, DO NOT USE

        Connect each source from `sources` with the corresponding weight from
        `weights` to all `samplers`.
    """
    raise NotImplementedError

    projections = {}

    column_names = ["weight"]

    is_exc = np.array(weights > 0., dtype=int)

    receptor_types = ["inhibitory", "excitatory"]

    for i_r, rectype in enumerate(receptor_types):
        conn_list = []
        idx = is_exc == i_r

        for i, weight in it.izip(np.where(idx)[0], weights[idx]):
            for j in xrange(len(samplers)):
                conn_list.append((i, j, np.abs(weight)))

        projections[rectype] = sim.Projection(sources, samplers,
            sim.FromListConnector(conn_list, column_names=column_names),
            synapse_type=sim.StaticSynapse(),
            receptor_type=rectype)

    return projections

def connect_one_to_one(sim, sources, population, weights):
    """
        `sources` should be a list of lists of sources to be
        connected to each sampler.

        `population` is either a list of population or a population.

        `weights` should have the same shape as `sources` (list of numpy
        arrays).
    """
    projections = c.defaultdict(list)

    column_names = ["weight"]

    is_exc = np.array(weights > 0., dtype=int)

    receptor_types = ["inhibitory", "excitatory"]

    for j, (s_weights, s_sources) in enumerate(it.izip(
            weights, sources)):
        is_exc = np.array(s_weights > 0., dtype=int)

        if isinstance(population, sim.Population):
            pop = population[j:j+1]
        else:
            pop = population[j]

        for i_r, rectype in enumerate(receptor_types):
            conn_list = []

            idx = is_exc == i_r

            for i, weight in it.izip(np.where(idx)[0], s_weights[idx]):
                conn_list.append((i, j, np.abs(weight) if pop.conductance_based
                    else weight))

            projections[rectype].append(sim.Projection(s_sources,
                pop,
                sim.FromListConnector(conn_list, column_names=column_names),
                synapse_type=sim.StaticSynapse(),
                receptor_type=rectype))

    return projections

def get_population_from_samplers(sim, samplers):
    """
        Please note that this function assumes
        that all samplers are adjacent to each other in
        the corresponding network population.
    """
    if isinstance(samplers, sim.Population):
        return samplers
    elif len(samplers) == 1:
        return samplers[0].population
    elif samplers[0].network["population"] is None:
        return [s.population for s in samplers]
    elif samplers[0].network["index"] == 0\
            and samplers[-1].network["index"]-1 == len(samplers):
        return samplers[0].network["population"]
    else:
        return samplers[0].network["population"][
                samplers[0].network["index"]:samplers[-1].network["index"]]


class PoissonSourceConfiguration(SourceConfiguration):
    """
        Positive weights: excitatory
        Negative weights: inhibitory
    """
    data_attribute_types = {
            "rates" : np.ndarray,
            "weights" : np.ndarray,
        }

    def create_connect(self, sim, samplers, duration, nest_optimized=True,
            **kwargs):
        """
            Shall create and connect the sources to the samplers.
        """
        # we need to distinguish three cases:
        # whether we are connecting to a regular population (calibration etc)
        # or to a list of samplers that a) have the same rates or b) have
        # different rates

        if hasattr(sim, "nest") and nest_optimized:
            sources, projections = self.create_nest_optimized(
                    sim, samplers, duration)

        else:
            sources = self.create_regular(sim, samplers, duration)
            population = get_population_from_samplers(sim, samplers)

            if isinstance(samplers, sim.Population):
                weights = [self.weights for s in samplers]
            else:
                weights = [s.source_config.weights for s in samplers]

            projections = connect_one_to_one(sim, sources, population,
                    weights=weights)

        return sources, projections

    def create_regular(self, sim, samplers, duration):
        source_params = {"start" : 0.}
        source_t = sim.SpikeSourcePoisson
        source_params["duration"] = duration

        log.info("Setting up Poisson sources.")

        if isinstance(samplers, sim.Population):
            rates = [self.rates] * len(samplers)
        else:
            rates = [s.source_config.rates for s in samplers]

        sources = [sim.Population(len(r), source_t(**source_params))
                for r in rates]

        for s_sources, l_rates in it.izip(sources, rates):
            for src, rate in it.izip(s_sources, l_rates):
                src.rate = rate

        num_sources = len(rates) * len(samplers)
        log.info("Created {} sources.".format(num_sources))

        return sources

    def create_nest_optimized(self, sim, samplers, duration):
        log.info("Applying NEST-specific optimization in source creation.")

        if "lookahead_poisson_generator" in sim.nest.Models():
            source_model = "lookahead_poisson_generator"
            source_model_kwargs = {
                    "steps_lookahead" : 10000
                }
        else:
            source_model = "poisson_generator"
            source_model_kwargs = {}

        sources = {}
        projections = {}
        # _ps = _per_sampler
        if not isinstance(samplers, sim.Population):
            num_sources_per_sampler = np.array((len(s.calibration.source_config.rates)
                for s in samplers))
            rates = np.hstack((s.calibration.source_config.rates for s in samplers))
            weights = (s.calibration.source_config.weights for s in samplers)

        else:
            # if we have one population all get the same sources
            num_sources_per_sampler = np.zeros(len(samplers) * len(self.rates),
                    dtype=int) + len(self.rates)
            rates = np.hstack((self.rates for s in samplers))
            weights = it.repeat(self.weights)

        # we want a mapping from each samplers sources into a large flattened
        # array
        offset_per_sampler = np.r_[np.cumsum(num_sources_per_sampler)]
        def id_to_sampler(idx):
            sampler=0
            idx -= offset_per_sampler[sampler]
            while idx > 0:
                sampler += 1
                idx -= offset_per_sampler[sampler]
            return sampler

        uniq_rates, idx_to_src_id = np.unique(rates, return_inverse=True)

        log.info("Creating {} different poisson sources.".format(
            uniq_rates.size))
        poisson_gen_t = sim.native_cell_type(source_model)
        sources = sim.Population(uniq_rates.size,
                poisson_gen_t(start=0., stop=duration, **source_model_kwargs))

        sources.set(rate=uniq_rates)

        log.info("Connecting poisson sources to samplers.")
        # manage which source is connected to what sampler
        connections = {"exc": [], "inh": []}
        conn_type = ["inh", "exc"]

        population = get_population_from_samplers(sim, samplers)

        cur_source = 0
        for i, (sampler, weights) in enumerate(it.izip(samplers, weights)):
            for j, weight in enumerate(weights):
                connections[conn_type[weight > 0]].append(
                        (idx_to_src_id[cur_source], i, np.abs(weight)
                            if population.conductance_based else weight))
                cur_source += 1

        projections = {}
        for ct in conn_type:
            projections[ct] = sim.Projection(
                    sources, population,
                    receptor_type={"exc":"excitatory", "inh":"inhibitory"}[ct],
                    synapse_type=sim.StaticSynapse(),
                    connector=sim.FromListConnector(connections[ct],
                        column_names=["weight"]))

        return sources, projections


class FixedSpikeTrainConfiguration(SourceConfiguration):
    """
        Positive weights: excitatory
        Negative weights: inhibitory

        Spike times in ms.
    """
    data_attribute_types = {
            "rates" : np.ndarray, # for theoretical calculations
            "weights" : np.ndarray,
            "spike_times" : np.ndarray,
            "spike_ids" : np.ndarray,
        }

    def create_connect(self, sim, samplers, **kwargs):
        """
            Shall create and connect the sources to the samplers.
        """
        sources = self.create(sim)
        self.connect(sim, sources, samplers)

        return sources, projections

    def create(self, sim):
        num_sources = len(self.weights)
        sources = sim.Population(num_sources, sim.SpikeSourceArray())
        for i, src in enumerate(sources):
            local_spike_times = self.spike_times[self.spike_ids == i]
            src.spike_times = local_spike_times
            src.rate = self.rates[i]

        log.info("Created {} fixed spike train sources.".format(num_sources)) 

        return sources


