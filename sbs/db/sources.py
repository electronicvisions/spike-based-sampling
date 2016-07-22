#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import itertools as it
import collections as c

from ..logcfg import log
from .core import Data
from ..conversion import weight_pynn_to_nest

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
    sampler_same_src_cfg = it.groupby(
        samplers, lambda s: s.source_config.__class__)

    results = []
    for _, it_samplers in sampler_same_src_cfg:
        l_samplers = list(it_samplers)
        results.append(l_samplers[0].source_config.create_connect(
            sim, l_samplers, duration=duration, **kwargs))

    #  if len(results) == 1:
        #  return results[0]
    #  else:
    return results


class SourceConfiguration(Data):

    def get_distribution_parameters(self):
        """
            Return paramters needed for calculating the theoretical membrane
            distribution.
        """
        raise NotImplementedError

    def create_connect(self, sim, samplers, **kwargs):
        """
            Shall create and connect the sources to the samplers.

            samplers can be either a population of sampling neurons that all
            receive a similarly configured stimulus (typically used during
            calibration) or a list of actual sampler objects that all might have
            different source configuration (but of the same type).

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

        if isinstance(population, sim.common.BasePopulation):
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
        This is a helper function that streamlines the source creation process
        (DNRY ftw) because all source creation routines would have to perform
        the same if-else checks.

        It returns either a Population/-View if all samplers were
        created at once or a list of Populations/-Views if the
        samplers lie in different networks.

        The main goal is that after calling this function, the source creation
        routines can be sure to be dealing with PyNN-objects.

        Since the sources can be created both with LIFsampler objects and
        PyNN-Populations, we need a helper function that either returns the
        PyNN-Population or extracts the underlying objects from the samplers.

        We have to differentiate several cases: Samplers can be created in
        the same network or not. In the first case we are able to return a
        single Population/PopulationView, in the other case we have to return a
        list of Population/PopulationViews.

        TODO: Replace the list of Population/-Views with an Assembly if
        Assemblies work correctly now.

    """
    if isinstance(samplers, sim.common.BasePopulation):
        return samplers
    elif len(samplers) == 1:
        return samplers[0].population

    retval = []

    # we group the samplers by what PyNN-Population they belong too.
    for pop, ss in it.groupby(samplers, lambda s:s.network["population"]):
        if pop is None:
            retval.extend((s.population for s in ss))

        ss = list(ss)

        if len(ss) == len(samplers):
            # all samplers belong to the same population
            if len(ss) == len(pop):
                return pop
            else:
                return pop[np.array([s.network["index"] for s in samplers])]

        else:
            if len(ss) == len(pop):
                retval.append(pop)
            else:
                retval.append(
                        pop[np.array([s.network["index"] for s in samplers])])

    return retval



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

            if isinstance(samplers, sim.common.BasePopulation):
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

        if isinstance(samplers, sim.common.BasePopulation):
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
        if not isinstance(samplers, sim.common.BasePopulation):
            num_sources_per_sampler = np.array([len(s.source_config.rates)
                for s in samplers])
            rates = np.hstack((s.source_config.rates for s in samplers))
            weights = np.hstack((s.source_config.weights for s in samplers))

        else:
            # if we have one population all get the same sources
            num_sources_per_sampler = np.zeros(len(samplers),
                    dtype=int) + len(self.rates)
            rates = np.hstack((self.rates for s in samplers))
            weights = np.hstack((self.weights for s in samplers))
            if self.weights[0]*self.weights[1]>0:
                raise ValueError("Noise weights are both excitatory. Aborting.")

        # we want a mapping from each samplers sources into a large flattened
        # array
        offset_per_sampler = np.cumsum(num_sources_per_sampler)
        def idx_parrot_to_sampler(idx):
            """
                Parrot id to sampler id.
            """
            return np.where(idx < offset_per_sampler)[0][0]

        uniq_rates, idx_parrot_to_generator = np.unique(rates, return_inverse=True)

        log.info("Creating {} different poisson sources.".format(
            uniq_rates.size))

        # PyNN is once again trying to smart in some way when it comes to
        # native cell types (creating parrot_neurons in a non-optional way), so
        # we just do everything in CyNEST, FFS!
        import nest
        gid_generators = np.array(nest.Create(source_model, uniq_rates.size))

        # acting as sources
        gid_parrots = np.array(nest.Create("parrot_neuron", rates.size))

        connections = {
                "generator_to_parrot" : [],
                "parrot_to_sampler" : [],
            }

        nest.SetStatus(gid_generators.tolist(), [{
            "rate": r,
            "start": 0.,
            "stop": duration} for r in uniq_rates])
        if len(source_model_kwargs) > 0:
            nest.SetStatus(gid_generators.tolist(), source_model_kwargs)

        list_pop = get_population_from_samplers(sim, samplers)

        if isinstance(list_pop, sim.common.BasePopulation):
            list_pop = [list_pop]

        gid_samplers = np.hstack([p.all_cells.tolist() for p in list_pop])

        connections["generator_to_parrot"].append(
                nest.Connect(gid_generators[idx_parrot_to_generator].tolist(),
                    gid_parrots.tolist(), "one_to_one"))

        idx_samplers = np.array([idx_parrot_to_sampler(i_parrot)
            for i_parrot in xrange(gid_parrots.size)])
        connect_gid_samplers = gid_samplers[idx_samplers]

        connections["parrot_to_sampler"].append(
                nest.Connect(gid_parrots.tolist(),
                    connect_gid_samplers.tolist(),
                    "one_to_one",
                    {"weight": weight_pynn_to_nest(weights)}))

        sources = {
                "generators" : gid_generators,
                "parrots" : gid_parrots,
            }

        return sources, connections


    def get_distribution_parameters(self):
        """
            Return paramters needed for calculating the theoretical membrane
            distribution.
        """
        is_exc = self.weights > 0.
        is_inh = np.logical_not(is_exc)

        return {
            "rates_exc" : self.rates[is_exc],
            "rates_inh" : self.rates[is_inh],
            "weights_exc" : self.weights[is_exc],
            "weights_inh" : self.weights[is_inh],
        }


class FixedSpikeTrainConfiguration(SourceConfiguration):
    """
        Positive weights: excitatory
        Negative weights: inhibitory

        Rates are only used to compute weight translations etc and are not
        actually used when creating the spike sources.

        Spike times in ms.
    """
    data_attribute_types = {
            "rates" : np.ndarray,
            "weights" : np.ndarray,
            "spike_times" : np.ndarray,
            "spike_ids" : np.ndarray,
        }

    def create_connect(self, sim, samplers, **kwargs):
        """
            Shall create and connect the sources to the samplers.
        """
        # TODO: Implement improved NEST version
        return self.create_connect_regular(sim, samplers)


    def create_connect_regular(self, sim, samplers):
        #  Creates the different SpikeSourceArrays.
        population = get_population_from_samplers(sim, samplers)

        # list of numpy array with the corresponding spike times
        all_spike_times = [] 

        # all connection tuples
        conn_list = []

        # helper function to get the index of corresponding spike times
        def get_index(spike_times):
            # Assumes spike times are sorted
            for i, st in enumerate(all_spike_times):
                if spike_times.size == st.size and np.all(st == spike_times):
                    return i
            else:
                all_spike_times.append(spike_times)
                return len(all_spike_times)-1

        # note which source is connected to what samplers with what weight
        if isinstance(samplers, sim.common.BasePopulation):
            # all samplers receive same spikes
            for i, w in enumerate(self.weights):
                source_id = get_index(self.spike_times[self.spike_ids == i])
                for j in xrange(len(samplers)):
                    conn_list.append((source_id, j, w))
        else:
            # each sampelr might have different spike times
            for j, s in enumerate(samplers):
                sc = s.source_config
                for i, w in enumerate(sc.weights):
                    source_id = get_index(sc.spike_times[sc.spike_ids == i])
                    conn_list.append((source_id, j, w))

        sources = self.create_sources_regular(sim, all_spike_times)

        projections = self.connect_sources_regular(sim, sources, population,
                conn_list)

        return sources, projections

    def create_sources_regular(self, sim, spike_times):
        # create the unique sources
        num_sources = len(spike_times)
        sources = sim.Population(num_sources, sim.SpikeSourceArray())
        for src, st in it.izip(sources, spike_times):
            src.spike_times = st
        log.info("Created {} fixed spike train sources.".format(num_sources)) 
        return sources

    def connect_sources_regular(self, sim, sources, population, conn_list):
        connection_types = ["inh", "exc"]
        projections = {st: [] for st in connection_types}

        if isinstance(population, sim.common.BasePopulation):
            # one population for all samplers
            if population.conductance_based:
                trans = lambda w: np.abs(w)
            else:
                trans = lambda w: w

            conn_lists = [
                [(pre, post, trans(weight))
                    for pre, post, weight in conn_list if weight <  0],
                [(pre, post, trans(weight))
                    for pre, post, weight in conn_list if weight >=  0],
            ]
            for ct, cl in it.izip(connection_types, conn_lists):
                projections[ct].append(sim.Projection(
                        sources, population,
                        receptor_type={"exc":"excitatory", "inh":"inhibitory"}[ct],
                        synapse_type=sim.StaticSynapse(),
                        connector=sim.FromListConnector(cl,
                            column_names=["weight"])))

        else:
            # each sampler has its own population
            for j, pop in enumerate(population):
                if pop.conductance_based:
                    trans = lambda w: np.abs(w)
                else:
                    trans = lambda w: w

                conn_lists = [
                    [(pre, 0, trans(weight))
                        for pre, post, weight in conn_list
                        if weight <  0 and post == j],
                    [(pre, 0, trans(weight))
                        for pre, post, weight in conn_list
                        if weight >=  0 and post == j],
                ]
                for ct, cl in it.izip(connection_types, conn_lists):
                    projections[ct].append(sim.Projection(
                            sources, pop,
                            receptor_type={"exc":"excitatory", "inh":"inhibitory"}[ct],
                            synapse_type=sim.StaticSynapse(),
                            connector=sim.FromListConnector(cl,
                                column_names=["weight"])))

        return projections

    def get_distribution_parameters(self):
        """
            Return paramters needed for calculating the theoretical membrane
            distribution.
        """
        is_exc = self.weights > 0.
        is_inh = np.logical_not(is_exc)

        return {
            "rates_exc" : self.rates[is_exc],
            "rates_inh" : self.rates[is_inh],
            "weights_exc" : self.weights[is_exc],
            "weights_inh" : self.weights[is_inh],
        }


