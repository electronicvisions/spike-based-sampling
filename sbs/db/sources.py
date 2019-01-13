#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import itertools as it
import collections as c

from ..logcfg import log
from .core import Data
from ..conversion import weight_pynn_to_nest
from .neuron_parameters import NeuronParameters

__all__ = [
        "sources_create_connect",
        "SourceConfiguration",
        "PoissonSourceConfiguration",
        "FixedSpikeTrainConfiguration",
        "NoiseNetworkSourceConfiguration",
        "PoissonPoolSourceConfiguration",
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
            calibration) or a list of actual sampler objects that all might
            have different source configuration (but of the same type).

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

        projections[rectype] = sim.Projection(
            sources, samplers,
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

            projections[rectype].append(
                sim.Projection(s_sources, pop,
                               sim.FromListConnector(
                                   conn_list, column_names=column_names),
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
    for pop, pop_samplers in it.groupby(
            samplers, lambda s: s.network["population"]):
        if pop is None:
            retval.extend((s.population for s in pop_samplers))

        pop_samplers = list(pop_samplers)

        if len(pop_samplers) == len(samplers):
            # all samplers belong to the same population,
            # hence we can return a single Population/-View.
            # (usually, this is the default case)
            if len(pop_samplers) == len(pop):
                # the population is only made up of the current samplers,
                # hence we can return the population itself (less overhead)
                return pop
            else:
                # there are other neurons in the population that should not be
                # connected to the sources, hence we need to return a
                # PopulationView that only contains 'our' samplers
                return pop[np.array([s.network["index"] for s in samplers])]

        else:
            # there are other populations/samplers, hence we need to return a
            # list
            if len(pop_samplers) == len(pop):
                # the population is only made up of the current samplers,
                # hence we can return the population itself (less overhead)
                retval.append(pop)
            else:
                # there are other neurons in the population that should not be
                # connected to the sources, hence we need to return a
                # PopulationView that only contains 'our' samplers
                retval.append(
                        pop[np.array([s.network["index"] for s in samplers])])

    return retval


class PoissonSourceConfiguration(SourceConfiguration):
    """
        Positive weights: excitatory
        Negative weights: inhibitory
    """
    data_attribute_types = {
            "rates": np.ndarray,
            "weights": np.ndarray,
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
        source_params = {"start": 0.}
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
                    "steps_lookahead": 10000
                }
        else:
            source_model = "poisson_generator"
            source_model_kwargs = {}

        if not isinstance(samplers, sim.common.BasePopulation):
            num_sources_per_sampler = np.array([len(s.source_config.rates)
                                                for s in samplers])
            rates = np.hstack((s.source_config.rates for s in samplers))
            weights = np.hstack((s.source_config.weights for s in samplers))

        else:
            # if we have one population all get the same sources
            num_sources_per_sampler = (np.zeros(len(samplers), dtype=int) +
                                       len(self.rates))
            rates = np.hstack((self.rates for s in samplers))
            weights = np.hstack((self.weights for s in samplers))
            if self.weights[0]*self.weights[1] > 0:
                raise ValueError("Noise weights are both excitatory. "
                                 "Aborting.")

        # we want a mapping from each samplers sources into a large flattened
        # array
        offset_per_sampler = np.cumsum(num_sources_per_sampler)

        def idx_parrot_to_sampler(idx):
            """
                Parrot id to sampler id.
            """
            return np.where(idx < offset_per_sampler)[0][0]

        uniq_rates, idx_parrot_to_generator = np.unique(rates,
                                                        return_inverse=True)

        log.info("Creating {} different poisson sources.".format(
            uniq_rates.size))

        # PyNN is once again trying to smart in some way when it comes to
        # native cell types (creating parrot_neurons in a non-optional way), so
        # we just do everything in CyNEST, FFS!
        import nest
        gid_generators = np.array(nest.Create(source_model, uniq_rates.size))

        # acting as sources
        gid_parrots = np.array(nest.Create("parrot_neuron", rates.size))

        # dictionary to nest connection-tuples, these are NOT PyNN-projections!
        nest_connections = {
                "generator_to_parrot": [],
                "parrot_to_sampler": [],
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

        nest_connections["generator_to_parrot"].append(
                nest.Connect(gid_generators[idx_parrot_to_generator].tolist(),
                             gid_parrots.tolist(), "one_to_one"))

        idx_samplers = np.array([idx_parrot_to_sampler(i_parrot)
                                 for i_parrot in xrange(gid_parrots.size)])
        connect_gid_samplers = gid_samplers[idx_samplers]

        nest_connections["parrot_to_sampler"].append(
                nest.Connect(gid_parrots.tolist(),
                             connect_gid_samplers.tolist(),
                             "one_to_one",
                             {"weight": weight_pynn_to_nest(weights)}))

        sources = {
                "generators": gid_generators,
                "parrots": gid_parrots,
            }

        return sources, nest_connections

    def get_distribution_parameters(self):
        """
            Return paramters needed for calculating the theoretical membrane
            distribution.
        """
        is_exc = self.weights > 0.
        is_inh = np.logical_not(is_exc)

        return {
            "rates_exc": self.rates[is_exc],
            "rates_inh": self.rates[is_inh],
            "weights_exc": self.weights[is_exc],
            "weights_inh": self.weights[is_inh],
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
            "rates": np.ndarray,
            "weights": np.ndarray,
            "spike_times": np.ndarray,
            "spike_ids": np.ndarray,
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
                def trans(w):
                    return np.abs(w)
            else:
                def trans(w):
                    return w

            conn_lists = [
                [(pre, post, trans(weight))
                    for pre, post, weight in conn_list if weight < 0],
                [(pre, post, trans(weight))
                    for pre, post, weight in conn_list if weight >= 0],
            ]
            for ct, cl in it.izip(connection_types, conn_lists):
                projections[ct].append(sim.Projection(
                        sources, population,
                        receptor_type={"exc": "excitatory",
                                       "inh": "inhibitory"}[ct],
                        synapse_type=sim.StaticSynapse(),
                        connector=sim.FromListConnector(
                            cl, column_names=["weight"])))

        else:
            # each sampler has its own population
            for j, pop in enumerate(population):
                if pop.conductance_based:
                    def trans(w):
                        return np.abs(w)
                else:
                    def trans(w):
                        return w

                conn_lists = [
                    [(pre, 0, trans(weight))
                        for pre, post, weight in conn_list
                        if weight < 0 and post == j],
                    [(pre, 0, trans(weight))
                        for pre, post, weight in conn_list
                        if weight >= 0 and post == j],
                ]
                for ct, cl in it.izip(connection_types, conn_lists):
                    projections[ct].append(sim.Projection(
                            sources, pop,
                            receptor_type={"exc": "excitatory",
                                           "inh": "inhibitory"}[ct],
                            synapse_type=sim.StaticSynapse(),
                            connector=sim.FromListConnector(
                                cl, column_names=["weight"])))

        return projections

    def get_distribution_parameters(self):
        """
            Return paramters needed for calculating the theoretical membrane
            distribution.
        """
        is_exc = self.weights > 0.
        is_inh = np.logical_not(is_exc)

        return {
            "rates_exc": self.rates[is_exc],
            "rates_inh": self.rates[is_inh],
            "weights_exc": self.weights[is_exc],
            "weights_inh": self.weights[is_inh],
        }


class NoiseNetworkSourceConfiguration(SourceConfiguration):
    """
        Noise network supplying samplers with noise.
    """

    data_attribute_types = {
       # network attributes
       "N": int,  # number of neurons in noise network
       "gamma": float,  # percentage of excitatory neurons (in [0,1])
       "epsilon": float,  # connectivity (#presynaptic partners=epsilon*N)

       "epsilon_external": float,  # connectivity to samplers

       "neuron_parameters": NeuronParameters,

       # synapse parameters
       "delay_internal": float,  # within netowrk
       "delay_external": float,  # to samplers

       "g": float,  # relative weight of inhibitory synapses
                    # g= (J_I * tau_I * |V_rest-V_rev_I|)
                    #   /(J_E * tau_E * |V_rest-V_rev_E|)

       "JE": float,  # excitatory weight [µS]/[nA]
       "f_J_external": float,  # factor with which the external weights are
                               # multiplied

       "rate": float,  # rate with which each noise neuron is assumed to
                       # fire on average (only used for initial calibration)

       "seed": int,  # random seed
    }

    data_attribute_defaults = {
            "f_J_external": 1.,
            "seed": 424242,
        }

    @property
    def JI(self):
        params = self.neuron_parameters

        if hasattr(params, "e_rev_E"):
            return (
                self.g * self.JE * params.tau_syn_E *
                np.abs(params.v_rest - params.e_rev_E)
                / (params.tau_syn_I * np.abs(params.v_rest - params.e_rev_I)))
        else:
            return (self.g * self.JE
                    * params.tau_syn_E
                    / params.tau_syn_I)

    @property
    def num_exc(self):
        return int(np.around(self.N * self.gamma))

    @property
    def num_inh(self):
        return self.N - self.num_exc

    @property
    def indegree_external_exc(self):
        """
            Number of (excitatory) presynaptic partners each sampler receives
            input from.
        """
        return int(np.around(self.num_exc * self.epsilon_external))

    @property
    def indegree_external_inh(self):
        """
            Number of (inhibitory) presynaptic partners each sampler receives
            input from.
        """
        return int(np.around(self.num_inh * self.epsilon_external))

    @property
    def indegree_exc(self):
        """
            Number of (excitatory) presynaptic partners each neuron inside the
            noise network receives input from.
        """
        return int(np.around(self.num_exc * self.epsilon))

    @property
    def indegree_inh(self):
        """
            Number of (inhibitory) presynaptic partners each neuron inside the
            noise network receives input from.
        """
        return int(np.around(self.num_inh * self.epsilon))

    def get_JI(self, target_pop):
        """
            Returns the target inhibitory weight based on whether the target is
            conductance (positive weight) or current (negative weight) based.
        """
        if target_pop.conductance_based:
            return self.JI
        else:
            return -self.JI

    def get_distribution_parameters(self):
        """
            Return paramters needed for calculating the theoretical membrane
            distribution.
        """
        rate_exc = self.rate * self.epsilon_external * self.num_exc
        rate_inh = self.rate * self.epsilon_external * self.num_inh

        return {
            "rates_exc": np.array([rate_exc]),
            "rates_inh": np.array([rate_inh]),
            "weights_exc": np.array([self.JE]),
            "weights_inh": np.array([-self.JI]),
        }

    def create_connect(self, sim, samplers, **kwargs):
        """
            If samplers is None, only the noise network is created.
        """
        # we need to distinguish three cases:
        # whether we are connecting to a regular population (calibration etc)
        # or to a list of samplers that a) have the same rates or b) have
        # different rates
        if isinstance(samplers, sim.Population) or samplers is None:
            sampler_nn_cfgs = [(samplers, self)]
        else:
            sampler_nn_cfgs = (
                (list(cfg_samplers), cfg) for cfg, cfg_samplers
                in it.groupby(samplers, lambda s: s.source_config))

        sources = []
        projections = []

        # seed is always taken from the current source object
        rng = sim.NumpyRNG(seed=self.seed)

        for samplers, cfg in sampler_nn_cfgs:
            log.info(
                "Creating noice network of size {} to supply samplers.".format(
                    cfg.N))
            source = sim.Population(
                    cfg.N, cfg.neuron_parameters.get_pynn_model_object(sim)())
            source.set(**cfg.neuron_parameters.get_pynn_parameters())
            sources.append(source)

            source.initialize(v=rng.uniform(
                cfg.neuron_parameters.v_reset,
                cfg.neuron_parameters.v_thresh,
                size=cfg.N))

            src_exc = source[:cfg.num_exc]
            src_inh = source[-cfg.num_inh:]

            if cfg.epsilon > 0.:
                projections.append({
                    "internal:": {
                        "EE": sim.Projection(
                            src_exc, src_exc,
                            connector=sim.FixedNumberPreConnector(
                                n=cfg.indegree_exc,
                                allow_self_connections=False,
                                with_replacement=False,
                                rng=rng),
                            synapse_type=sim.StaticSynapse(
                                weight=cfg.JE, delay=cfg.delay_internal),
                            receptor_type="excitatory"),

                        "EI": sim.Projection(
                            src_exc, src_inh,
                            connector=sim.FixedNumberPreConnector(
                                n=cfg.indegree_exc,
                                allow_self_connections=False,
                                with_replacement=False,
                                rng=rng),
                            synapse_type=sim.StaticSynapse(
                                weight=cfg.JE, delay=cfg.delay_internal),
                            receptor_type="excitatory"),

                        "IE": sim.Projection(
                            src_inh, src_exc,
                            connector=sim.FixedNumberPreConnector(
                                n=cfg.indegree_inh,
                                allow_self_connections=False,
                                with_replacement=False,
                                rng=rng),
                            synapse_type=sim.StaticSynapse(
                                weight=cfg.get_JI(src_exc),
                                delay=cfg.delay_internal),
                            receptor_type="inhibitory"),

                        "II": sim.Projection(
                            src_inh, src_inh,
                            connector=sim.FixedNumberPreConnector(
                                n=cfg.indegree_inh,
                                allow_self_connections=False,
                                with_replacement=False,
                                rng=rng),
                            synapse_type=sim.StaticSynapse(
                                weight=cfg.get_JI(src_inh),
                                delay=cfg.delay_internal),
                            receptor_type="inhibitory"),
                        },
                    })

            if samplers is None:
                log.warn("No samplers supplied, only created noise network.")
                continue

            pops = get_population_from_samplers(sim, samplers)

            if isinstance(pops, sim.Population):
                pops = [pops]

            projections[-1]["external"] = []
            for pop in pops:
                JE = cfg.JE * cfg.f_J_external
                JI = cfg.JI * cfg.f_J_external
                log.info("Noise network: {} exc src @ {} / {} inh @ {}".format(
                    cfg.indegree_external_exc, JE,
                    cfg.indegree_external_inh, JI))
                projections[-1]["external"].append({
                    "E": sim.Projection(
                        src_exc, pop,
                        connector=sim.FixedNumberPreConnector(
                            n=cfg.indegree_external_exc,
                            allow_self_connections=False,
                            with_replacement=False,
                            rng=rng),
                        synapse_type=sim.StaticSynapse(
                            weight=cfg.JE * cfg.f_J_external,
                            delay=cfg.delay_external),
                        receptor_type="excitatory"),

                    "I": sim.Projection(
                        src_inh, pop,
                        connector=sim.FixedNumberPreConnector(
                            n=cfg.indegree_external_inh,
                            allow_self_connections=False,
                            with_replacement=False,
                            rng=rng),
                        synapse_type=sim.StaticSynapse(
                            weight=cfg.get_JI(pop) * cfg.f_J_external,
                            delay=cfg.delay_external),
                        receptor_type="inhibitory"),
                })

        return sources, projections

    def measure_firing_rates(self, sim_name, duration,
                             burn_in_time=500.,
                             sim_setup_kwargs=None):
        """
            Measure and return the average firing rates of the noise network.
        """
        from ..gather_data import nn_measure_firing_rates

        return nn_measure_firing_rates(
                self, sim_name, duration,
                burn_in_time=burn_in_time,
                sim_setup_kwargs=sim_setup_kwargs)


class PoissonPoolSourceConfiguration(SourceConfiguration):
    """
        A pool of Poisson sources with fixed rate.

        The main use is to compare the sampling performance of a Boltzmann
        machine supplied with Noise network and Poisson pool. The former
        actively decorralates the input spike trains received by every sampling
        unit, the latter doesn't.

        Each neuron in the Boltzmann machine will be connected to a subset of
        Poisson sources. This will lead to shared input correlations in the
        noise input to different samplers.
    """

    # if the specified source model is not available, this will be used
    fallback_nest_source_model = "poisson_generator"
    fallback_nest_source_model_kwargs = {}

    data_attribute_types = {
       # network attributes
       "N": int,  # number of neurons in noise network
       "gamma": float,  # percentage of excitatory neurons (in [0,1])

       "nest_source_model": str,
       "nest_source_model_kwargs": dict,

       "epsilon_external": float,  # connectivity to samplers

       # synapse parameters
       "delay_external": float,  # to samplers

       "g": float,  # relative weight of inhibitory synapses
                    # g= (J_I * tau_I * |V_rest-V_rev_I|)
                    #   /(J_E * tau_E * |V_rest-V_rev_E|)

       "JE": float,  # excitatory weight [µS]/[nA] from Poisson Pool to the
                     # functional network

       "rate": float,  # rate with which each poisson source spikes

       "seed": int,  # random seed
    }

    data_attribute_defaults = {
        "seed": 424242,

        "nest_source_model": "poisson_generator",
        "nest_source_model_kwargs": {},
    }

    @property
    def JI(self):
        return self.g * self.JE

    @property
    def num_exc(self):
        return int(np.around(self.N * self.gamma))

    @property
    def num_inh(self):
        return self.N - self.num_exc

    @property
    def indegree_external_exc(self):
        """
            Number of (excitatory) presynaptic partners each sampler receives
            input from.
        """
        return int(np.around(self.num_exc * self.epsilon_external))

    @property
    def indegree_external_inh(self):
        """
            Number of (inhibitory) presynaptic partners each sampler receives
            input from.
        """
        return int(np.around(self.num_inh * self.epsilon_external))

    def get_JI(self, target_pop):
        """
            Returns the target inhibitory weight based on whether the target is
            conductance (positive weight) or current (negative weight) based.
        """
        if target_pop.conductance_based:
            return self.JI
        else:
            return -self.JI

    def get_distribution_parameters(self):
        """
            Return paramters needed for calculating the theoretical membrane
            distribution.
        """
        rate_exc = self.rate * self.epsilon_external * self.num_exc
        rate_inh = self.rate * self.epsilon_external * self.num_inh

        return {
            "rates_exc": np.array([rate_exc]),
            "rates_inh": np.array([rate_inh]),
            "weights_exc": np.array([self.JE]),
            "weights_inh": np.array([-self.JI]),
        }

    def create_connect(self, sim, samplers, **kwargs):
        """
            If samplers is None, only the noise network is created.
        """
        # we need to distinguish three cases:
        # whether we are connecting to a regular population (calibration etc)
        # or to a list of samplers that a) have the same rates or b) have
        # different rates
        assert hasattr(sim, "nest"), "Only nest compatible!"
        nest = sim.nest

        if isinstance(samplers, sim.Population) or samplers is None:
            sampler_cfgs = [(samplers, self)]
        else:
            sampler_cfgs = (
                    (list(cfg_samplers), cfg) for cfg, cfg_samplers
                    in it.groupby(samplers, lambda s: s.source_config))

        sources = []
        projections = []

        for samplers, cfg in sampler_cfgs:
            log.info("Creating poisson network of size {} "
                     "to supply samplers.".format(cfg.N))

            if self.nest_source_model in nest.Models():
                model_name = self.nest_source_model
                model_kwargs = self.nest_source_model_kwargs
            else:
                log.warn("{} not available in nest, falling back to: "
                         "{}".format(self.nest_source_model,
                                     self.fallback_nest_source_model))
                model_name = self.fallback_nest_source_model
                model_kwargs = self.fallback_nest_source_model_kwargs

            model_kwargs["rate"] = self.rate

            gid_generator = nest.Create(model_name, n=1, params=model_kwargs)
            gid_parrots = nest.Create("parrot_neuron", n=self.N)

            sources.append({
                "generator": gid_generator,
                "parrots": gid_parrots
            })

            nest.Connect(gid_generator, gid_parrots, "all_to_all")

            gid_exc = gid_parrots[:cfg.num_exc]
            gid_inh = gid_parrots[-cfg.num_inh:]

            if samplers is None:
                log.warn("No samplers supplied, only created noise network.")
                continue

            pops = get_population_from_samplers(sim, samplers)

            if isinstance(pops, sim.Population):
                pops = [pops]

            # we dont need to get the connections from nest, so we leave it
            # empty
            projections.append({})

            for pop in pops:
                log.info("Noise network: {} exc src @ {} / {} inh @ {}".format(
                    cfg.indegree_external_exc, cfg.JE,
                    cfg.indegree_external_inh, cfg.JI))

                gids = pop.all_cells.tolist()
                nest.Connect(gid_exc, gids, conn_spec={
                    "rule": "fixed_indegree",
                    "indegree": self.indegree_external_exc,
                    "autapses": False,
                    "multapses": False,
                }, syn_spec={
                    "model": "static_synapse",
                    "weight": weight_pynn_to_nest(self.JE),
                    "delay": self.delay_external,
                })

                nest.Connect(gid_inh, gids, conn_spec={
                    "rule": "fixed_indegree",
                    "indegree": self.indegree_external_inh,
                    "autapses": False,
                    "multapses": False,
                }, syn_spec={
                    "model": "static_synapse",
                    "weight": weight_pynn_to_nest(-self.JI),
                    "delay": self.delay_external,
                })

        return sources, projections
