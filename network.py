#!/usr/bin/env python
# encoding: utf-8

import collections as c
import itertools as it
import numpy as np
import logging
import sys
from pprint import pformat as pf

from .logcfg import log
from . import db
from . import samplers
from . import utils
from . import gather_data
from . import meta

class Distribution(object):
    """
        Class to store and access all different distributions.

        A state is a simple tuple over 0s and 1s.
    """
    class Probability(object):
        """
            Probabilities that have array-like access.
        """

        def __init__(self, dist):
            # keep track of parent distribution
            self.dist = dist
            self.initialize()

        def initialize(self):
            pass

        def __getitem__(self, state):
            """
                Return probability of `state` if present, else 0.
            """
            raise NotImplementedError

        def __setitem__(self, state, value):
            """
                Set probability of `state` if present, else fail silently.
            """
            raise NotImplementedError

        def normalize(self, total=None):
            if total is None:
                self._probs /= self._probs.sum()
            else:
                self._probs /= total


    def __init__(self, sampler_idx):
        self.num_samplers = len(sampler_idx)
        self.sampler_idx = sampler_idx
        self.probs = self.Probability(self)

        # global to local idx
        self.idx_gtl = self._reverse_sampler_mapping()

    def _reverse_sampler_mapping(self):
        return {sid: i for i, sid in enumerate(self.sampler_idx)}

    def all_states(self):
        """
            Return an iterator over all possible states.
        """
        raise NotImplementedError

    def all_probabilities(self):
        """
            Return all probabilities in a flattened manner (same order as
            states).
        """
        raise NotImplementedError

    def all_items(self):
        """
            Return an iterator over all state/probability pairs.
        """
        return it.izip(self.all_states(), self.all_probabilities())

    def normalize(self, total=None):
        """
            Normalize probability, `total` can be used to specify a different
            total value to which the distribution should be normalized.
        """
        self.probs.normalize(total)



class CompleteDistribution(Distribution):
    """
        Full distribution.
    """
    class Probability(Distribution.Probability):
        def initialize(self):
            num_samplers = self.dist.num_samplers
            shape = tuple((2 for i in xrange(num_samplers)))
            self._probs = np.zeros(shape)

        def __getitem__(self, state):
            return self._probs[state]

        def __setitem__(self, state, value):
            self._probs[state] = value

    def all_states(self):
        return np.ndindex(*self.probs._probs.shape)

    def all_probabilities(self):
        return self.probs._probs.flatten()


class ExclusiveDistribution(Distribution):
    """
        Only one sampler may be active at any time.
    """
    class Probability(Distribution.Probability):
        def initialize(self):
            num_samplers = self.dist.num_samplers
            self._probs = np.zeros((num_samplers+1))

        def state_iterator(self, state):
            """
                Iterate over all possible states while accounting for slices
            """
            if log.getEffectiveLevel() <= logging.DEBUG:
                log.debug("Current state: {}".format(pf(state)))

            iterables = map(lambda s: xrange(
                    s.start if s.start is not None else 0,
                    s.stop if s.stop is not None else 2,
                    s.step if s.step is not None else 1)
                if isinstance(s, slice) else (s,), state)
            if log.getEffectiveLevel() <= logging.DEBUG:
                log.debug("Current state: {}".format(pf(iterables)))

            return it.product(*iterables)

        def __getitem__(self, state):
            values = []
            for ss in self.state_iterator(state):
                state_sum = sum(ss)
                if state_sum == 0:
                    values.append(self._probs[-1])
                elif state_sum == 1:
                    j = 0
                    while ss[j] == 0:
                        j += 1
                    values.append(self._probs[j])
                else:
                    values.append(0.)

            if len(values) == 1:
                return values[0]
            else:
                return np.array(values)

        def __setitem__(self, state, value):
            for i, ss in enumerate(self.state_iterator(state)):
                state_sum = sum(ss)
                if state_sum == 0:
                    self._probs[-1] = value
                elif state_sum == 1:
                    j = 0
                    while ss[j] == 0:
                        j += 1
                    if utils.check_list_array(value):
                        self._probs[j] = value[i]
                    else:
                        self._probs[j] = value
                else:
                    return 0.

    def all_states(self):
        num_samplers = self.num_samplers
        state = [0] * num_samplers
        for i in xrange(num_samplers):
            if i > 0:
                state[i-1] = 0
            state[i] = 1
            yield tuple(state)

        yield (0,) * num_samplers


    def all_probabilities(self):
        return self.probs._probs.flatten()


@meta.HasDependencies
class BoltzmannMachine(object):
    """
        A set of samplers connected as Boltzmann machine.
    """

    def __init__(self, num_samplers, sim_name="pyNN.nest",
            pynn_model=None,
            neuron_parameters=None, neuron_index_to_parameters=None,
            neuron_parameters_db_ids=None):
        """
        Sets up a Boltzmann network.

        `pynn_model` is the string of the pyNN model used. Note that if
        neuron_parmas is a list, `pynn_model` also has to be.

        There are several ways to specify neuron_parameters:

        `neuron_parameters` as a single dictionary:
        ====
            All samplers will have the same parameters specified by
            neuron_parameters.

        `neuron_parameters` as a list of dictionaries of length `num_samplers`:
        ====
            Sampler `i` will have paramaters `neuron_parameters[i]`.

        `neuron_parameters` as a list of dictionaries of length <
        `num_samplers` and `neuron_index_to_parameters` is list of length
        `num_samplers` of ints:
        ====
            Sampler `i` will have parameters
            `neuron_parameters[neuron_index_to_parameters[i]]`.

        `neuron_parameters_db_ids` is a list of ints of length `num_samplers`:
        ====
            Sampler `i` will load its parameters from database entry with
            id `neuron_parameters_db_ids[i]`.

        `neuron_parameters_db_ids` is a single id:
        ====
            All samplers will load the same neuron parameters with the
            corresponding id.
        """
        log.info("Creating new BoltzmannMachine.")
        self.sim_name = sim_name
        self.num_samplers = num_samplers

        if isinstance(pynn_model, basestring):
            pynn_model = [pynn_model] * num_samplers

        if neuron_parameters is not None:
            if not isinstance(neuron_parameters, c.Sequence):
                neuron_parameters = [neuron_parameters]
                neuron_index_to_parameters = [0] * num_samplers

            elif neuron_index_to_parameters is None:
                neuron_index_to_parameters = range(num_samplers)

            self.samplers = [samplers.LIFsampler(
                sim_name=self.sim_name,
                pynn_model=pynn_model[i],
                neuron_parameters=neuron_parameters[i],
                silent=True)\
                        for i in neuron_index_to_parameters]

        elif neuron_parameters_db_ids is not None:
            if not isinstance(neuron_parameters_db_ids, c.Sequence):
                neuron_parameters_db_ids = (neuron_parameters_db_ids,)\
                        * self.num_samplers
            self.samplers = [samplers.LIFsampler(id=id, sim_name=self.sim_name,
                silent=True) for id in neuron_parameters_db_ids]
        else:
            raise Exception("Please provide either parameters or ids in the "
                    "database!")

        self.weights_theo = np.zeros((num_samplers, num_samplers))
        # biases are set to zero automaticcaly by the samplers

        self.delays = 0.1
        self.sampler_idx = range(self.num_samplers)

    ########################
    # pickle serialization #
    ########################
    # generally we only save the ids of samplers and calibrations used
    # (we can be sure that only saved samplers are used in the network as there
    # is no way to calibrated them from the network)
    # plus record biases and weights
    def __getstate__(self):
        log.debug("Reading state information for pickling.")
        state = {
                "calibration_ids" : [sampler.get_calibration_id()
                    for sampler in self.samplers],
            }
        # avoid unnecessary conversions for weights
        for wt in ["theo", "bio"]: # weight type
            weights = getattr(self, "_weights_{}".format(wt), None)
            if weights is not None:
                state["weights"] = {"type": wt, "value": weights}

        # biases are not as important so we just save the same as the weights
        # ones (we assume that the user will almost never mix bio and
        # theoretical biases/weights and if he does the precision/performance
        # difference is acceptable)
        for i, sampler in enumerate(self.samplers):
            state["weights"]["value"][i, i] =\
                getattr(sampler, "bias_{}".format(state["weights"]["type"]))

        state["delays"] = self.delays

        state["sampler_idx"] = self.sampler_idx

        state["sim_name"] = self.sim_name
        state["num_samplers"] = self.num_samplers
        state["params_ids"] = [sampler.get_parameters_id()
                for sampler in self.samplers]

        return state

    def __setstate__(self, state):
        log.debug("Setting state information for unpickling.")
        self.__init__(state["num_samplers"],
                sim_name=state["sim_name"],
                neuron_parameters_db_ids=state["params_ids"])

        self.sampler_idx = state["sampler_idx"]

        for i, cid in enumerate(state["calibration_ids"]):
            if cid is not None:
                self.samplers[i].load_calibration(id=cid)

        # set the biases
        bias_type = "bias_{}".format(state["weights"]["type"])
        for i, sampler in enumerate(self.samplers):
            setattr(self.samplers[i], bias_type,
                    state["weights"]["value"][i, i])

        # restore whatever weight type we saved
        setattr(self, "weights_{}".format(state["weights"]["type"]),
                state["weights"]["value"])

        self.delays = state["delays"]

    ######################
    # regular attributes #
    ######################

    @meta.DependsOn("weights_bio")
    def weights_theo(self, weights=None):
        """
            Set or retrieve the connection weights

            Can be a scalar to set all weights to the same value.

            Automatic conversion:
            After the weights have been set in either biological or theoretical
            units both can be retrieved and the conversion will be done when
            needed.
        """
        if weights is not None:
            # setter part
            return self._check_weight_matrix(weights)
        else:
            # getter part
            return self.convert_weights_bio_to_theo(self.weights_bio)

    @meta.DependsOn("weights_theo")
    def weights_bio(self, weights=None):
        """
            Set or retrieve the connection weights

            Can be a scalar to set all weights to the same value.

            Automatic conversion:
            After the weights have been set in either biological or theoretical
            units both can be retrieved and the conversion will be done when
            needed.
        """
        if weights is not None:
            # setter part
            return self._check_weight_matrix(weights)
        else:
            # getter part
            return self.convert_weights_theo_to_bio(self.weights_theo)

    def _check_weight_matrix(self, weights):
        weights = np.array(weights)

        if len(weights.shape) == 0:
            scalar_weight = weights
            weights = np.empty((self.num_samplers, self.num_samplers))
            weights.fill(scalar_weight)

        expected_shape = (self.num_samplers, self.num_samplers)
        assert weights.shape == expected_shape,\
                "Weight matrix shape {}, expected {}".format(weights.shape,
                        expected_shape)
        for i in xrange(self.num_samplers):
            weights[i, i] = 0.
        return weights

    @property
    def biases_theo(self):
        return np.array([s.bias_theo for s in self.samplers])

    @property
    def biases_bio(self):
        return np.array([s.bias_bio for s in self.samplers])

    @biases_theo.setter
    def biases_theo(self, biases):
        if not utils.check_list_array(biases):
            biases = it.repeat(biases)

        for b, sampler in it.izip(biases, self.samplers):
            sampler.bias_theo = b

    @biases_bio.setter
    def biases_bio(self, biases):
        if not utils.check_list_array(biases):
            biases = it.repeat(biases)

        for b, sampler in it.izip(biases, self.samplers):
            sampler.bias_bio = b

    def convert_weights_bio_to_theo(self, weights):
        weights = utils.fill_diagonal(weights, 0.)
        # the column index denotes the target neuron, hence we convert there
        for j, sampler in enumerate(self.samplers):
            weights[:, j] = sampler.convert_weights_bio_to_theo(weights[:, j])
        return weights

    def convert_weights_theo_to_bio(self, weights):
        weights = utils.fill_diagonal(weights, 0.)
        # the column index denotes the target neuron, hence we convert there
        for j, sampler in enumerate(self.samplers):
            weights[:, j] = sampler.convert_weights_theo_to_bio(weights[:, j])

        return weights

    @meta.DependsOn()
    def delays(self, delays):
        """
            Delays can either be a scalar to indicate a global delay or an
            array to indicate the delays between the samplers.
        """
        delays = np.array(delays)
        if len(delays.shape) == 0:
            scalar_delay = delays
            delays = np.empty((self.num_samplers, self.num_samplers))
            delays.fill(scalar_delay)
        return delays

    def load_calibration(self, *ids):
        """
            Load the specified calibration ids from the samplers.

            For any id not specified, the latest configuration will be loaded.

            Returns a list of sampler(-parameter) ids that failed.
        """
        failed = []
        for i, sampler in enumerate(self.samplers):
            if i < len(ids):
                id = ids[i]
            else:
                id = None
            if not sampler.load_calibration(id=id):
                failed.append(samper.db_params.id)

        return failed

    def all_samplers_same_model(self):
        """
            Returns true of all samplers have the same pynn model.

            If this returns False, expect `self.population` to be a list of
            size-1 populations unless specified differently during creation.
        """
        return all(
            ((sampler.pynn_model == self.samplers[0].pynn_model)\
                for sampler in self.samplers))

    #######################
    # PROBABILITY methdos #
    #######################

    # methods to gather data
    @meta.DependsOn()
    def spike_data(self, spike_data):
        """
            The spike data from which to compute distributions.
        """
        assert "spiketrains" in spike_data
        assert "duration" in spike_data
        return spike_data

    def gather_spikes(self, duration, dt=0.1, burn_in_time=100.):
        self.spike_data = gather_data.gather_network_spikes(self,
                duration=duration, dt=dt, burn_in_time=burn_in_time)

    @meta.DependsOn("spike_data")
    def ordered_spikes(self):
        log.info("Getting ordered spikes")
        return utils.get_ordered_spike_idx(self.spike_data["spiketrains"])

    def save_spikes(self, filename):
        utils.save_pickle(self.spike_data, filename)

    def load_spikes(self, filename):
        """
            Returns True if successfully loaded, False otherwise.
        """
        try:
            self.spike_data = utils.load_pickle(filename)
            return True
        except:
            if log.getEffectiveLevel() <= logging.DEBUG:
                log.debug(sys.exc_info()[0])
            return False

    def generate_states(self, sampler_idx=None, exclusive_states=False):
        """
            `sampler_idx` can be used to limit the distribution to the list of
            indices supplied.

            If `exclusive_states` is True, there will only be N+1 states
            (instead of 2**N) where the i-th state is the one where sampler i
            is active and all others are 0. The last state is the zero state.

            Returns both a list of states as well as an array to hold 
        """
        if exclusive_states:
            return ExclusiveDistribution(sampler_idx=sampler_idx)
        else:
            return CompleteDistribution(sampler_idx=sampler_idx)

    @meta.DependsOn()
    def sampler_idx(self, sampler_idx):
        return set(sampler_idx)

    @meta.DependsOn("spike_data", "sampler_idx")
    def dist_marginal_sim(self):
        """
            Marginal distribution computed from spike data.

            `**state_args` are passed to `self.generate_states`.
        """
        log.info("Calculating marginal probability distribution for {} "
                "samplers.".format(len(self.sampler_idx)))

        marginals = np.zeros((len(self.sampler_idx),))

        for i in self.sampler_idx:
            sampler = self.samplers[i]
            spikes = self.spike_data["spiketrains"][i]
            marginals[i] = len(spikes) * sampler.db_params.tau_refrac

        marginals /= self.spike_data["duration"]

        return marginals

    @meta.DependsOn("spike_data", "sampler_idx")
    def dist_joint_sim(self):
        return self.get_dist_joint_sim(dist=CompleteDistribution(
            sampler_idx=self.sampler_idx))

    @meta.DependsOn("spike_data", "sampler_idx")
    def dist_joint_exclusive_sim(self):
        return self.get_dist_joint_sim(dist=ExclusiveDistribution(
            sampler_idx=self.sampler_idx))

    def get_dist_joint_sim(self, dist):
        """
            Joint distribution computed from spike data.

            `**state_args` are passed to `self.generate_states`.
        """

        log.info("Calculating joint probability distribution for {} samplers."\
                .format(len(dist.sampler_idx)))

        tau_sampler = np.zeros((dist.num_samplers,))

        current_time = 0.
        i_spike = 0

        log.debug("Number of spikes: {}".format(len(self.ordered_spikes)))

        while current_time < self.spike_data["duration"]:
            log.debug("Currently at: {}ms".format(current_time))
            next_inactivation = min(
                    it.chain(tau_sampler[tau_sampler > 0], (np.inf,)))
            if i_spike < len(self.ordered_spikes):
                next_spike = self.ordered_spikes["t"][i_spike] - current_time
            else:
                next_spike = self.spike_data["duration"] - current_time

            # check out if the next event is a spike or a simple inactivation of
            # a sampler
            if next_inactivation > next_spike:
                is_spike = i_spike < len(self.ordered_spikes)
                time_step = next_spike

            else:
                is_spike = False
                time_step = next_inactivation

            current_state = tuple(np.array(tau_sampler > 0, dtype=int))

            # note that the current state is on for the next time
            dist.probs[current_state] += time_step

            tau_sampler -= time_step

            if is_spike:
                sampler_id = self.ordered_spikes["id"][i_spike]
                # adjust current spike
                tau_sampler[dist.idx_gtl[sampler_id]] =\
                    self.samplers[sampler_id].db_params.tau_refrac

                # find next spike
                i_spike += 1
                # skip all spikes from samplers we do not care about
                while i_spike < len(self.ordered_spikes)\
                            and self.ordered_spikes["id"][i_spike]\
                        not in dist.sampler_idx:
                    i_spike += 1

            current_time += time_step

        dist.normalize(self.spike_data["duration"])

        return dist


    @meta.DependsOn("sampler_idx")
    def dist_joint_theo(self):
        """
            Joint distribution for all selected samplers.
        """
        return self.get_dist_joint_theo(
                dist=CompleteDistribution(sampler_idx=self.sampler_idx),
                weights=self.weights_theo,
                biases=self.biases_theo)

    @meta.DependsOn("dist_joint_theo")
    def dist_marginal_theo(self):
        """
            Marginal distribution
        """
        return self.get_dist_marginal_from_joint(self.dist_joint_theo)

    @classmethod
    def get_dist_joint_theo(cls, dist, weights, biases):
        """
            Simple script that calculates joint distributions for all states in
            dist.
        """
        log.info("Calculating joint theoretical distribution for {} samplers."\
                .format(len(dist.sampler_idx)))
        sampler_idx = list(dist.sampler_idx)
        lc_biases = biases[sampler_idx]
        lc_weights = utils.fill_diagonal(weights[sampler_idx][:, sampler_idx],
                value=lc_biases)

        log.info("Biases: \n" + pf(lc_biases))
        log.info("Weights: \n" + pf(lc_weights))

        for state in dist.all_states():
            arr_state = np.array(state)
            prob = np.exp(arr_state.T.dot(lc_weights.dot(arr_state)))
            dist.probs[state] = prob

        dist.normalize()
        return dist

    @classmethod
    def get_dist_marginal_from_joint(cls, dist):
        """
            Marginal for supplied distribution.
        """
        marginals = np.zeros((dist.num_samplers, 2))
        for i in xrange(dist.num_samplers):
            log.info("Sampler #" + str(i))
            for s in (0, 1):
                states = (slice(None),) * i + (s,)\
                        + (slice(None),) * (dist.num_samplers-i-1)
                probs = np.array(dist.probs[states])
                marginals[i, s] = probs.sum()

        marginals /= marginals.sum(axis=1).reshape(-1, 1)
        return marginals


    ################
    # PYNN methods #
    ################

    def create(self, duration=None, population=None):
        """
            Create the sampling network and return the pynn object.

            If `duration` is None, the duration from the source configuration
            used for calibration will be used.

            If population is not None it should have length `self.num_samplers`.
            Also, if you specify different samplers to have different
            pynn_models, make sure that the list of pynn_objects provided
            supports those!

            Returns the newly created or specified popluation object for the
            samplers.
            If `populations` was None
        """
        exec "import {} as sim".format(self.sim_name) in globals(), locals()

        all_samplers_same_model = self.all_samplers_same_model()

        if population is None and not all_samplers_same_model:
            log.warn("The samplers have different pynn_models. "
            "Therefore there will be one population per sampler. "
            "This is rather inefficient.")

        if population is None and all_samplers_same_model:
            log.info("Setting up single population for all samplers.")
            population = sim.Population(self.num_samplers,
                    getattr(sim, self.samplers[0].pynn_model)())

        elif population is None:
            population = []

        for i, sampler in enumerate(self.samplers):
            if isinstance(population, sim.Population):
                local_pop = population[i:i+1]
            elif len(population) > i:
                local_pop = population[i]
            else:
                local_pop = None

            retval = sampler.create(duration=duration, population=local_pop)

            # if every sampler creates its own object we need to keep track
            if local_pop is None:
                population.append(retval)

        # we dont set any connections for weights that are == 0.
        weight_is = {}
        weight_is["exc"] = self.weights_bio > 0.
        weight_is["inh"] = self.weights_bio < 0.

        receptor_type = {"exc" : "excitatory", "inh" : "inhibitory"}

        global_delay = len(self.delays.shape) > 0

        self.projections = {}
        for wt in ["exc", "inh"]:
            if weight_is[wt].sum() == 0:
                # there are no weights of the current type, continue
                continue

            log.info("Connecting {} weights.".format(receptor_type[wt]))

            if global_delay:
                delays = self.delays
            else:
                delays = self.delays[weight_is[wt]]

            weights = utils.fill_diagonal(self.weights_bio.copy(), value=0)
            weights[np.logical_not(weight_is[wt])] = np.NaN

            if wt == "inh":
                weights = weights.copy() * -1

            self.projections[wt] = sim.Projection(population, population,
                    connector=sim.ArrayConnector(weight_is[wt]),
                    synapse_type=sim.StaticSynapse(
                        weight=weights, delay=delays),
                    receptor_type=receptor_type[wt])
            self.projections[wt].set(delay=delays)

        self.population = population 

        return self.population, self.projections


