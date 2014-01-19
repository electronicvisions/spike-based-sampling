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
from . import cutils
from . import gather_data
from . import meta
from . import buildingblocks as bb


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
        self.selected_sampler_idx = range(self.num_samplers)

        self.population = None
        self.projections = None

    ########################
    # pickle serialization #
    ########################
    # generally we only save the ids of samplers and calibrations used
    # (we can be sure that only saved samplers are used in the BM-network as
    # there is no way to calibrate them from the BM-network)
    # plus record biases and weights
    def __getstate__(self):
        log.debug("Reading state information for pickling.")
        state = {
                "calibration_ids" : [sampler.get_calibration_id()
                    for sampler in self.samplers],
                "current_basename" : db.current_basename,
            }
        state["weights"] = self.weights_theo

        state["biases"] = self.biases_theo

        state["delays"] = self.delays

        state["selected_sampler_idx"] = self.selected_sampler_idx

        state["sim_name"] = self.sim_name
        state["num_samplers"] = self.num_samplers
        state["params_ids"] = [sampler.get_parameters_id()
                for sampler in self.samplers]

        state["spike_data"] = self.spike_data

        return state

    def __setstate__(self, state):
        log.debug("Setting state information for unpickling.")

        if state["current_basename"] != db.current_basename:
            raise Exception("Database mismatch, this network should be "
            "restored with db {}".format(state["current_basename"]))

        self.__init__(state["num_samplers"],
                sim_name=state["sim_name"],
                neuron_parameters_db_ids=state["params_ids"])

        self.selected_sampler_idx = state["selected_sampler_idx"]

        for i, cid in enumerate(state["calibration_ids"]):
            if cid is not None:
                self.samplers[i].load_calibration(id=cid)

        self.weights_theo = state["weights"]
        self.biases_theo = state["biases"]

        self.delays = state["delays"]
        self.spike_data = state["spike_data"]

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
        weights = utils.fill_diagonal(weights, 0.)
        return weights

    @meta.DependsOn("biases_bio")
    def biases_theo(self, biases=None):
        if biases is None:
            # getter
            return np.array([s.bias_theo for s in self.samplers])
        else:
            #setter
            if not utils.check_list_array(biases):
                biases = it.repeat(biases)

            for b, sampler in it.izip(biases, self.samplers):
                sampler.bias_theo = b

    @meta.DependsOn("biases_theo")
    def biases_bio(self, biases=None):
        if biases is None:
            # getter
            return np.array([s.bias_bio for s in self.samplers])
        else:
            # setter
            if not utils.check_list_array(biases):
                biases = it.repeat(biases)

            for b, sampler in it.izip(biases, self.samplers):
                sampler.bias_bio = b

    def convert_weights_bio_to_theo(self, weights):
        # the column index denotes the target neuron, hence we convert there
        for j, sampler in enumerate(self.samplers):
            weights[:, j] = sampler.convert_weights_bio_to_theo(weights[:, j])
        return weights

    def convert_weights_theo_to_bio(self, weights):
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
                failed.append(sampler.db_params.id)

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

    ################
    # MISC methods #
    ################

    def save(self, filename):
        """
            Save the current Boltzmann network in zipped-pickle form.

            The pickle will contain current spike_data but nothing that can be
            recomputed rather quickly such as distributions.

            NOTE: Neuron parameters and loaded calibrations will only be
            included as Ids in the database. So make sure to keep the same
            database around if you want to restore a boltzmann network.
        """
        utils.save_pickle(self, filename)

    @classmethod
    def load(cls, filename):
        """
            Returns successfully loaded boltzmann network or None.
        """
        try:
            return utils.load_pickle(filename)
        except IOError:
            if log.getEffectiveLevel() <= logging.DEBUG:
                log.debug(sys.exc_info()[0])
            return None


    #######################
    # PROBABILITY methdos #
    #######################

    # methods to gather data
    @meta.DependsOn()
    def spike_data(self, spike_data=None):
        """
            The spike data from which to compute distributions.
        """
        if spike_data is not None:
            assert "spiketrains" in spike_data
            assert "duration" in spike_data
            return spike_data
        else:
            # We are requesting data when there is None
            return None

    def gather_spikes(self, duration, dt=0.1, burn_in_time=100.):
        log.info("Gathering spike data in subprocess..")
        self.spike_data = gather_data.gather_network_spikes(self,
                duration=duration, dt=dt, burn_in_time=burn_in_time)

    @meta.DependsOn("spike_data")
    def ordered_spikes(self):
        log.info("Getting ordered spikes")
        return utils.get_ordered_spike_idx(self.spike_data["spiketrains"])

    @meta.DependsOn()
    def selected_sampler_idx(self, selected_sampler_idx):
        return np.array(list(set(selected_sampler_idx)), dtype=np.int)

    @meta.DependsOn("spike_data", "selected_sampler_idx")
    def dist_marginal_sim(self):
        """
            Marginal distribution computed from spike data.
        """
        log.info("Calculating marginal probability distribution for {} "
                "samplers.".format(len(self.selected_sampler_idx)))

        marginals = np.zeros((len(self.selected_sampler_idx),))

        for i in self.selected_sampler_idx:
            sampler = self.samplers[i]
            spikes = self.spike_data["spiketrains"][i]
            marginals[i] = len(spikes) * sampler.db_params.tau_refrac

        marginals /= self.spike_data["duration"]

        return marginals

    @meta.DependsOn("spike_data", "selected_sampler_idx")
    def dist_joint_sim(self):
        # tau_refrac per selected sampler
        tau_refrac_pss = np.array([self.samplers[i].db_params.tau_refrac
                for i in self.selected_sampler_idx])

        spike_ids = np.require(self.ordered_spikes["id"], requirements=["C"])
        spike_times = np.require(self.ordered_spikes["t"], requirements=["C"])

        return cutils.get_bm_joint_sim(spike_ids, spike_times,
                self.selected_sampler_idx, tau_refrac_pss,
                self.spike_data["duration"])

    @meta.DependsOn("selected_sampler_idx", "biases_theo", "weights_theo")
    def dist_marginal_theo(self):
        """
            Marginal distribution
        """
        ssi = self.selected_sampler_idx
        lc_biases = self.biases_theo[ssi]
        lc_weights = self.weights_theo[ssi][:, ssi]

        lc_biases = np.require(lc_biases, requirements=["C"])
        lc_weights = np.require(lc_weights, requirements=["C"])

        return cutils.get_bm_marginal_theo(lc_weights, lc_biases)
        # return self.get_dist_marginal_from_joint(self.dist_joint_theo)

    @meta.DependsOn("selected_sampler_idx", "biases_theo", "weights_theo")
    def dist_joint_theo(self):
        """
            Joint distribution for all selected samplers.
        """
        log.info("Calculating joint theoretical distribution for {} samplers."\
                .format(len(self.selected_sampler_idx)))

        ssi = self.selected_sampler_idx
        lc_biases = self.biases_theo[ssi]
        lc_weights = self.weights_theo[ssi][:, ssi]

        lc_biases = np.require(lc_biases, requirements=["C"])
        lc_weights = np.require(lc_weights, requirements=["C"])

        joint = cutils.get_bm_joint_theo(lc_weights, lc_biases)

        return joint

    ################
    # PLOT methods #
    ################

    @meta.plot_function("comparison_dist_marginal")
    def plot_dist_marginal(self, logscale=True, fig=None, ax=None):
        width = 1./3.

        idx = np.arange(self.dist_marginal_theo.size, dtype=np.int)

        if logscale:
            ax.set_yscale("log")
            min_val = min(self.dist_marginal_theo.min(),
                    self.dist_marginal_sim.min())

            # find corresponding exponent
            bottom = 10**np.floor(np.log10(min_val))
        else:
            bottom = 0.

        ax.bar(idx, height=self.dist_marginal_theo.flatten(), width=width,
                bottom=bottom,
                color="r", edgecolor="None", label="marginal theo")

        ax.bar(idx+width, height=self.dist_marginal_sim.flatten(), width=width,
                bottom=bottom,
                color="b", edgecolor="None", label="marginal sim")

        ax.legend(loc="best")

        ax.set_xlim(0, idx[-1]+2*width)

        ax.set_xlabel("sampler index $i$")
        ax.set_ylabel("$p_{ON}$(sampler $i$)")

    @meta.plot_function("comparison_dist_joint")
    def plot_dist_joint(self, logscale=True, fig=None, ax=None):
        width = 1./3.

        idx = np.arange(self.dist_joint_theo.size, dtype=np.int)

        if logscale:
            ax.set_yscale("log")
            min_val = min(self.dist_joint_theo.min(),
                    self.dist_joint_sim.min())

            # find corresponding exponent
            bottom = 10**np.floor(np.log10(min_val))
        else:
            bottom = 0.

        ax.bar(idx, height=self.dist_joint_theo.flatten(), width=width,
                bottom=bottom,
                color="r", edgecolor="None", label="joint theo")

        ax.bar(idx+width, height=self.dist_joint_sim.flatten(), width=width,
                bottom=bottom,
                color="b", edgecolor="None", label="joint sim")

        ax.legend(loc="best")

        ax.set_xlabel("state")
        ax.set_ylabel("probability")

        ax.set_xlim(0, idx[-1]+2*width)

        ax.set_xticks(idx+width)
        ax.set_xticklabels(labels=["\n".join(map(str, state))
            for state in np.ndindex(*self.dist_joint_theo.shape)])


    ################
    # PYNN methods #
    ################

    def create(self, duration, _nest_optimization=True):
        """
            Create the sampling network and return the pynn object.

            If population is not None it should have length `self.num_samplers`.
            Also, if you specify different samplers to have different
            pynn_models, make sure that the list of pynn_objects provided
            supports those!

            Returns the newly created or specified popluation object for the
            samplers.

            `_nest_optimization`: If True the network will try to use as few
            sources as possible with the nest specific `poisson_generator` type.
            For the coder's sanity
        """
        exec "import {} as sim".format(self.sim_name) in globals(), locals()

        assert self.all_samplers_same_model(),\
                "The samplers have different pynn_models."

        # only perform nest optimizations when we have nest as simulator and
        # the user requests it
        _nest_optimization = _nest_optimization and hasattr(sim, "nest")

        log.info("Setting up population.")
        population = sim.Population(self.num_samplers,
                getattr(sim, self.samplers[0].pynn_model)())

        for i, sampler in enumerate(self.samplers):
            local_pop = population[i:i+1]

            # if we are performing nest optimizations, the sources will be
            # created afterwards
            sampler.create(duration=duration, population=local_pop,
                    create_pynn_sources=not _nest_optimization)

        if _nest_optimization:
            # make sure the objects returned are referenced somewhere
            self._nest_sources, self._nest_projections =\
                    bb.create_nest_optimized_sources(
                    sim, self.samplers, population, duration)

        # we dont set any connections for weights that are == 0.
        weight_is = {}
        weight_is["exc"] = self.weights_bio > 0.
        weight_is["inh"] = self.weights_bio < 0.

        receptor_type = {"exc" : "excitatory", "inh" : "inhibitory"}

        global_delay = len(self.delays.shape) == 0

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

            weights = self.weights_bio.copy()
            # weights[np.logical_not(weight_is[wt])] = np.NaN

            if wt == "inh":
                weights = weights.copy() * -1

            # Not sure that array connector does what we want
            # self.projections[wt] = sim.Projection(population, population,
                    # connector=sim.ArrayConnector(weight_is[wt]),
                    # synapse_type=sim.StaticSynapse(
                        # weight=weights, delay=delays),
                    # receptor_type=receptor_type[wt])

            connection_list = []
            for i_pre, i_post in it.izip(*np.nonzero(weight_is[wt])):
                connection_list.append(
                        (i_pre, i_post, weights[i_pre, i_post], self.delays\
                            if global_delay else self.delays[i_pre, i_post])
                    )

            self.projections[wt] = sim.Projection(population, population,
                    synapse_type=sim.StaticSynapse(),
                    connector=sim.FromListConnector(
                        connection_list,
                        column_names=["weight", "delay"]),
                    receptor_type=receptor_type[wt])

        self.population = population 

        return self.population, self.projections


