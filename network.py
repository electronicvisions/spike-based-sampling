#!/usr/bin/env python
# encoding: utf-8

import collections as c
import itertools as it
import numpy as np
import logging
import sys
import copy
from pprint import pformat as pf

import pylab as p

from .logcfg import log
from . import db
from . import samplers
from . import utils
from . import cutils
from . import gather_data
from . import meta
from . import buildingblocks as bb
# from . import cells

# cells.patch_pynn()

@meta.HasDependencies
class BoltzmannMachineBase(object):
    """
        A set of samplers connected as Boltzmann machine.
    """

    def __init__(self, num_samplers, sim_name="pyNN.nest",
            sampler_config=None):
        """
            Sets up a Boltzmann network.

            sampler_config:
                Either one or a lilst of size `num_samplers` of
                SamplerConfiguration objects with which the samplers are
                initialized.
        """
        log.info("Creating new {}.".format(self.__class__.__name__))
        self.sim_name = sim_name
        self.num_samplers = num_samplers

        self.population = None
        self.projections = None

        self.auto_sync_biases = True

        if sampler_config is None:
            raise ValueError("Did not specify sampler parameters.")

        if isinstance(sampler_config, db.SamplerConfiguration):
            sampler_config = [sampler_config]\
                    * self.num_samplers

        self.samplers = [samplers.LIFsampler(npc)
                for npc in sampler_config]

        self.weights_theo = 0.
        # biases are set to zero automaticcaly by the samplers

        self.saturating_synapses_enabled = False
        self.delays = 0.1

    ########################
    # pickle serialization #
    ########################
    # generally we only save the ids of samplers and calibrations used
    # (we can be sure that only saved samplers are used in the BM-network as
    # there is no way to calibrate them from the BM-network)
    # plus record biases and weights
    # def __getstate__(self):
        # log.debug("Reading state information for pickling.")
        # state = {
                # "calibration_ids" : [sampler.get_calibration_id()
                    # for sampler in self.samplers],
                # "current_basename" : db.current_basename,
            # }
        # state["weights"] = self.weights_theo

        # state["biases"] = self.biases_theo

        # state["delays"] = self.delays

        # state["sim_name"] = self.sim_name
        # state["num_samplers"] = self.num_samplers
        # state["params_ids"] = [sampler.get_parameters_id()
                # for sampler in self.samplers]

        # state["saturating_synapses_enabled"] = self.saturating_synapses_enabled

        # state["tso_params"] = self.tso_params

        # return state

    # def __setstate__(self, state):
        # log.debug("Setting state information for unpickling.")

        # if state["current_basename"] != db.current_basename:
            # raise Exception("Database mismatch, this network should be "
            # "restored with db {}".format(state["current_basename"]))

        # self.__init__(state["num_samplers"],
                # sim_name=state["sim_name"],
                # neuron_parameters_db_ids=state["params_ids"])

        # for i, cid in enumerate(state["calibration_ids"]):
            # if cid is not None:
                # self.samplers[i].load_calibration(id=cid)

        # self.weights_theo = state["weights"]
        # self.biases_theo = state["biases"]

        # self.delays = state["delays"]

        # self.tso_params = state["tso_params"]

        # self.saturating_synapses_enabled = state["saturating_synapses_enabled"]

    ######################
    # regular attributes #
    ######################
    @meta.DependsOn()
    def sim_name(self, name):
        """
            The full simulator name.
        """
        if not name.startswith("pyNN."):
            name = "pyNN." + name
        return name

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

    @meta.DependsOn()
    def saturating_synapses_enabled(self, value):
        """
            Use TSO to model saturating synapses between neurons.
        """
        assert isinstance(value, bool)
        return value

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
                if self.is_created and self.auto_sync_biases:
                    sampler.sync_bias_to_pynn()

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
                if self.is_created and self.auto_sync_biases:
                    sampler.sync_bias_to_pynn()

    def convert_weights_bio_to_theo(self, weights):
        conv_weights = np.zeros_like(weights)
        # the column index denotes the target neuron, hence we convert there
        for j, sampler in enumerate(self.samplers):
            conv_weights[:, j] = sampler.convert_weights_bio_to_theo(weights[:, j])
        return conv_weights

    def convert_weights_theo_to_bio(self, weights):
        conv_weights = np.zeros_like(weights)
        # the column index denotes the target neuron, hence we convert there
        for j, sampler in enumerate(self.samplers):
            conv_weights[:, j] = sampler.convert_weights_theo_to_bio(weights[:, j])

        return conv_weights

    @meta.DependsOn()
    def delays(self, delays):
        """
            Delays can either be a scalar to indicate a global delay or an
            array to indicate the delays between the samplers.
        """
        if self.is_created:
            log.warn("A PyNN object was already created. Its delays will not "
                    "be modified!")
        delays = self._check_delays(delays)
        return delays

    @meta.DependsOn()
    def tau_refracs(self):
        """
            Collects all tau_refracs from all samplers.

            Note: Assumes they do not change over the course of simulation!
        """
        return np.array([s.neuron_parameters.tau_refrac for s in self.samplers])

    @meta.DependsOn()
    def tso_params(self, params=None):
        """
            Specify custom TSO parameters.

            (Taken from NEST source doctstrings:)
             U          double - probability of release increment (U1) [0,1], default=0.5
             u          double - Maximum probability of release (U_se) [0,1], default=0.5
             x          double - current scaling factor of the weight, default=U
             tau_rec    double - time constant for depression in ms, default=800 ms
             tau_fac    double - time constant for facilitation in ms, default=0 (off)
        """
        if params is None:
            return {"U": 1., "u": 1., "x" : 0.}
        else:
            return params

    def all_samplers_same_model(self):
        """
            Returns true of all samplers have the same pynn model.

            If this returns False, expect `self.population` to be a list of
            size-1 populations unless specified differently during creation.
        """
        return all(
            ((sampler.pynn_model == self.samplers[0].pynn_model)\

                for sampler in self.samplers))

    @property
    def is_created(self):
        return self.population is not None

    ################
    # PYNN methods #
    ################

    def create_no_return(self, sim_setup_kwargs=None, *args, **kwargs):
        """
            Runs self.create(*args, **kwargs) but does not return the created
            PyNN objects.

            This is only useful when wrapping the class in another subprocess.

            If sim_setup_kwargs is not None, the dictionary will be used to
            setup PyNN.
        """
        if sim_setup_kwargs is not None:
            exec "import {} as sim".format(self.sim_name) in globals(), locals()
            sim.setup(**sim_setup_kwargs)
        self.create(*args, **kwargs)

    def create(self, duration=None, _nest_optimization=True,
            _nest_source_model=None, _nest_source_model_kwargs=None):
        """
            Create the sampling network and return the pynn object.

            If population is not None it should have length `self.num_samplers`.
            Also, if you specify different samplers to have different
            pynn_models, make sure that the list of pynn_objects provided
            supports those!

            Returns the newly created or specified popluation object for the
            samplers and a dictionary over the projections.

            `_nest_optimization`: If True the network will try to use as few
            sources as possible with the nest specific `poisson_generator` type.

            If a different source model should be used, it can be specified via
            _nest_source_model (string) and the corresponding kwargs.
            If the source model needs a parrot neuron that repeats its spikes
            in order to function, please note it.
        """
        assert duration is not None, "Duration must be set!"

        exec "import {} as sim".format(self.sim_name) in globals(), locals()

        assert self.all_samplers_same_model(),\
                "The samplers have different pynn_models."

        # only perform nest optimizations when we have nest as simulator and
        # the user requests it
        _nest_optimization = _nest_optimization and hasattr(sim, "nest")

        log.info("Setting up population for duration: {}s".format(duration))

        log.info("PyNN-model: {}".format(self.samplers[0].pynn_model))
        population = sim.Population(self.num_samplers,
                getattr(sim, self.samplers[0].pynn_model)())

        for i, sampler in enumerate(self.samplers):
            local_pop = population[i:i+1]

            # if we are performing nest optimizations, the sources will be
            # created afterwards
            sampler.create(duration=duration, population=local_pop,
                    create_pynn_sources=not _nest_optimization)

        if _nest_optimization:
            log.info("Creating nest sources of type {}.".format(_nest_source_model))

            # make sure the objects returned are referenced somewhere
            self._nest_sources, self._nest_projections =\
                    bb.create_nest_optimized_sources(
                    sim, self.samplers, population, duration,
                    source_model=_nest_source_model,
                    source_model_kwargs=_nest_source_model_kwargs)

        self.population = population

        return self.population, None


    ####################
    # INTERNAL methods #
    ####################

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

    def _check_delays(self, delays):
        delays = np.array(delays)

        if len(delays.shape) == 0:
            scalar_delay = delays
            delays = np.empty((self.num_samplers, self.num_samplers))
            delays.fill(scalar_delay)

        return delays


@meta.HasDependencies
class ThoroughBM(BoltzmannMachineBase):
    """
        A BoltzmannMachine focused on getting thorough representations of
        probability distributions.
    """

    def __init__(self, *args, **kwargs):
        super(ThoroughBM, self).__init__(*args, **kwargs)
        self.selected_sampler_idx = range(self.num_samplers)

    ################
    # PyNN methods #
    ################

    def create(self, **kwargs):
        super(ThoroughBM, self).create(**kwargs)

        exec "import {} as sim".format(self.sim_name) in globals(), locals()

        _nest_optimization = kwargs.get("_nest_optimization", True)\
                and hasattr(sim, "nest")

        # we dont set any connections for weights that are == 0.
        weight_is = {}
        weight_is["exc"] = self.weights_bio > 0.
        weight_is["inh"] = self.weights_bio < 0.

        receptor_type = {"exc" : "excitatory", "inh" : "inhibitory"}

        global_delay = len(self.delays.shape) == 0

        column_names = ["weight", "delay"]

        tau_rec_overwritten = "tau_rec" in self.tso_params

        if self.saturating_synapses_enabled:
            log.info("Creating saturating synapses.")
            if not tau_rec_overwritten:
                column_names.append("tau_rec")
                tau_rec = []
                for sampler in self.samplers:
                    pynn_params = sampler.get_pynn_parameters()
                    tau_rec.append({
                            "exc" : pynn_params["tau_syn_E"],
                            "inh" : pynn_params["tau_syn_I"],
                        })
            else:
                log.info("TSO: tau_rec overwritten.")
        else:
            log.info("Creating non-saturating synapses.")

        self.projections = {}
        for wt in ["exc", "inh"]:
            if weight_is[wt].sum() == 0:
                # there are no weights of the current type, continue
                continue

            log.info("Connecting {} weights.".format(receptor_type[wt]))

            weights = self.weights_bio.copy()
            # weights[np.logical_not(weight_is[wt])] = np.NaN

            if wt == "inh":
                weights *= (np.array([("_curr_" in s.pynn_model)
                    for s in self.samplers], dtype=int) * 2) - 1

            if self.saturating_synapses_enabled and _nest_optimization:
                # using native nest synapse model, we need to take care of
                # weight transformations ourselves
                weights *= 1000.

            # Not sure that array connector does what we want
            # self.projections[wt] = sim.Projection(population, population,
                    # connector=sim.ArrayConnector(weight_is[wt]),
                    # synapse_type=sim.StaticSynapse(
                        # weight=weights, delay=delays),
                    # receptor_type=receptor_type[wt])

            connection_list = []
            for i_pre, i_post in it.izip(*np.nonzero(weight_is[wt])):
                connection = (i_pre, i_post, weights[i_pre,i_post],
                    self.delays if global_delay else self.delays[i_pre, i_post])
                if self.saturating_synapses_enabled and not tau_rec_overwritten:
                    connection += (tau_rec[i_post][wt],)
                connection_list.append(connection)

            if self.saturating_synapses_enabled:
                if not _nest_optimization:
                    tso_params = copy.deepcopy(self.tso_params)
                    try:
                        del tso_params["u"]
                        del tso_params["x"]
                    except KeyError:
                        pass
                    synapse_type = sim.TsodyksMarkramSynapse(weight=0.,
                            **tso_params)
                else:
                    log.info("Using 'tsodyks2_synapse' native synapse model.")
                    synapse_type = sim.native_synapse_type("tsodyks2_synapse")(
                            **self.tso_params)


            else:
                synapse_type = sim.StaticSynapse(weight=0.)

            self.projections[wt] = sim.Projection(self.population, self.population,
                    synapse_type=synapse_type,
                    connector=sim.FromListConnector(connection_list,
                        column_names=column_names),
                    receptor_type=receptor_type[wt])

        return self.population, self.projections


    ########################
    # pickle serialization #
    ########################
    # def __getstate__(self):
        # state = super(ThoroughBM, self).__getstate__(self)

        # state["selected_sampler_idx"] = self.selected_sampler_idx
        # state["spike_data"] = self.spike_data

        # return state

    # def __setstate__(self, state):
        # super(ThoroughBM, self).__setstate__(state)

        # self.spike_data = state["spike_data"]
        # self.selected_sampler_idx = state["selected_sampler_idx"]



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

    def gather_spikes(self, duration, dt=0.1, burn_in_time=100.,
            create_kwargs=None, sim_setup_kwargs=None, initial_vmem=None):
        """
            sim_setup_kwargs are the kwargs for the simulator (random seeds).

            initial_vmem are the initialized voltages for all samplers.
        """
        log.info("Gathering spike data in subprocess..")
        self.spike_data = gather_data.gather_network_spikes(self,
                duration=duration, dt=dt, burn_in_time=burn_in_time,
                create_kwargs=create_kwargs,
                sim_setup_kwargs=sim_setup_kwargs,
                initial_vmem=initial_vmem)

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
            marginals[i] = len(spikes) * sampler.neuron_parameters.tau_refrac

        marginals /= self.spike_data["duration"]

        return marginals

    @meta.DependsOn("spike_data", "selected_sampler_idx")
    def dist_joint_sim(self):
        # tau_refrac per selected sampler
        tau_refrac_pss = np.array([self.samplers[i].neuron_parameters.tau_refrac
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

    @meta.plot_function("weights_theo")
    def plot_weights_theo(self, fig=None, ax=None):
        self._plot_weights(self.weights_theo, self.biases_theo,
                label="theoretical values", fig=fig, ax=ax)

    @meta.plot_function("weights_bio")
    def plot_weights_bio(self, fig=None, ax=None):
        self._plot_weights(self.weights_bio, self.biases_theo,
                label="biological values", fig=fig, ax=ax)

    ####################
    # INTERNAL methods #
    ####################

    def _plot_weights(self, weights, biases, label="", cmap="jet", fig=None, ax=None):
        cmap = p.get_cmap(cmap)

        matrix = weights.copy()
        for i in xrange(matrix.shape[0]):
            matrix[i, i] = biases[i]

        imshow = ax.imshow(matrix, cmap=cmap, interpolation="nearest")
        cbar = fig.colorbar(imshow, ax=ax)
        cbar.set_label(label)

        ax.set_xlabel("sampler id")
        ax.set_ylabel("sampler id")


@meta.HasDependencies
class RapidBMBase(BoltzmannMachineBase):
    """
        Rapid version of the regular BoltzmannMachine, intended for usage with
        learning algorithms and rapid weight changes.

        Currently only the NEST backend is supported.
    """
    nest_synapse_type = "cd_connection"

    def __init__(self, *args, **kwargs):
        super(RapidBMBase, self).__init__(*args, **kwargs)
        self._binary_state_set_externally = False
        self._sim = None
        self.time_current = 0.1

        self.time_sim_step = 30. # ms
        self.time_wipe = 50. # time between silence and imprint
        self.time_imprint = 30. # how long are the 
        self._update_num = 0

        # has to be non-None for the propagation to binary state to work
        self.last_spiketimes = np.zeros(self.num_samplers) - 1000.

        self._ensure_cd_pynn_models()

        # needed for the CD based updates (set automatically when update is
        # queued)
        self._force_spike_before_run = False
        self.time_force_spike = 30.

        # shape: (num_factors, 2) first is eta, second is data, third is
        # model/recon
        self.update_factors = np.zeros((self.num_samplers, 3))

    def create(self, connectivity_matrix=None, **kwargs):
        """
            (See also: BoltzmannMachineBase.create)

            If connectivity_matrix is a boolean array it can be used to specify
            which synapses should initially be connected (for learning).

            Otherwise the connectivity_matrix is inferred from the current
            weight configuration.
        """
        exec "import {} as sim".format(self.sim_name) in globals(), locals()

        self._sim = sim

        assert hasattr(self._sim, "nest"), "Only works with NEST."

        kwargs["duration"] = self._sim.nest.GetKernelStatus()["T_max"]

        super(RapidBMBase, self).create(**kwargs)

        self._sampler_gids = self.population.all_cells.tolist()

        self.last_spiketime_detector = self._sim.Population(1,
                self._sim.native_cell_type("last_spike_detector")())

        self._proj_pop_lsd = self._sim.Projection(self.population,
                self.last_spiketime_detector, self._sim.AllToAllConnector())

        log.info("Creating imprint circuitry…")
        self._create_imprint_circuitry()

        log.info("Connecting samplers…")
        self._create_connectivity(connectivity_matrix=connectivity_matrix)

        self.update_samplers()

        return self.population, None

    def queue_update(self):
        """
            Indicate that the synapses should update in the next run.
        """
        self._force_spike_before_run = True
        self._update_num += 1

    def continue_run(self, runtime):
        """
            Continue running the network for the specified time without
            imprinting/wiping any state.
        """
        self.time_current = self._sim.run_for(runtime)

    def run(self):
        """
            Run the network for self.time_sim_step milliseconds; after that the
            binary state can be inspected.
        """
        time_till = self.prepare_run()
        self._sim.run_until(time_till)
        self.process_run()
        return self.time_current

    def prepare_run(self):
        """
            When using several RapidBMBase at the same time, manually
            set up a run with this function.

            Returns the needed time for which the simulation needs to be run.

            Do not forget to call process_run after the run is complete.

            Returns the time until which the sim has to be run.
        """
        self.update_samplers()
        # self.update_weights()
        # self.update_biases()
        time_force_spike = None
        if self._force_spike_before_run:
            time_force_spike = self._prepare_force_spike()
            self._force_spike_before_run = False

        total_time = time_force_spike
        if self._binary_state_set_externally:
            total_time = self._prepare_imprint(time_force_spike)

        if total_time is None:
            total_time = self.time_current + self.time_sim_step

        return total_time

    def process_run(self):
        """
            After every manual run, call this function to process the new
            information.
        """
        self.time_current = self._sim.simulator.state.t

    def update_weights(self):
        weights = self.weights_bio.copy() * 1000. # convert to nest manually

        weights = weights[self.connectivity_matrix]

        # for conn, weight in it.izip(self._nest_connections, weights):
            # self._sim.nest.SetStatus([conn], {"weight" : weight})
        self._sim.nest.SetStatus(self._nest_connections, "weight", weights)

    def get_sampler_update(self):
        """
            Should return a dictionary with list or single values to indicate
            what updated parameters are written to the samplers.
        """
        update = []
        for i,s in enumerate(self.samplers):
            update.append({
                "E_L": s.get_v_rest_from_bias(),
                "factor_weight_conversion_exc" :
                    s.factor_weights_theo_to_bio_exc * 1000.,
                "factor_weight_conversion_inh":
                    s.factor_weights_theo_to_bio_inh * 1000.,
                "factor_eta" : self.update_factors[i, 0],
                "factor_data" : self.update_factors[i, 1],
                "factor_model" : self.update_factors[i, 2],
                "update_num" : self._update_num,
            })
        return update

    def update_samplers(self):
        update = self.get_sampler_update()
        self._sim.nest.SetStatus(self.population.all_cells.tolist(), update)

    ##############
    # Properties #
    ##############

    @meta.DependsOn()
    def time_current(self, time):
        """
            Current simulation time.
        """
        return time

    @meta.DependsOn()
    def time_sim_step(self, value):
        """
            The length of one simulation step.
        """
        return value

    @meta.DependsOn()
    def time_wipe(self, value):
        """
            The length of one simulation step.
        """
        return value

    @meta.DependsOn()
    def time_imprint(self, value):
        """
            The length of one simulation step.
        """
        return value

    @meta.DependsOn()
    def time_force_spike(self, value):
        """
            The length of one simulation step.
        """
        return value

    @meta.DependsOn("time_current")
    def last_spiketimes(self, value=None):
        # this is only done so that a non-None value can be set during __init__
        if value is not None:
            return value

        indices, times = self._sim.nest.GetStatus(
                self.last_spiketime_detector.all_cells.tolist(),
                ["indices", "times"])[0]

        if times.size != self.population.size:
            # subtract offset if there were other neurons created beforehand
            # this assumes that the population was created at once and is
            # consecutively indexed
            indices -= int(self._sampler_gids[0])

            old_times = times
            times = np.zeros(self.population.size)

            times[indices] = old_times
        return times

    @meta.DependsOn("last_spiketimes")
    def binary_state(self, state=None):
        """
            Binary state imprinted on the network.

            Updated after every run based on the last spike times.

            Can also be set externally to 0/1:
                0: Clamped off
                1: Clamped on

            Any other value marks the state as undefined and it will not be
            enforced.
        """
        if state is None:
            state = self.time_current - self.last_spiketimes < self.tau_refracs
            return np.array(state, dtype=int)

        else:
            self._binary_state_set_externally = True
            return state

    @meta.DependsOn()
    def calibration_data(self, value=None):
        # value is ignored, can be used to recalculate

        # First column: alpha
        # Second column: offset
        calib_data = np.empty((self.num_samplers, 2), dtype=np.float64)
        for i, sampler in enumerate(self.samplers):
            calib_data[i, 0] = sampler.calibration.alpha
            calib_data[i, 1] = sampler.calibration.v_p05

        return calib_data

    @meta.DependsOn("time_current", "weights_bio")
    def weights_theo(self, value=None):
        if value is None:
            assert self.is_created
            return self._format_weights(np.array(self._sim.nest.GetStatus(
                self._nest_connections, "weight_theo")))
        else:
            value = self._check_weight_matrix(value)
            if self.is_created:
                self._write_weights(value, kind="theo")
            return value

    @meta.DependsOn("time_current", "weights_theo")
    def weights_bio(self, value=None):
        if value is None:
            assert self.is_created
            return self._format_weights(np.array(self._sim.nest.GetStatus(
                self._nest_connections, "weight"))) / 1000.
        else:
            value = self._check_weight_matrix(value)
            if self.is_created:
                self._write_weights(value, kind="bio")
            return value


    ####################
    # INTERNAL methods #
    ####################

    def _format_weights(self, weights):
        return weights.reshape(self.num_samplers, self.num_samplers)

    def _write_weights(self, weights, kind="theo"):
        assert self.is_created
        if kind == "theo":
            label = "weight_theo"
        elif kind == "bio":
            label = "weight"
        else:
            raise ValueError("Invalid weight type supplied.")

        data = [{label: w} for w in weights.reshape(-1)]
        self._sim.nest.SetStatus(self._nest_connections, data)

    def _create_imprint_circuitry(self):
        raise NotImplementedError

    def _prepare_imprint(self, wipe_start=None):
        raise NotImplementedError

    def _prepare_force_spike(self, time_start=None):
        raise NotImplementedError



    def _create_connectivity(self, connectivity_matrix=None):
        # TODO: Add support for TSO
        if self.saturating_synapses_enabled:
            log.warn(self.__class__.__name__ + " currently does not support "\
                    "saturating synapses.")

        global_delay = len(self.delays.shape) == 0

        nest = self._sim.nest

        if connectivity_matrix is None:
            connectivity_matrix = self.weights_bio != 0.

        else:
            assert connectivity_matrix.shape == (self.population.size,) * 2
            assert connectivity_matrix.dtype == np.bool

        self.connectivity_matrix = connectivity_matrix

        gids = self._sampler_gids

        for src, tgt in it.izip(*np.where(connectivity_matrix)):
            nest.Connect(gids[src], gids[tgt], syn_spec={
                "weight" : 0.,
                "delay"  : self.delays[src, tgt] if not global_delay
                            else self.delays,
                "model" : self.nest_synapse_type})

        self._nest_connections = nest.GetConnections(gids, gids)

    def _ensure_cd_pynn_models(self):
        for i, sampler in enumerate(self.samplers):
            if not sampler.pynn_model.endswith("_cd"):
                new_model = sampler.pynn_model + "_cd"
                log.warn("[Sampler #{}] Changing model from {} -> {}…".format(i,
                    sampler.pynn_model, new_model))
                sampler.neuron_parameters.pynn_model = new_model


@meta.HasDependencies
class RapidBMCurrentImprint(RapidBMBase):
    """
        Rapid Boltzmann machine that imprints the needed network state via
        current stimulation.
    """
    @meta.DependsOn()
    def current_imprint(self, current=None):
        """
            Current with which the network state is imprinted [nA].
        """
        if current is None:
            # if current wasn't set return the default one
            return 10.

        return current

    @meta.DependsOn()
    def current_wipe(self, current=None):
        """
            Current with which the network state is imprinted [nA].
        """
        if current is None:
            current = 10.

        if hasattr(self, "_wipe_gen_id"):
            self._sim.nest.SetStatus(self._wipe_gen_id, {
                "amplitude": -1 * np.abs(current) * 1000.,
                "start" : 0.,
                "stop" : 0.,
            })
        return current

    @meta.DependsOn()
    def current_force_spike(self, current=None):
        """
            Current with which the network state is imprinted [nA].
        """
        if current is None:
            return 10.

        if hasattr(self, "_force_spike_gen_id"):
            self._sim.nest.SetStatus(self._force_spike_gen_id, {
                "amplitude": np.abs(current) * 1000.,
                "start" : 0.,
                "stop" : 0.,
            })
        return current

    def _prepare_force_spike(self, time_start=None):
        if time_start is None:
            time_start = self.time_current + 2*self._sim.simulator.state.dt

        time_stop = time_start + self.time_force_spike

        self._sim.nest.SetStatus(self._force_spike_gen_id, {
                "start" : time_start,
                "stop" : time_stop,
            })
        return time_stop


    def _prepare_imprint(self, wipe_start=None):
        binary_state = self.binary_state

        imprint_idx = np.where((binary_state == 0)\
                + (self.binary_state == 1))[0]

        if wipe_start is None:
            wipe_start = self.time_current + 2*self._sim.simulator.state.dt

        imprint_start = wipe_start + self.time_wipe
        imprint_stop = imprint_start + self.time_imprint

        self._sim.nest.SetStatus(self._wipe_gen_id, {
                "start" : wipe_start,
                "stop" : imprint_start,
            })

        if log.getEffectiveLevel() <= logging.DEBUG:
            log.debug("Wipe start/stop: {:.1f}/{:.1f}".format(wipe_start,
                imprint_start))
            log.debug("Imprint start/stop: {:.1f}/{:.1f}".format(imprint_start,
                imprint_stop))

        for state in [0, 1]:
            imprint_idx = np.array(binary_state == state, dtype=int)

            self._sim.nest.SetStatus(
                self._imprint_gen_ids[imprint_idx].tolist(), {
                "start" : imprint_start,
                "stop" : imprint_stop,
                "amplitude" : self.current_imprint * 1000. * (2*state - 1),
            })

        return imprint_start + self.time_sim_step

    def _create_imprint_circuitry(self):
        # dc generator that inhibits all samplers to imprint a new state
        self._wipe_gen_id = self._sim.nest.Create("dc_generator")
        self._force_spike_gen_id = self._sim.nest.Create("dc_generator")

        # dc generators that imprint the actual network state
        self._imprint_gen_ids = np.array(
                self._sim.nest.Create("dc_generator", self.population.size))

        # this writes the amplitude to the nest objects
        self.current_wipe = self.current_wipe
        self.current_force_spike = self.current_force_spike

        self._sim.nest.Connect(self._wipe_gen_id,
                self.population.all_cells.tolist(), "all_to_all")
        self._sim.nest.Connect(self._force_spike_gen_id,
                self.population.all_cells.tolist(), "all_to_all")
        self._sim.nest.Connect(self._imprint_gen_ids.tolist(),
                self.population.all_cells.tolist(), "one_to_one")


@meta.HasDependencies
class RapidBMSpikeImprint(RapidBMBase):
    """
        WIP - DO NOT USE!

        Version of the Rapid Boltzmann machine that imprints the current state
        via external spikes only.
    """

    def __init__(self, *args, **kwargs):
        super(RapidBMSpikeImprint, self).__init__(*args, **kwargs)
        # the weight with the current binary state is imprinted on the network
        self.imprint_weight_theo = 50.
        self.num_wipe_spikes = 1
        self.time_current = 0.1 # so that the imprint spikes are set properly

    def _create_imprint_circuitry(self):
        self._imprint_gen_ids = np.array(
                self._sim.nest.Create("spike_generator", self.population.size))
        self._sim.nest.Connect(self._imprint_gen_ids.tolist(),
                self.population.all_cells.tolist(), 'one_to_one')

    @meta.DependsOn("imprint_weight_theo")
    def imprint_weights_bio(self):
        """
            Array of shape (n, 2) where the first column is inhibitory, the
            second the excitatory bio weight.

            The row denots the sampler.
        """
        weights_theo = np.array([[-1.], [1.]]) * self.imprint_weight_theo

        weights_theo = np.repeat(weights_theo, len(self.samplers), axis=1)

        weights_bio = self.convert_weights_theo_to_bio(weights_theo)

        return weights_bio.T

    @meta.DependsOn()
    def imprint_weight_theo(self, weight):
        """
            Weight with which the network state is imprinted.
        """
        return weight

    def _prepare_imprint(self):
        imprint_weights = self.imprint_weights_bio * 1000.
        binary_state = self.binary_state

        wipe_time_start = self.time_current + 2*self._sim.simulator.state.dt
        wipe_times = np.linspace(0., self.time_wipe, self.num_wipe_spikes,
                endpoint=False)
        wipe_times += wipe_time_start

        imprint_time = wipe_time_start + self.time_wipe

        imprint_idx = np.where((binary_state == 0)\
                + (self.binary_state == 1))[0]

        # update all stimulated neurons
        for i, gid in enumerate(self._imprint_gen_ids[imprint_idx]):
            spike_weights = imprint_weights[i:i+1, [0]*len(wipe_times)
                    + [binary_state[i]]].flatten()
            spike_weights[:-1] /= self.num_wipe_spikes
            self._sim.nest.SetStatus(
                [gid], {
                "spike_times" : np.r_[wipe_times, np.array([imprint_time])],
                # "spike_times" : spike_time,
                "spike_weights" : spike_weights,
            })

        # update all unstimulated ones
        for i, gid in enumerate(
                self._imprint_gen_ids[np.logical_not(imprint_idx)]):
            spike_weights = imprint_weights[i:i+1, [0]*len(wipe_times)]\
                        /self.num_wipe_spikes
            self._sim.nest.SetStatus(
                [gid], {
                "spike_times" : wipe_times,
                # "spike_times" : spike_time,
                "spike_weights" : spike_weights.flatten()
            })

        self._binary_state_set_externally = False


class MixinRBM(object):
    """
        Mixin with some conviencience functions for dealing with multilayer
        RBMs.

        Note that the weight matrices have a new format here to conserve space:
        * There are several weight matrices in a list (since the layers can
          have different sizes).

        Theoretical weights:
        * The i-th entry in this list has the shape (n_layer_i, n_layer_i+1)
        * This is because the theoretical weights have to be symmetric.

        Biological weights:
        * The i-th entry in this list has the shape (2, n_layer_i, n_layer_i+1)
        * The entries in the first row describe the connections from the i-th
          layer to the (i+1)-th, the second row in the opposite direction.
        * This is due to the fact that samplers can have different parameters
          and so the conversion for the same theoretical weight can lead to two
          different weights.

        Setting biological weights directly is currently not supported.
    """

    def __init__(self, num_units_per_layer=None, *args, **kwargs):

        assert num_units_per_layer is not None
        assert len(num_units_per_layer) > 1, "Need to have at least two layers"

        self.num_units_per_layer = num_units_per_layer
        self._layer_id_offset = np.r_[0, np.cumsum(self.num_units_per_layer)]
        self.num_layers = len(num_units_per_layer)

        kwargs["num_samplers"] = sum(num_units_per_layer)

        super(MixinRBM, self).__init__(*args, **kwargs)

    def convert_weights_theo_to_bio(self, weights, out=None):
        if out is None:
            conv_weights = [np.zeros(w.shape) for w in weights]
        else:
            conv_weights = out

        id_offset = self._layer_id_offset

        for i_l in xrange(self.num_layers-1):
            l_weights = conv_weights[i_l]
            l_theo_weights = weights[i_l]

            # conversion of first layer to second
            for j, sampler in enumerate(
                    self.samplers[id_offset[i_l+1]:id_offset[i_l+2]]):
                l_weights[0, :, j] = sampler.convert_weights_theo_to_bio(
                        l_theo_weights[0, :, j])

            # conversion of second layer to first
            for j, sampler in enumerate(
                    self.samplers[id_offset[i_l]:id_offset[i_l+1]]):
                l_weights[1, j, :] = sampler.convert_weights_theo_to_bio(
                        l_theo_weights[1, j, :])

        return conv_weights

    def convert_weights_bio_to_theo(self, weights):
        log.error(
            "Setting biological weights directly is currently not supported.")
        return None

    def update_weights(self):

        # # on the first run it computes the weights twice, but that is okay
        # weights = self.convert_weights_theo_to_bio(self.weights_theo,
                # out=self.weights_bio)
        # weights = [w * 1000. for layer_w in self.weights_bio
                # for w in layer_w.reshape(-1)]
        log.info("Converting and loading weights…")
        params = [{"weight": w} for w in it.chain(*(w.reshape(-1)
            for w in self.weights_bio))]

        # for conn, weight in it.izip(self._nest_connections, weights):
            # self._sim.nest.SetStatus([conn], {"weight" : weight})
        log.info("Sending weights to NEST…")
        self._sim.nest.SetStatus(self._nest_connections, params)
        log.info("Done updating weights.")

    def _check_delays(self, delays):
        if np.isscalar(delays):
            global_delay = float(delays)
            delays = [global_delay * np.ones((2,
                self.num_units_per_layer[i_l],
                self.num_units_per_layer[i_l+1]))
                    for i_l in xrange(self.num_layers-1)]

        else:
            assert isinstance(delays, list)

        return delays

    def _format_weights(self, weights):
        orig_weights = weights
        upl = self.num_units_per_layer
        nupl = np.array(self.num_units_per_layer)
        offset = nupl[1:] * nupl[:-1] * 2
        offset = np.r_[0, np.cumsum(offset)]

        weights = []
        for i in xrange(self.num_layers-1):
            lw = orig_weights[offset[i]:offset[i+1]].reshape(2,
                upl[i], upl[i+1])
            # the connections in nest are sorted the other way around
            # lw[1, :, :] = lw[1, :, :].T.reshape(lw.shape[1:])
            weights.append(lw)

        return weights

    def _create_connectivity(self, connectivity_matrix=None):
        # TODO: Add support for TSO
        if self.saturating_synapses_enabled:
            log.warn(self.__class__.__name__ + " currently does not support "\
                    "saturating synapses.")

        # we ignore the connectivity matrix
        self._nest_connections = []

        # if not global_delay:
            # log.error("Non-Global delays are currently NOT supported in RBMs.")

        nest = self._sim.nest

        offset = self._layer_id_offset

        gids = self._sampler_gids

        for i_l in xrange(self.num_layers-1):
            log.info("Creating connections from layers {} <-> {}".format(
                i_l, i_l+1))
            nest.Connect(
                    gids[offset[i_l]:offset[i_l+1]],
                    gids[offset[i_l+1]:offset[i_l+2]], 'all_to_all',
                    syn_spec={"model" : self.nest_synapse_type})
            nest.Connect(
                    gids[offset[i_l+1]:offset[i_l+2]],
                    gids[offset[i_l]:offset[i_l+1]], 'all_to_all',
                    syn_spec={"model" : self.nest_synapse_type})

        connections = nest.GetConnections(gids, gids)

        # readjust connections
        gids_to_conn = {(conn[0], conn[1]): conn for conn in connections}

        # append connections in the same order as they are in the first half
        # TODO: Evaluate possibility of cache trashing
        final_connections = list(connections[:len(connections)/2])
        for fc in connections[:len(connections)/2]:
            final_connections.append(gids_to_conn[(fc[1], fc[0])])

        self._nest_connections = final_connections

        log.info("Setting weights to zero.")
        nest.SetStatus(self._nest_connections, "weight", 0.)

        log.info("Setting delays.")
        nest.SetStatus(self._nest_connections, "delay",
                [d for l_delay in self.delays for d in l_delay.reshape(-1)])

        log.info("Done connecting.")

    def _check_weight_matrix(self, weights):

        if not isinstance(weights, list):
            assert np.isscalar(weights)

            scalar_weight = weights

            all_weights = []

            for i in xrange(self.num_layers-1):
                weights = np.empty((2, self.num_units_per_layer[i],
                            self.num_units_per_layer[i+1]))
                weights.fill(scalar_weight)

                all_weights.append(weights)

            return all_weights

        for i, w in enumerate(weights):
            expected_shape = (2, self.num_units_per_layer[i],
                    self.num_units_per_layer[i+1])
            assert w.shape == expected_shape,\
                    "Weight matrix shape {}, expected {} (layer {} <-> {})"\
                    .format(w.shape, expected_shape, i, i+1)
        return weights

    def _write_weights(self, weights, kind="theo"):
        # TODO Convert theoretical weights as well before writing
        assert self.is_created
        # if kind == "theo":
            # weight_theo = weights
            # weight_bio = self.convert_weights_theo_to_bio(weight_theo)
        # elif kind == "bio":
            # weight_bio = weights
            # weight_theo = self.convert_weights_bio_to_theo(weight_bio)
        # else:
            # raise ValueError("Invalid weight type supplied.")

        label = {
                "theo" : "weight_theo",
                "bio" : "weight",
            }[kind]
        # data = [{"weight": w_b, "weight_theo" : w_t}
                # for lwb, lwt in it.chain(*it.izip(weight_bio, weight_theo))
                # for w_b, w_t in it.izip(lwb.reshape(-1), lwt.reshape(-1))]
        data = [{label : w} for layer_w in weights for w in layer_w.reshape(-1)]
            # it.chain(layer_w[0].reshape(-1), layer_w[1].T.reshape(-1))}]
        self._sim.nest.SetStatus(self._nest_connections, data)


@meta.HasDependencies
class RapidRBMCurrentImprint(MixinRBM, RapidBMCurrentImprint):
    """
        RBM with current imprints.
    """
    pass

