#!/usr/bin/env python
# encoding: utf-8

import collections as c
import itertools as it
import numpy as np

from . import db
from . import samplers

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
        """
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
                neuron_parameters=neuron_parameters[i])\
                        for i in neuron_index_to_parameters]

        elif neuron_parameters_db_ids is not None:
            self.samplers = [samplers.LIFsampler(id=id) for id in
                    neuron_parameters_db_ids]
        else:
            raise Exception("Please provide either parameters or ids in the "
                    "database!")

        self.weights_theo = np.zeros((num_samplers, num_samplers))
        # biases are set to zero automaticcaly by the samplers

        self.delays = 0.

    ########################
    # pickle serialization #
    ########################
    # generally we only save the ids of samplers and calibrations used
    # (we can be sure that only saved samplers are used in the network as there
    # is no way to calibrated them from the network)
    # plus record biases and weights
    def __getinitargs__(self):
        args = (
                # num_samplers
                self.num_samplers,
                # sim_name
                self.sim_name,
                # pynn_model
                None,
                # neuron_parameters
                None,
                # neuron_index_to_parameters
                None,
                # neuron_parameters_db_ids
                [sampler.get_parameters_id() for sampler in self.samplers],
            )
        return args

    def __getstate__(self):
        state = {
                "calibration_ids" : [sampler.get_calibration_id()
                    for sampler in samplers],
            }
        # avoid unnecessary conversions for weights
        for wt in ["theo", "bio"]: # weight type
            weights = getattr(self, "_weights_{}".format(wt))
            if weights is not None:
                state["weights"] = {"type": wt, "value": weights}

        # biases are not as important so we just save the theoretical ones
        state["biases"] = [sampler.bias_theo for sampler in self.samplers]

        state["delays"] = self.delays

        return state

    def __setstate__(self, state):
        for i, cid in enumerate(state["calibration_ids"]):
            if cid is not None:
                self.samplers[i].load_calibration(id=cid)

        # restore whatever weight type we saved
        setattr(self, "weights_{}".format(state["weights"]["type"]),
                state["weights"]["value"])

        # set the biases
        for i, bias in enumerate(state["biases"]):
            self.samplers[i].bias_theo = bias

        self.delays = state["delays"]

    ######################
    # regular attributes #
    ######################

    @property
    def weights_theo(self):
        """
            Set or retrieve the connection weights
            Automatic conversion:
            After the weights have been set in either biological or theoretical
            units both can be retrieved and the conversion will be done when
            needed.
        """
        # converts the weights from bio only when user requests it
        if self._weights_theo is None:
            self._weights_theo = self.convert_weights_bio_to_theo(weights)
        return self._weights_theo

    @property
    def weights_bio(self):
        # converts the weights from theo only when user requests it
        if self._weights_bio is None:
            self._weights_bio = self.convert_weights_theo_to_bio(weights)
        return self._weights_bio

    def _check_weight_matrix(self, weights):
        weights = np.array(weights)
        assert weights.shape == (self.num_samplers, self.num_samplers),\
                "Weight matrix has wrong shape"
        return weights

    @weights_theo.setter
    def set_weights_theo(self, weights):
        self._weights_theo = self._check_weight_matrix(weights)
        self._weights_bio = None

    @weights_bio.setter
    def set_weights_bio(self, weights):
        self._weights_bio = self._check_weight_matrix(weights)
        self._weights_theo = None

    @property
    def biases_theo(self):
        return np.array([s.bias_theo for s in self.samplers])

    @property
    def biases_bio(self):
        return np.array([s.bias_bio for s in self.samplers])

    @biases_theo.setter
    def set_biases_theo(self, biases):
        if not isinstance(biases, c.Sequence):
            biases = it.repeat(biases)

        for b, sampler in it.izip(biases, self.samplers):
            sampler.bias_theo = b

    @biases_bio.setter
    def set_biases_bio(self, biases):
        if not isinstance(biases, c.Sequence):
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

    @property
    def delays(self):
        """
            Delays can either be a scalar to indicate a global delay or an
            array to indicate the delays between the samplers.
        """
        return self._delays

    @delays.setter
    def set_delays(self, delays):
        self._delays = np.array(delays)

    def load_single_calibration(self, id):
        """
            Load the same calibration for all samplers.

            Returns a list of sampler(-parameter) ids that failed.
        """
        failed = []
        for sampler in self.samplers:
            if not sampler.load_configuration(id=id):
                failed.append(samper.db_params.id)

        return failed

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
                    getattr(sim, self.samplers[0].pynn_model))

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

        connector = sim.AllToAllConnector()

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

            weights = self.weights[weight_is[wt]]

            if wt == "inh":
                weights = weights.copy() * -1

            self.projections[wt] = sim.Projection(population, population,
                    connector=connector,
                    delays=delays,
                    weights=weights,
                    receptor_type=receptor_type[wt])

        self.population = population 

        return self.population, self.projections


