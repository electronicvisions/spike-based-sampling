#!/usr/bin/env python
# encoding: utf-8

import collections as c
import itertools as it

from . import samplers

class BoltzmannMachine(object):
    """
        A set of samplers connected as Boltzmann machine.
    """

    def __init__(self, num_samplers, sim_name="pyNN.nest",
            neuron_model=None,
            neuron_params=None, neuron_index_to_params=None,
            neuron_params_db_ids=None):
        """
        Sets up a Boltzmann network.

        `neuron_model` is the string of the pyNN model used. Note that if
        neuron_parmas is a list, `pynn_neuron_model` also has to be.

        There are several ways to specify neuron_parameters:

        `neuron_params` as a single dictionary:
        ====
            All samplers will have the same parameters specified by
            neuron_params.

        `neuron_params` as a list of dictionaries of length `num_samplers`:
        ====
            Sampler `i` will have paramaters `neuron_params[i]`.

        `neuron_params` as a list of dictionaries of length < `num_samplers`
        and `neuron_index_to_params` is list of length `num_samplers` of ints:
        ====
            Sampler `i` will have parameters
            `neuron_params[neuron_index_to_params[i]]`.

        `neuron_params_db_ids` is a list of ints of length `num_samplers`:
        ====
            Sampler `i` will load its parameters from database entry with
            id `neuron_params_db_ids[i]`.
        """
        self.sim_name = sim_name
        self.num_samplers = num_samplers

        if isinstance(neuron_model, basestring):
            neuron_model = [neuron_model] * num_samplers

        if neuron_params is not None:
            if not isinstance(neuron_params, c.Sequence):
                neuron_params = [neuron_params]
                neuron_index_to_params = [0] * num_samplers

            elif neuron_index_to_params is None:
                neuron_index_to_params = range(num_samplers)

            self.samplers = [samplers.LIFsampler(
                sim_name=self.sim_name,
                neuron_model=neuron_model[i],
                neuron_parameters=neuron_params[i])\
                        for i in range(num_samplers)]

        elif neuron_params_db_ids is not None:
            self.samplers = [samplers.LIFsampler(id=id) for id in\
                    range(num_samplers)]
        else:
            raise Exception("Please provide either parameters or ids in the "
                    "database!")

        self.weights_theo = np.zeros((num_samplers, num_samplers))
        # biases are set to zero automaticcaly by the samplers

    @property
    def weights_theo(self):
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

    @weights_theo.setter
    def set_weights_theo(self, weights):
        self._weights_bio = None
        self._weights_theo = weights

    @weights_bio.setter
    def set_weights_bio(self, weights):
        self._weights_bio = weights
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
        # TODO: Implement me
        return weights

    def convert_weights_bio_to_theo(self, weights):
        weights = utils.fill_diagonal(weights, 0.)
        # TODO: Implement me
        return weights

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
            if not sampler.load_configuration(id=id):
                failed.append(samper.db_params.id)

        return failed

    def create(self, pynn_object=None):
        """
            Create the sampling network and return the pynn object.
        """
        # TODO: Implement me!
        return pynn_object

