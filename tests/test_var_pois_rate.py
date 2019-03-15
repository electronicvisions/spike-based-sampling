#!/usr/bin/env python2
# encoding: utf-8


import unittest
import numpy as np
from pprint import pformat as pf

import sbs
log = sbs.log

sim_name = "pyNN.nest"


def check_mpg():
    return sbs.utils.ensure_visionary_nest_model_available(
        "multi_poisson_generator")


@unittest.skipUnless(check_mpg(),
                     "requires multi poisson generator models from "
                     "visionary NEST module")
class TestVarPoisRate(unittest.TestCase):
    def setUp(self):
        import nest
        nest.ResetKernel()
        if "multi_poisson_generator" not in nest.Models():
            nest.Install("visionarymodule")

    def test_sample_network_var_poisson_rate_cond(self):
        """
            How to setup and evaluate a Boltzmann machine. Please note that in
            order to instantiate BMs all needed neuron parameters need to be in
            the database and calibrated.

        """
        np.random.seed(4241414)

        # Load calibration data in order to create network.
        sampler_config = sbs.db.SamplerConfiguration.load(
                "test-calibration-cond.json")

        # We set the variation behaviour of the rates via the source
        # configuration of the sampler configuration. If we do not set it
        # specifically, the source configuration from the calibration file
        # would be used. Since a calibration on an array of different rates is
        # not sensible, we set it here. We specify the weights, rates and times
        # of each poisson input of a sampler. Details about the correct syntax
        # are provided in the documentation of this source configuration class.

        # Define the rate changes of an excitatory Poisson source.
        rate_changes = np.array([[0., 1000.],
                                 [2000., 100.]])

        poisson_weights = np.array([0.001, -0.001])

        sampler_config.source_config = \
            sbs.db.MultiPoissonVarRateSourceConfiguration(
                weight_per_source=poisson_weights,
                rate_changes_per_source=[rate_changes] * len(poisson_weights))

        # Choose the number of samplers in the network.
        bm = sbs.network.ThoroughBM(num_samplers=5, sim_name=sim_name,
                                    sampler_config=sampler_config)

        # Choose weights (here random) and symmetrize them.
        weights = np.random.randn(bm.num_samplers, bm.num_samplers)
        weights = (weights + weights.T) / 2.
        bm.weights_theo = weights

        # Choose biases (here random).
        bm.biases_theo = np.random.randn(bm.num_samplers)

        # Sample the network and save it.
        bm.gather_spikes(
                duration=1e5,  burn_in_time=500., dt=0.1,
                sim_setup_kwargs=sbs.utils.get_default_setup_kwargs(sim_name))

        log.info("Weights (theo):\n" + pf(bm.weights_theo))
        log.info("Biases (theo):\n" + pf(bm.biases_theo))

        log.info("Weights (bio):\n" + pf(bm.weights_bio))
        log.info("Biases (bio):\n" + pf(bm.biases_bio))

        log.info("Spikes: {}".format(pf(bm.ordered_spikes)))

        log.info("Spike-data: {}".format(pf(bm.spike_data)))

        bm.selected_sampler_idx = range(bm.num_samplers)

    def test_sample_network_var_poisson_rate_curr(self):
        """
            How to setup and evaluate a Boltzmann machine. Please note that in
            order to instantiate BMs all needed neuron parameters need to be in
            the database and calibrated.

        """
        np.random.seed(421243)

        # Load calibration data in order to create network.
        sampler_config = sbs.db.SamplerConfiguration.load(
                "test-calibration-curr.json")

        # We set the variation behaviour of the rates via the source
        # configuration of the sampler configuration. If we do not set it
        # specifically, the source configuration from the calibration file
        # would be used. Since a calibration on an array of different rates is
        # not sensible, we set it here. We specify the weights, rates and times
        # of each poisson input of a sampler. Details about the correct syntax
        # are provided in the documentation of this source configuration class.

        # Define the rate changes of an excitatory Poisson source.
        rate_changes = np.array([[0., 1000.],
                                 [2000., 100.]])

        poisson_weights = np.array([0.001, -0.001])

        sampler_config.source_config = \
            sbs.db.MultiPoissonVarRateSourceConfiguration(
                weight_per_source=poisson_weights,
                rate_changes_per_source=[rate_changes] * len(poisson_weights))

        # Choose the number of samplers in the network.
        bm = sbs.network.ThoroughBM(num_samplers=5, sim_name=sim_name,
                                    sampler_config=sampler_config)

        # Choose weights (here random) and symmetrize them.
        weights = np.random.randn(bm.num_samplers, bm.num_samplers)
        weights = (weights + weights.T) / 2.
        bm.weights_theo = weights

        # Choose biases (here random).
        bm.biases_theo = np.random.randn(bm.num_samplers)

        # Sample the network and save it.
        bm.gather_spikes(
                duration=1e5,  burn_in_time=500., dt=0.1,
                sim_setup_kwargs=sbs.utils.get_default_setup_kwargs(sim_name))

        log.info("Weights (theo):\n" + pf(bm.weights_theo))
        log.info("Biases (theo):\n" + pf(bm.biases_theo))

        log.info("Weights (bio):\n" + pf(bm.weights_bio))
        log.info("Biases (bio):\n" + pf(bm.biases_bio))

        log.info("Spikes: {}".format(pf(bm.ordered_spikes)))

        log.info("Spike-data: {}".format(pf(bm.spike_data)))

        bm.selected_sampler_idx = range(bm.num_samplers)

    def test_vmem_dist_var_poisson_rate_curr(self):
        """
            This tutorial shows how to record and plot the distribution of the
            free membrane potential.
        """
        np.random.seed(422441)

        # Load calibration data in order to create network.
        sampler_config = sbs.db.SamplerConfiguration.load(
            "test-calibration-curr.json")

        # Define the rate changes of an excitatory Poisson source.
        rate_changes = np.array([[0., 1000.],
                                 [2000., 100.]])

        poisson_weights = np.array([0.001, -0.001])

        sampler_config.source_config = \
            sbs.db.MultiPoissonVarRateSourceConfiguration(
                weight_per_source=poisson_weights,
                rate_changes_per_source=[rate_changes] * len(poisson_weights))

        sampler = sbs.samplers.LIFsampler(sampler_config, sim_name=sim_name)

        sampler.measure_free_vmem_dist(duration=1e4, dt=0.01,
                                       burn_in_time=500.)
        sampler.plot_free_vmem(prefix="test_vmem_dist_var_poisson_rate_curr-",
                               save=True)
        sampler.plot_free_vmem_autocorr(
            prefix="test_vmem_dist_var_poisson_rate_curr-", save=True)
