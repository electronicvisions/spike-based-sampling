#!/usr/bin/env python2
# encoding: utf-8


import unittest
import numpy as np
from pprint import pformat as pf

import sbs
log = sbs.log

sim_name = "pyNN.nest"


def check_model():
    return sbs.utils.ensure_visionary_nest_model_available(
        "sinusoidal_poisson_generator")


# TODO: Test individual_spike_trains = False!

@unittest.skipUnless(check_model(),
                     "requires sinusoidal poisson generator models")
class TestSinusoidalPoissonGenerator(unittest.TestCase):
    def setUp(self):
        import pyNN.nest as sim

        self.nest = sim.nest

        sim.setup(timestep=0.1, spike_precision="on_grid")

        self.sim = sim

    def tearDown(self):
        import pyNN.nest as sim
        sim.end()

    def test_sample_network_sinusoidal_poisson_rate_cond(self):
        """
            How to setup and evaluate a Boltzmann machine. Please note that in
            order to instantiate BMs all needed neuron parameters need to be in
            the database and calibrated.

        """
        np.random.seed(424242)

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

        sampler_config.source_config = self.get_source_config()

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
        bm.gather_spikes(duration=1e5,  burn_in_time=500., dt=0.1,
                         sim_setup_kwargs={"spike_precision": "on_grid"})

        log.info("Weights (theo):\n" + pf(bm.weights_theo))
        log.info("Biases (theo):\n" + pf(bm.biases_theo))

        log.info("Weights (bio):\n" + pf(bm.weights_bio))
        log.info("Biases (bio):\n" + pf(bm.biases_bio))

        log.info("Spikes: {}".format(pf(bm.ordered_spikes)))

        log.info("Spike-data: {}".format(pf(bm.spike_data)))

        bm.selected_sampler_idx = range(bm.num_samplers)

    def test_sample_network_sinusoidal_poisson_rate_curr(self):
        """
            How to setup and evaluate a Boltzmann machine. Please note that in
            order to instantiate BMs all needed neuron parameters need to be in
            the database and calibrated.

        """
        np.random.seed(424242)

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

        sampler_config.source_config = self.get_source_config()

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
        bm.gather_spikes(duration=1e5,  burn_in_time=500., dt=0.1,
                         sim_setup_kwargs={"spike_precision": "on_grid"})

        log.info("Weights (theo):\n" + pf(bm.weights_theo))
        log.info("Biases (theo):\n" + pf(bm.biases_theo))

        log.info("Weights (bio):\n" + pf(bm.weights_bio))
        log.info("Biases (bio):\n" + pf(bm.biases_bio))

        log.info("Spikes: {}".format(pf(bm.ordered_spikes)))

        log.info("Spike-data: {}".format(pf(bm.spike_data)))

        bm.selected_sampler_idx = range(bm.num_samplers)

    def test_vmem_dist_sinusoidal_poisson_rate_curr(self):
        """
            This tutorial shows how to record and plot the distribution of the
            free membrane potential.
        """
        np.random.seed(424142)

        # Load calibration data in order to create network.
        sampler_config = sbs.db.SamplerConfiguration.load(
            "test-calibration-curr.json")

        sampler_config.source_config = self.get_source_config()

        sampler = sbs.samplers.LIFsampler(sampler_config, sim_name=sim_name)

        sampler.measure_free_vmem_dist(duration=1e4, dt=0.01,
                                       burn_in_time=500.)
        sampler.plot_free_vmem(
                prefix="test_vmem_dist_sinusoidal_poisson_rate_curr-",
                save=True)
        sampler.plot_free_vmem_autocorr(
            prefix="test_vmem_dist_sinusoidal_poisson_rate_curr-", save=True)

    def get_source_config(self):
        poisson_weights = np.array([0.001, -0.001])

        return sbs.db.SinusPoissonSourceConfiguration(
                    weights=poisson_weights,
                    rates=[10000., 10000.],
                    amplitudes=[2000., 2000.],
                    frequencies=[5., 5.],
                    phases=[100., 0.],
                    individual_spike_trains=True)
