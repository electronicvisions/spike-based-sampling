#!/usr/bin/env python2
# encoding: utf-8

import unittest
import numpy as np

import sbs


class TestSimpleWeightTranslation(unittest.TestCase):

    def setUp(self):
        self.sample_neuron_parameters = {
                "cm": .2,
                "tau_m": 1.,
                "e_rev_E": 0.,
                "e_rev_I": -100.,
                "v_thresh": -50.,
                "tau_syn_E": 10.,
                "v_rest": -50.,
                "tau_syn_I": 10.,
                "v_reset": -50.001,
                "tau_refrac": 10.,
                "i_offset": 0.,
            }

    def tearDown(self):
        pass

    def test_conversion(self):
        # quick calibration because we do not care about quality
        sampler_config = "sample-config"

        sampler = sbs.simple.calibration(self.sample_neuron_parameters,
                                         sampler_config,
                                         sim_duration_ms=1e3)

        num_samplers = 10

        # Set random symmetric weights.
        weights = np.random.randn(num_samplers, num_samplers)
        weights = (weights + weights.T) / 2.

        # Set random biases.
        biases = np.random.randn(num_samplers)

        weights_bio = sbs.simple.convert_weights_theo_to_bio(
                sampler_config, weights)
        biases_bio = sbs.simple.convert_biases_theo_to_bio(
                sampler_config, biases)

        weights_theo = sbs.simple.convert_weights_bio_to_theo(
                sampler, weights_bio)
        biases_theo = sbs.simple.convert_biases_bio_to_theo(
                sampler, biases_bio)

        # translation theo -> bio -> theo should be identity up to floating
        # point
        self.assertTrue(np.allclose(weights, weights_theo))
        self.assertTrue(np.allclose(biases, biases_theo))

    def test_array_like_calibration(self):
        sampler_config = "sample-config-array-like"

        sbs.simple.calibration(
                self.sample_neuron_parameters,
                sampler_config,
                noise_rate_exc=np.array([300., 400.]),
                noise_rate_inh=[250., 800.],
                noise_weight_exc=[0.001, 0.0005],
                noise_weight_inh=np.array([-0.001, -0.0005]),
                sim_duration_ms=1e3)
