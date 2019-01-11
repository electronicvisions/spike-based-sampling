#!/usr/bin/env python2
# encoding: utf-8

import numpy as np
import os.path as osp
import sys
import unittest

sys.path.insert(sys.path, 1, osp.join(osp.abspath(__file__), "..", "example"))
import simple_weight_translation as swt  # noqa: E402


class TestSimpleWeightTranslation(unittest.TestCase):
    sample_neuron_parameters = {
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

    def test_conversion(self):
        # quick calibration because we do not care about quality
        sampler_config = "sample-config"

        sampler = swt.calibration(self.sample_neuron_parameters,
                                  sampler_config,
                                  sim_duration_ms=1e3)

        num_samplers = 10

        # Set random symmetric weights.
        weights = np.random.randn(num_samplers, num_samplers)
        weights = (weights + weights.T) / 2.

        # Set random biases.
        biases = np.random.randn(num_samplers)

        weights_bio = swt.convert_weights_theo_to_bio(
                sampler_config, weights)
        biases_bio = swt.convert_biases_theo_to_bio(
                sampler_config, biases)

        weights_theo = swt.convert_weights_bio_to_theo(
                sampler, weights_bio)
        biases_theo = swt.convert_biases_bio_to_theo(
                sampler, biases_bio)

        # translation theo -> bio -> theo should be identity up to floating
        # point
        self.assertTrue(np.testing.assert_almost_equal(weights, weights_theo))
        self.assertTrue(np.testing.assert_almost_equal(biases, biases_theo))
