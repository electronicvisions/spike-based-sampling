#!/usr/bin/env python2
# encoding: utf-8
#
from __future__ import print_function

import unittest
import numpy as np

import sbs


class TestPairwiseCorrelations(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_basic(self):
        tau = 10.0
        duration = 1000.
        spikes = np.arange(0., duration, 20.)

        all_spikes = [
                spikes,
                spikes + 5.0,
                spikes + 10.0
            ]

        result = sbs.utils.get_pairwise_correlations(all_spikes, tau, duration)

        print(result)

        expected = np.array([[0.5,  0.25, 0.],
                             [0.25, 0.5,  0.25],
                             [0.,   0.25, 0.5]])

        self.assertTrue(np.allclose(result, expected))

    def test_overlapping(self):
        tau = [20., 30., 10.]
        duration = 1000.
        spikes = np.arange(0., duration, 20.)

        all_spikes = [
                spikes,
                np.r_[spikes + 5.0, 0.],  # one spike in the beggining
                                          # (test sorting)
                spikes + 10.0
            ]

        result = sbs.utils.get_pairwise_correlations(all_spikes, tau, duration)

        print(result)

        expected = np.array([[1.0, 1.0, 0.5],
                             [1.0, 1.0, 0.5],
                             [0.5, 0.5, 0.5]])

        self.assertTrue(np.allclose(result, expected))

    def test_ignore_until(self):
        tau = [20., 30., 10.]
        duration = 1000.
        spikes = np.arange(0., duration, 20.)

        all_spikes = [
                spikes,
                np.r_[spikes + 5.0],
                spikes + 10.0
            ]

        result = sbs.utils.get_pairwise_correlations(
                all_spikes, tau, duration, ignore_until=100.)

        print(result)

        expected = np.array([[1.0, 1.0, 0.5],
                             [1.0, 1.0, 0.5],
                             [0.5, 0.5, 0.5]])

        self.assertTrue(np.allclose(result, expected))
