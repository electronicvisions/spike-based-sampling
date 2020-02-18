#!/usr/bin/env python2
# encoding: utf-8

from __future__ import print_function

import unittest
import numpy as np
from pprint import pformat as pf

import sbs
log = sbs.log


class TestGroupIdenticalParameters(unittest.TestCase):

    def compare_group_identical_parameters(self, to_group, expected):
        generated = list(sbs.utils.group_identical_parameters(to_group))

        print("Generated:")
        print(generated)
        print("Expected:")
        print(expected)

        def sort_by_min_index(to_sort):
            return sorted(to_sort, key=lambda tup: tup[0].min())

        for ((gen_idx, gen_params), (exp_idx, exp_params)) in zip(
                sort_by_min_index(generated),
                sort_by_min_index(expected)):
            self.assertTrue(
                np.all(gen_idx == exp_idx),
                "Indices missmatch: {} is not {}".format(pf(gen_idx),
                                                         pf(exp_idx)))
            self.assertDictEqual(
                gen_params, exp_params,
                "Parameter missmatch: {} is not {}".format(pf(gen_params),
                                                           pf(exp_params)))

    def test_from_docstring(self):
        self.compare_group_identical_parameters(
            {
                "foo": [1, 1, 2],
                "bar": [2, 2, 4],
                "deadbeef": 0
            },
            [
                (np.array([0, 1]), {"foo": 1, "bar": 2, "deadbeef": 0}),
                (np.array([2]),    {"foo": 2, "bar": 4, "deadbeef": 0})
            ])

    def test_simple_nest(self):
        self.compare_group_identical_parameters(
            dict(
                rate=np.array([10000., 10000.]),
                amplitude=np.array([2000., 2000.]),
                frequency=np.array([5., 5.]),
                phase=np.array([0., 0.]),
            ),
            [
                (np.array([0, 1]), {"rate": 10000.,
                                    "amplitude": 2000.,
                                    "frequency": 5.,
                                    "phase": 0.}),
            ])

    def test_simple_02(self):
        self.compare_group_identical_parameters(
            dict(
                rate=np.array([10000., 10000.]),
                amplitude=np.array([2000., 2000.]),
                frequency=np.array([5., 5.]),
                phase=np.array([200., 0.]),
            ),
            [
                (np.array([0]), {"rate": 10000.,
                                 "amplitude": 2000.,
                                 "frequency": 5.,
                                 "phase": 200.}),
                (np.array([1]), {"rate": 10000.,
                                 "amplitude": 2000.,
                                 "frequency": 5.,
                                 "phase": 0.})
            ])
