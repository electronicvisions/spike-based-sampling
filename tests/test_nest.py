#!/usr/bin/env python
# encoding: utf-8

"""
    Some tests based on tutorial functions…
"""

import unittest

import numpy as np

import sbs
sbs.gather_data.set_subprocess_silent(True)

log = sbs.log

# The backend of choice. Both should work but when using neuron, we need to
# disable saturating synapses for now.
sim_name = "pyNN.nest"
# sim_name = "pyNN.neuron"


class TestNest(unittest.TestCase):

    def setUp(self):

        import pyNN.nest as sim

        self.nest = sim.nest

        sim.setup(timestep=0.1, spike_precision="off_grid")

        self.sim = sim

    def tearDown(self):

        import pyNN.nest as sim
        sim.end()

    def test_change_poisson_rate(self):
        """
            Change the single(!) poisson rate of a boltzmann machine in the
            middle of an experiment!

            Please note that this functionality should be implemented in a rate
            changing soure (which is ongoing work).
        """
        np.random.seed(42)

        sampler_config = sbs.db.SamplerConfiguration.load(
                "test-calibration-cond.json")

        bm = sbs.network.ThoroughBM(
                num_samplers=5, sim_name=sim_name,
                sampler_config=sampler_config)

        # Set random symmetric weights.
        weights = np.random.randn(bm.num_samplers, bm.num_samplers)
        weights = (weights + weights.T) / 2.
        bm.weights_theo = weights

        # Set random biases.
        bm.biases_theo = np.random.randn(bm.num_samplers)

        bm.saturating_synapses_enabled = True
        bm.use_proper_tso = True

        bm.create(duration=10000.)

        self.sim.run(100.)

        sbs.utils.nest_change_poisson_rate(bm, 2000.)

        self.sim.run(100.)
