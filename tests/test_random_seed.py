#!/usr/bin/env python

# encoding: utf-8

"""
    Some tests based on tutorial functions
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


# custom Exception for testing
class TestError(Exception):
    pass


# RunInSubprocess-wrapped method that is raising an exception that should be
# propagated to the host in the form of an IOError. Raising the original
# Exception is problematic if the host-process cannot easily import the module
# it is defined in - such as PyNest - hence every exception raised in the
# remote process is converted to a string and wrapped in RemoteError)
@sbs.comm.RunInSubprocess
def raise_test_error():
    raise TestError


# some example neuron parameters
neuron_params = {
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


class TestBasics(unittest.TestCase):

    def test_01_calibration(self):
        """
            A sample calibration procedure.

            The prefix 01 is so that this test gets executed before any
            sampling is done.
        """
        # Since we only have the neuron parameters for now, lets create those
        # first
        nparams = sbs.db.NeuronParametersConductanceExponential(
                **neuron_params)

        # Now we create a sampler object. We need to specify what simulator we
        # want along with the neuron model and parameters.
        # The sampler accepts both only the neuron parameters or a full sampler
        # configuration as argument.
        sampler = sbs.samplers.LIFsampler(nparams, sim_name=sim_name)

        # Now onto the actual calibration. For this we only need to specify our
        # source configuration and how long/with how many samples we want to
        # calibrate.

        source_config = sbs.db.PoissonSourceConfiguration(
                rates=3000.,
                weights=np.array([-1., 1]) * 0.001,
            )

        # We need to specify the remaining calibration parameters
        calibration = sbs.db.Calibration(
                duration=1e4, num_samples=150, burn_in_time=500., dt=0.01,
                source_config=source_config,
                sim_name=sim_name,
                sim_setup_kwargs={"spike_precision": "on_grid"})
        # Do not forget to specify the source configuration!

        # here we could give further kwargs for the pre-calibration phase when
        # the slope of the sigmoid is searched for
        sampler.calibrate(calibration)

        # Afterwards, we need to save the calibration.
        sampler.write_config("test-calibration-cond")

        # Finally, the calibration function can be plotted using the following
        # command ("calibration.png" in the current folder):
        sampler.plot_calibration(prefix="test_basics_cond-", save=True)

    def test_sample_network(self):
        """
            How to setup and evaluate a Boltzmann machine. Please note that in
            order to instantiate BMs all needed neuron parameters need to be in
            the database and calibrated.

            Does the same thing as sbs.tools.sample_network(...).
        """
        np.random.seed(4215123)
        duration = 1e4
        sampler_config = sbs.db.SamplerConfiguration.load(
                "test-calibration-cond.json")
        bm = sbs.network.ThoroughBM(
                num_samplers=5, sim_name=sim_name,
                sampler_config=sampler_config)
        weights = np.random.randn(bm.num_samplers, bm.num_samplers)
        weights = (weights + weights.T) / 2.
        bm.weights_theo = weights
        bm.biases_theo = np.random.randn(bm.num_samplers)
        bm.saturating_synapses_enabled = True
        bm.use_proper_tso = True

        if bm.sim_name == "pyNN.neuron":
            bm.saturating_synapses_enabled = False

        bm.gather_spikes(duration=duration, dt=0.1, burn_in_time=500.,
                         sim_setup_kwargs={"rng_seeds": [42424242]})
        spikes1 = np.array(bm.ordered_spikes)
        bm.gather_spikes(duration=duration, dt=0.1, burn_in_time=500.,
                         sim_setup_kwargs={"rng_seeds": [42424244]})
        spikes2 = bm.ordered_spikes

        self.assertTrue(spikes1 != spikes2)
