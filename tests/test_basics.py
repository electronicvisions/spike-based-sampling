#!/usr/bin/env python
# encoding: utf-8

"""
    Some tests based on tutorial functions…
"""

import unittest

import numpy as np
from pprint import pformat as pf
import os

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
                rates=np.array([3000.] * 2),
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

    def test_02_calibration_curr(self):
        """
            A sample calibration procedure.
        """
        nparams = sbs.db.NeuronParametersCurrentExponential(
            **{k: v for k, v in neuron_params.iteritems()
               if not k.startswith("e_rev_")})

        sampler = sbs.samplers.LIFsampler(nparams, sim_name=sim_name)

        source_config = sbs.db.PoissonSourceConfiguration(
                rates=np.array([1000.] * 2),
                weights=np.array([-1., 1]) * 0.001,
            )

        calibration = sbs.db.Calibration(
                duration=1e4, num_samples=150, burn_in_time=500., dt=0.01,
                source_config=source_config,
                sim_name=sim_name,
                sim_setup_kwargs={"spike_precision": "on_grid"})

        sampler.calibrate(calibration)

        sampler.write_config("test-calibration-curr")

        sampler.plot_calibration(prefix="test_basics_curr-", save=True)

    def test_vmem_dist(self):
        """
            This tutorial shows how to record and plot the distribution of the
            free membrane potential.
        """
        sampler_config = sbs.db.SamplerConfiguration.load(
                "test-calibration-cond.json")

        sampler = sbs.samplers.LIFsampler(sampler_config, sim_name=sim_name)

        sampler.measure_free_vmem_dist(duration=1e4, dt=0.01,
                                       burn_in_time=500.)
        sampler.plot_free_vmem(prefix="test_basics_cond-", save=True)
        sampler.plot_free_vmem_autocorr(
                prefix="test_basics_cond-", save=True)

    def test_sample_network(self):
        """
            How to setup and evaluate a Boltzmann machine. Please note that in
            order to instantiate BMs all needed neuron parameters need to be in
            the database and calibrated.

            Does the same thing as sbs.tools.sample_network(...).
        """
        np.random.seed(42)

        # Networks can be saved outside of the database.
        #  filename = "tutorial-network.pkl.gz"
        duration = 1e4

        # Try to load the network from file. This function returns None if no
        # network could be loaded.
        #  bm = sbs.network.ThoroughBM.load(filename)

        # No network loaded, we need to create it. We need to specify how
        # many samplers we want and what neuron parameters they should
        # have. Refer to the documentation for all the different ways this
        # is possible.

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

        # NOTE: By setting the theoretical weights and biases, the
        # biological ones automatically get calculated on-demand by
        # accessing bm.weights_bio and bm.biases_bio

        bm.saturating_synapses_enabled = True
        bm.use_proper_tso = True

        if bm.sim_name == "pyNN.neuron":
            bm.saturating_synapses_enabled = False

        bm.gather_spikes(duration=duration, dt=0.1, burn_in_time=500.)

        # Now we just print out some information and plot the distributions.
        log.info("Weights (theo):\n" + pf(bm.weights_theo))
        log.info("Biases (theo):\n" + pf(bm.biases_theo))

        log.info("Weights (bio):\n" + pf(bm.weights_bio))
        log.info("Biases (bio):\n" + pf(bm.biases_bio))

        log.info("Spikes: {}".format(pf(bm.ordered_spikes)))

        log.info("Spike-data: {}".format(pf(bm.spike_data)))

        bm.selected_sampler_idx = range(bm.num_samplers)

        log.info("Marginal prob (sim):\n" + pf(bm.dist_marginal_sim))

        log.info("Joint prob (sim):\n" +
                 pf(list(np.ndenumerate(bm.dist_joint_sim))))

        log.info("Marginal prob (theo):\n" + pf(bm.dist_marginal_theo))

        log.info("Joint prob (theo):\n" +
                 pf(list(np.ndenumerate(bm.dist_joint_theo))))

        dkl_marginal = sbs.utils.dkl_sum_marginals(
            bm.dist_marginal_theo, bm.dist_marginal_sim)
        log.info("DKL marginal: {}".format(dkl_marginal))
        self.assertTrue(dkl_marginal > 0)
        self.assertTrue(dkl_marginal < 1)

        dkl_joint = sbs.utils.dkl(
                bm.dist_joint_theo.flatten(), bm.dist_joint_sim.flatten())
        log.info("DKL joint: {}".format(dkl_joint))
        self.assertTrue(dkl_joint > 0)
        self.assertTrue(dkl_joint < 1)

        bm.plot_dist_marginal(prefix="test_basics_cond-", save=True)
        bm.plot_dist_joint(prefix="test_basics_cond-", save=True)

    def test_sample_network_curr(self):
        """
            How to setup and evaluate a Boltzmann machine. Please note that in
            order to instantiate BMs all needed neuron parameters need to be in
            the database and calibrated.

            Does the same thing as sbs.tools.sample_network(...).
        """
        np.random.seed(42)

        # Networks can be saved outside of the database.
        filename = "tutorial-network.pkl.gz"
        duration = 1e6

        # Try to load the network from file. This function returns None if no
        # network could be loaded.
        bm = sbs.network.ThoroughBM.load(filename)

        if bm is None:
            sampler_config = sbs.db.SamplerConfiguration.load(
                    "test-calibration-curr.json")

            bm = sbs.network.ThoroughBM(
                    num_samplers=5, sim_name=sim_name,
                    sampler_config=sampler_config)

            # Set random symmetric weights.
            weights = np.random.randn(bm.num_samplers, bm.num_samplers)
            weights = (weights + weights.T) / 2.
            bm.weights_theo = weights

            # Set random biases.
            bm.biases_theo = np.random.randn(bm.num_samplers)

            if bm.sim_name == "pyNN.neuron":
                bm.saturating_synapses_enabled = False

            bm.gather_spikes(duration=duration, dt=0.1, burn_in_time=500.)
            # bm.save(filename)

        # Now we just print out some information and plot the distributions.

        log.info("Weights (theo):\n" + pf(bm.weights_theo))
        log.info("Biases (theo):\n" + pf(bm.biases_theo))

        log.info("Weights (bio):\n" + pf(bm.weights_bio))
        log.info("Biases (bio):\n" + pf(bm.biases_bio))

        log.info("Spikes: {}".format(pf(bm.ordered_spikes)))

        log.info("Spike-data: {}".format(pf(bm.spike_data)))

        bm.selected_sampler_idx = range(bm.num_samplers)

        log.info("Marginal prob (sim):\n" + pf(bm.dist_marginal_sim))

        log.info("Joint prob (sim):\n" +
                 pf(list(np.ndenumerate(bm.dist_joint_sim))))

        log.info("Marginal prob (theo):\n" + pf(bm.dist_marginal_theo))

        log.info("Joint prob (theo):\n" +
                 pf(list(np.ndenumerate(bm.dist_joint_theo))))

        dkl_marginal = sbs.utils.dkl_sum_marginals(
            bm.dist_marginal_theo, bm.dist_marginal_sim)
        log.info("DKL marginal: {}".format(dkl_marginal))
        self.assertTrue(dkl_marginal > 0)
        self.assertTrue(dkl_marginal < 1)

        dkl_joint = sbs.utils.dkl(
                bm.dist_joint_theo.flatten(), bm.dist_joint_sim.flatten())
        log.info("DKL joint: {}".format(dkl_joint))
        self.assertTrue(dkl_joint > 0)
        self.assertTrue(dkl_joint < 1)

        bm.plot_dist_marginal(prefix="test_basics_curr-", save=True)
        bm.plot_dist_joint(prefix="test_basics_curr-", save=True)

    def test_sample_network_fixed_spikes(self):
        """
            Demonstrating how to setup and use FixedSpikeTrain sources
            with already pre-calibrated neurons.

            Please note that the rates of the FixedSpikeTrains should roughly
            correspond to the rates of the random sources used to calibrate the
            network!

            Also note that because these are just some fixed spike trains the
            DKL etc will be horrible in this example (but that is not the point
            here).
        """
        np.random.seed(42)

        duration = 1e4

        sampler_config = sbs.db.SamplerConfiguration.load(
                "test-calibration-cond.json")

        isi = 1000./sampler_config.calibration.source_config.rates[0]
        offset = 1.

        spike_times = np.arange(1., duration, isi)
        num_spikes = spike_times.size

        sampler_config.source_config = sbs.db.FixedSpikeTrainConfiguration(
                spike_times=np.r_[spike_times, spike_times+offset],
                spike_ids=np.array([0] * num_spikes + [1] * num_spikes),
                weights=sampler_config.calibration.source_config.weights
            )

        bm = sbs.network.ThoroughBM(
                num_samplers=5, sim_name=sim_name,
                sampler_config=sampler_config)

        # Set random symmetric weights.
        weights = np.random.randn(bm.num_samplers, bm.num_samplers)
        weights = (weights + weights.T) / 2.
        bm.weights_theo = weights

        # Set random biases.
        bm.biases_theo = np.random.randn(bm.num_samplers)

        # NOTE: By setting the theoretical weights and biases, the biological
        # ones automatically get calculated on-demand by accessing
        # bm.weights_bio and bm.biases_bio

        bm.saturating_synapses_enabled = True
        bm.use_proper_tso = True

        if bm.sim_name == "pyNN.neuron":
            bm.saturating_synapses_enabled = False

        bm.gather_spikes(duration=duration, dt=0.1, burn_in_time=500.)

        # Now we just print out some information and plot the distributions.

        log.info("Weights (theo):\n" + pf(bm.weights_theo))
        log.info("Biases (theo):\n" + pf(bm.biases_theo))

        log.info("Weights (bio):\n" + pf(bm.weights_bio))
        log.info("Biases (bio):\n" + pf(bm.biases_bio))

        log.info("Spikes: {}".format(pf(bm.ordered_spikes)))

        log.info("Spike-data: {}".format(pf(bm.spike_data)))

        bm.selected_sampler_idx = range(bm.num_samplers)

        log.info("Marginal prob (sim):\n" + pf(bm.dist_marginal_sim))

        log.info("Joint prob (sim):\n" +
                 pf(list(np.ndenumerate(bm.dist_joint_sim))))

        log.info("Marginal prob (theo):\n" + pf(bm.dist_marginal_theo))

        log.info("Joint prob (theo):\n" +
                 pf(list(np.ndenumerate(bm.dist_joint_theo))))

        dkl_marginal = sbs.utils.dkl_sum_marginals(
            bm.dist_marginal_theo, bm.dist_marginal_sim)
        log.info("DKL marginal: {}".format(dkl_marginal))
        self.assertTrue(dkl_marginal > 0)
        #  self.assertTrue(dkl_marginal < 1) # not assured here

        dkl_joint = sbs.utils.dkl(
                bm.dist_joint_theo.flatten(), bm.dist_joint_sim.flatten())
        log.info("DKL joint: {}".format(dkl_joint))
        self.assertTrue(dkl_joint > 0)
        #  self.assertTrue(dkl_joint < 1) # not assured here

        bm.plot_dist_marginal(prefix="test_basics_fixed_spike-", save=True)
        bm.plot_dist_joint(prefix="test_basics_fixed_spike-", save=True)

    def test_runinsubprocess_clientfail(self):
        # Simulate an exception raised in a method wrapped with
        # RunInSuprocess-decorator.
        #
        # If DEBUG is set in environment, the RunInSubprocess-decorator is
        # rendered mute and the original exception will be raised, therefore we
        # need to check for both RemoteError and TestError (as the tests are
        # ran with DEBUG set and unset in the corresponding jenkins-job).
        try:
            raise_test_error()
        except TestError:
            # if we are in debug mode the original exception should be raised
            self.assertTrue("DEBUG" in os.environ or
                            "SBS_NO_SUBPROCESS" in os.environ)
            return
        except sbs.comm.RemoteError as e:
            self.assertTrue(e.original_error_name == "TestError")
            return
        # if we reach this point, there was no exception -> fail the test
        self.fail("No exception raised.")
