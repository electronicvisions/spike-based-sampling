#!/usr/bin/env python2
# encoding: utf-8

import unittest
import numpy as np

from pprint import pformat as pf

import sbs
log = sbs.log

sim_name = "pyNN.nest"

neuron_params = {
        "cm": .2,
        "tau_m": 1.,
        "v_thresh": -50.,
        "tau_syn_E": 10.,
        "v_rest": -50.,
        "tau_syn_I": 10.,
        "v_reset": -50.001,
        "tau_refrac": 10.,
        "i_offset": 0.,
    }

nn_net_params = {
        "N": 384,  # size of noise network (variable)
        "gamma": 0.8,  # relative size of E/I subpopulations
        "epsilon": 0.1,  # connectivity
        "epsilon_external": 0.1,  # connectivity
        "neuron_parameters": sbs.db.NeuronParametersCurrentExponential(**{
            "cm": 1.0,  # (nF)
            "i_offset": 0.0,  # (nA)
            "tau_m": 20.0,  # (ms)
            "tau_refrac": 0.1,  # (ms)
            "tau_syn_E": 5.0,  # (ms)
            "tau_syn_I": 5.0,  # (ms)
            "v_reset": -60.0,  # (mV)
            "v_rest": -40.,  # (mV)
            "v_thresh": -50.0  # (mV)
        }),
        "JE": 0.0635,  # (nA) from matching to 0.2mV peak PSP
        "g": 10.,
        "delay_internal": 1.,  # (ms)
        "delay_external": 1.,  # (ms)
        "rate": 30.  # Hz
    }


class TestNN(unittest.TestCase):

    def test_01_calibration(self):
        """
            A sample calibration procedure.

            The prefix 01 is so that this test gets executed before any
            sampling is done.
        """
        # Since we only have the neuron parameters for now, lets create those
        # first
        # TODO: Have tests with conductace-based neurons
        nparams = sbs.db.NeuronParametersCurrentExponential(**neuron_params)

        # Now we create a sampler object. We need to specify what simulator we
        # want along with the neuron model and parameters.
        # The sampler accepts both only the neuron parameters or a full sampler
        # configuration as argument.
        sampler = sbs.samplers.LIFsampler(nparams, sim_name=sim_name)

        # Now onto the actual calibration. For this we only need to specify our
        # source configuration and how long/with how many samples we want to
        # calibrate.

        source_config = sbs.db.NoiseNetworkSourceConfiguration(**nn_net_params)

        # We need to specify the remaining calibration parameters
        calibration = sbs.db.Calibration(
                duration=1e4, num_samples=150, burn_in_time=500., dt=0.01,
                source_config=source_config,
                sim_name=sim_name,
                sim_setup_kwargs=sbs.utils.get_default_setup_kwargs(sim_name))
        # Do not forget to specify the source configuration!

        # here we could give further kwargs for the pre-calibration phase when
        # the slope of the sigmoid is searched for
        sampler.calibrate(calibration)

        # Afterwards, we need to save the calibration.
        sampler.write_config("test-calibration-nn-curr")

        # Finally, the calibration function can be plotted using the following
        # command ("calibration.png" in the current folder):
        sampler.plot_calibration(prefix="test_noisenet_curr-", save=True)

    def test_sample_network_curr(self):
        """
            How to setup and evaluate a Boltzmann machine. Please note that in
            order to instantiate BMs all needed neuron parameters need to be in
            the database and calibrated.

            Does the same thing as sbs.tools.sample_network(...).
        """
        np.random.seed(4242351)

        # Networks can be saved outside of the database.
        duration = 1e4

        sampler_config = sbs.db.SamplerConfiguration.load(
                "test-calibration-nn-curr.json")

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

        log.info("DKL marginal: {}".format(sbs.utils.dkl_sum_marginals(
            bm.dist_marginal_theo, bm.dist_marginal_sim)))

        log.info("DKL joint: {}".format(sbs.utils.dkl(
                bm.dist_joint_theo.flatten(), bm.dist_joint_sim.flatten())))

        bm.plot_dist_marginal(prefix="test_noice_curr-", save=True)
        bm.plot_dist_joint(prefix="test_noice_curr-", save=True)
