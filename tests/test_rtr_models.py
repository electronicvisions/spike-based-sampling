#!/usr/bin/env python2
# encoding: utf-8


import unittest
import numpy as np

from pprint import pformat as pf

import sbs
log = sbs.log

sim_name = "pyNN.nest"


def check_rtr_model_cond():
    return sbs.utils.ensure_visionary_nest_model_available("iaf_cond_exp_rtr")


def check_rtr_model_curr():
    return sbs.utils.ensure_visionary_nest_model_available("iaf_psc_exp_rtr")


class TestRTRModels(unittest.TestCase):
    @unittest.skipUnless(check_rtr_model_cond(),
                         "requires RTR models from visionary NEST module")
    def test_calibration_cond(self):
        """
            A sample calibration procedure.

            The prefix 01 is so that this test gets executed before any
            sampling is done.
        """
        sampler_config = sbs.db.SamplerConfiguration.load(
                "test-calibration-cond.json")

        # expand regular neuon parameters to native nest parameters
        sampler_config.neuron_parameters =\
            sbs.db.ConductanceExponentialRandomRefractory.convert(
                sampler_config.neuron_parameters)

        # set a random distribution
        sampler_config.neuron_parameters.tau_refrac_dist = {
                "distribution": "uniform",
                "low": 20.,
                "high": 30.,
            }
        sampler_config.neuron_parameters.tau_refrac = 25.

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

        sampler = sbs.samplers.LIFsampler(sampler_config, sim_name=sim_name)

        # here we could give further kwargs for the pre-calibration phase when
        # the slope of the sigmoid is searched for
        sampler.calibrate(calibration)

        # Afterwards, we need to save the calibration.
        sampler.write_config("test-calibration-cond-rtr")

    @unittest.skipUnless(check_rtr_model_cond(),
                         "requires RTR models from visionary NEST module")
    def test_sample_rtr_cond(self):
        """
            How to setup and evaluate a Boltzmann machine. Please note that in
            order to instantiate BMs all needed neuron parameters need to be in
            the database and calibrated.

            Does the same thing as sbs.tools.sample_network(...).
        """
        np.random.seed(422342134)

        # Networks can be saved outside of the database.
        duration = 1e4

        # Try to load the network from file. This function returns None if no
        # network could be loaded.
        # No network loaded, we need to create it. We need to specify how
        # many samplers we want and what neuron parameters they should
        # have. Refer to the documentation for all the different ways this
        # is possible.

        sampler_config = sbs.db.SamplerConfiguration.load(
                "test-calibration-cond.json")

        # expand regular neuon parameters to native nest parameters
        sampler_config.neuron_parameters =\
            sbs.db.ConductanceExponentialRandomRefractory.convert(
                sampler_config.neuron_parameters)

        # set a random distribution
        sampler_config.neuron_parameters.tau_refrac_dist = {
                "distribution": "uniform",
                "low": 20.,
                "high": 30.,
            }

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

        # NOTE: Currently we have no saturating synapses enabled for nest
        # native models, working on it to fix it!
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

    @unittest.skipUnless(check_rtr_model_curr(),
                         "requires RTR models from visionary NEST module")
    def test_sample_rtr_curr(self):
        """
            How to setup and evaluate a Boltzmann machine. Please note that in
            order to instantiate BMs all needed neuron parameters need to be in
            the database and calibrated.

            Does the same thing as sbs.tools.sample_network(...).
        """
        np.random.seed(42141414)

        # Networks can be saved outside of the database.
        duration = 1e4

        # Try to load the network from file. This function returns None if no
        # network could be loaded.
        # No network loaded, we need to create it. We need to specify how
        # many samplers we want and what neuron parameters they should
        # have. Refer to the documentation for all the different ways this
        # is possible.

        sampler_config = sbs.db.SamplerConfiguration.load(
                "test-calibration-curr.json")

        # expand regular neuon parameters to native nest parameters
        sampler_config.neuron_parameters =\
            sbs.db.CurrentExponentialRandomRefractory.convert(
                sampler_config.neuron_parameters)

        # set a random distribution
        sampler_config.neuron_parameters.tau_refrac_dist = {
                "distribution": "uniform_int",
                "low": 200,
                "high": 300,
            }

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

        # NOTE: Currently we have no saturating synapses enabled for nest
        # native models, working on it to fix it!
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
