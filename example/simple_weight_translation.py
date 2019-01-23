#!/usr/bin/env python2
# encoding: utf-8

"""
    A small minimal example to demonstrate how to achieve the translation of
    weights/biases for varying noise-/neuron-parameters.
"""

from __future__ import print_function

import numpy as np

import sbs
sbs.gather_data.set_subprocess_silent(False)
log = sbs.log

SIMULATOR = "pyNN.nest"

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


if __name__ == "__main__":
    filename_calibration = "default-sampler-config"

    # perform calibration (needed once for each set neuron/noise parameters)
    sbs.simple.calibration(sample_neuron_parameters, filename_calibration,
                           sim_duration_ms=1e3,
                           plot_calibration=True)

    num_samplers = 10

    # Set random symmetric weights.
    weights = np.random.randn(num_samplers, num_samplers)
    weights = (weights + weights.T) / 2.

    # Set random biases.
    biases = np.random.randn(num_samplers)

    weights_bio = sbs.simple.convert_weights_theo_to_bio(
            filename_calibration, weights)
    biases_bio = sbs.simple.convert_biases_theo_to_bio(
            filename_calibration, biases)

    print(weights_bio)
    print(biases_bio)
