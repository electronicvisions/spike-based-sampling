#!/usr/bin/env python
# encoding: utf-8

"""
    The whole calibration done in async fashion.

"""

import zmq
import numpy as np

context = zmq.Context()


def calibrate_sampler(db_neuron_params, db_partial_calibration, db_sources,
        sim_name="pyNN.nest"):
    """
        db_neuron_params:
            NeuronParameters instance that contains the parameters to use.

        db_partial_calibration:
            Calibration instance holding configuration parameters
            (duration, num_samples).

        db_sources:
            List of sources to use for calibration.
    """
    pass


def _run_calibartion(self, address):
    """
        This function is meant to be run by the spawned calibration process.
    """
