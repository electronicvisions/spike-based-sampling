#!/usr/bin/env python2
# encoding: utf-8

import json
import os.path as osp
import numpy as np
import copy
import collections as c


from .core import Data
from .sources import *
from ..logcfg import log
from .neuron_parameters import *


def setup(*args, **kwargs):
    log.error("The database was discontinued and is without use. Please update "
            "your script accordingly!")


class Fit(Data):
    data_attribute_types = {
        "alpha" : float,
        "v_p05" : float,
    }

    def is_valid(self):
        return self.alpha is not None and self.v_p05 is not None


class Calibration(Data):
    data_attribute_types = {
        "sim_name" : str,
        "sim_setup_kwargs" : dict,

        "duration" : float,
        "dt" : float,
        "burn_in_time" : float,

        "V_rest_min" : float,
        "V_rest_max" : float,
        "num_samples" : float,

        "samples_p_on" : np.ndarray,

        "fit" : Fit,
        "source_config" : SourceConfiguration,
    }

    def get_samples_v_rest(self):
        return np.linspace(
                self.V_rest_min, self.V_rest_max, self.num_samples,
                endpoint=True)


class SamplerConfiguration(Data):
    """
        The source configuration is in order to be able to specify a different
        source configuration from the one used during calibration.
    """

    data_attribute_types = {
        "neuron_parameters" : NeuronParameters,
        "calibration" : Calibration,
        "source_config" : SourceConfiguration,
    }


class PreCalibration(Data):
    """
        Used by the calibration routine to find the suitable slope.
    """
    data_attribute_types = {
        "sim_name" : str,
        "sim_setup_kwargs" : dict,

        "duration" : float,
        "dt" : float,
        "burn_in_time" : float,

        "max_search_steps" : int,

        "lower_bound": float,
        "upper_bound": float,

        "V_rest_min" : float,
        "V_rest_max" : float,
        "dV" : float,

        "source_config" : SourceConfiguration,
    }

    def get_samples_v_rest(self):
        return np.arange(
                self.V_rest_min, self.V_rest_max+self.dV, self.dV)


class VmemDistribution(Data):
    """
        Theoretical membrane distribution.
    """

    data_attribute_types = {
        "mean"    : float,
        "std"     : float,
        "g_tot"   : float,
        "tau_eff" : float,
    }


