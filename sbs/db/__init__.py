#!/usr/bin/env python2
# encoding: utf-8

import numpy as np

from .core import Data
from .sources import *  # noqa: F401,F403
from . import sources
from ..logcfg import log
from .neuron_parameters import *  # noqa: F401,F403
from . import neuron_parameters as nparams


def setup(*args, **kwargs):
    log.error("The database was discontinued and is without use. "
              "Please update your script accordingly!")


class Fit(Data):
    data_attribute_types = {
        "alpha": float,
        "v_p05": float,
    }

    def is_valid(self):
        return self.alpha is not None and self.v_p05 is not None


class Calibration(Data):
    data_attribute_types = {
        "sim_name": str,
        "sim_setup_kwargs": dict,

        "duration": float,
        "dt": float,
        "burn_in_time": float,

        "V_rest_min": float,
        "V_rest_max": float,
        "num_samples": float,

        "samples_p_on": np.ndarray,

        "fit": Fit,
        "source_config": sources.SourceConfiguration,  #
    }

    def get_samples_v_rest(self):
        return np.linspace(
                self.V_rest_min, self.V_rest_max, self.num_samples,
                endpoint=True)


class TsoParameters(Data):
    """
        Parameters for TSO-enabled weights

        Contains the regular TSO parameters ("u" and "x" are specific to the
        NEST-implementation for TSO).

        The weight_rescale is something that we need to do when improving
        mixing capabilities of networks, for instance.

        References:

        1st:
        ----
        Tsodyks, M. V., & Markram, H. (1997). The neural code between
        neocortical pyramidal neurons depends on neurotransmitter release
        probability. PNAS, 94(2), 719-23.

        The first tsodyks paper about TSO, but with no facilitation time
        constants in the model yet. With a tau_inact, it defines a more
        complicated model of short-term depression. Notice that there is a
        correction version in the equation (1-3)

        2nd:
        ----
        Markram, Henry, Yun Wang, and Misha Tsodyks. "Differential signaling
        via the same axon of neocortical pyramidal neurons." Proceedings of the
        National Academy of Sciences 95.9 (1998): 5323-5328.

        A disrete version of the TSO mechanism, added facilitation time
        constant. Here you find the definition of u, which is the current
        scaling factor of the weight, which will be 0 if tau_facil is 0 (no
        facilitation)

        3rd:
        ----
        Fuhrmann, Galit, et al. "Coding of temporal information by
        activity-dependent synapses." Journal of neurophysiology 87.1 (2002):
        140-148.

        Provides a continuous description of the equation in the 2nd reference.
    """

    data_attribute_types = {
        "U": float,
        "u": float,
        "x": float,

        "tau_rec": float,  # if not set, will be set to tau_syn
        "tau_fac": float,  # if not set, will be disabled

        "weight_rescale": float,  # weights need to be set *= weight_rescale
    }

    data_attribute_defaults = {
        "U": 1.,
        "u": 1.,
        "x": 0.,

        "weight_rescale": 1.,
    }


class SamplerConfiguration(Data):
    """
        The source configuration is in order to be able to specify a different
        source configuration from the one used during calibration.
    """

    data_attribute_types = {
        "neuron_parameters": nparams.NeuronParameters,
        "calibration": Calibration,
        "source_config": sources.SourceConfiguration,
        "tso_parameters": TsoParameters,
    }


class PreCalibration(Data):
    """
        Used by the calibration routine to find the suitable slope.
    """
    data_attribute_types = {
        "sim_name": str,
        "sim_setup_kwargs": dict,

        "duration": float,
        "dt": float,
        "burn_in_time": float,

        "max_search_steps": int,
        "min_num_points": int,

        "lower_bound": float,
        "upper_bound": float,

        "V_rest_min": float,
        "V_rest_max": float,
        "dV": float,

        "source_config": sources.SourceConfiguration,
    }

    def get_samples_v_rest(self):
        return np.arange(
                self.V_rest_min, self.V_rest_max+self.dV, self.dV)


class VmemDistribution(Data):
    """
        Theoretical membrane distribution.
    """

    data_attribute_types = {
        "mean":     float,
        "std":      float,
        "g_tot":    float,
        "tau_eff":  float,
    }
