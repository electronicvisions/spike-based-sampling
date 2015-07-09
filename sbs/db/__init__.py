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


def setup(*args, **kwargs):
    log.error("The database was discontinued and is without use. Please update "
            "your script accordingly!")


class NeuronParameters(Data):
    data_attribute_types = {
        "pynn_model" : str,
    }

    @property
    def g_l(self):
        "Leak conductance in µS"
        return self.cm / self.tau_m

    def get_pynn_parameters(self):
        dikt = self.get_dict()
        del dikt["pynn_model"]
        return dikt


class NeuronParametersConductanceExponential(NeuronParameters):
    pynn_model = "IF_cond_exp"

    data_attribute_types = dict(NeuronParameters.data_attribute_types.items()+{
        "cm"         : float, # nF  Capacity of the membrane
        "tau_m"      : float, # ms  Membrane time constant
        "tau_refrac" : float, # ms  Duration of refractory period
        "tau_syn_E"  : float, # ms  Decay time of excitatory synaptic curr
        "tau_syn_I"  : float, # ms  Decay time of inhibitory synaptic curr
        "e_rev_E"    : float, # mV  Reversal potential for exc inpt
        "e_rev_I"    : float, # mV  Reversal potential for inh inpt
        "i_offset"   : float, # nA  Offset current
        "v_rest"     : float, # mV  Rest potential
        "v_reset"    : float, # mV  Reset potential after a spike
        "v_thresh"   : float, # mV  Spike threshold
    }.items())


class NeuronParametersConductanceAlpha(NeuronParameters):
    pynn_model = "IF_cond_alpha"

    data_attribute_types = dict(NeuronParameters.data_attribute_types.items()+{
        "cm"         : float, # nF  Capacity of the membrane
        "tau_m"      : float, # ms  Membrane time constant
        "tau_refrac" : float, # ms  Duration of refractory period
        "tau_syn_E"  : float, # ms  Decay time of excitatory synaptic curr
        "tau_syn_I"  : float, # ms  Decay time of inhibitory synaptic curr
        "e_rev_E"    : float, # mV  Reversal potential for exc inpt
        "e_rev_I"    : float, # mV  Reversal potential for inh inpt
        "i_offset"   : float, # nA  Offset current
        "v_rest"     : float, # mV  Rest potential
        "v_reset"    : float, # mV  Reset potential after a spike
        "v_thresh"   : float, # mV  Spike threshold
    }.items())


class NeuronParametersCurrentExponential(NeuronParameters):
    pynn_model = "IF_curr_exp"

    data_attribute_types = dict(NeuronParameters.data_attribute_types.items()+{
        "cm"         : float, # nF  Capacity of the membrane
        "tau_m"      : float, # ms  Membrane time constant
        "tau_refrac" : float, # ms  Duration of refractory period
        "tau_syn_E"  : float, # ms  Decay time of excitatory synaptic curr
        "tau_syn_I"  : float, # ms  Decay time of inhibitory synaptic curr
        "i_offset"   : float, # nA  Offset current
        "v_rest"     : float, # mV  Rest potential
        "v_reset"    : float, # mV  Reset potential after a spike
        "v_thresh"   : float, # mV  Spike threshold
    }.items())


class NeuronParametersCurrentAlpha(NeuronParameters):
    pynn_model = "IF_curr_alpha"

    data_attribute_types = dict(NeuronParameters.data_attribute_types.items()+{
        "cm"         : float, # nF  Capacity of the membrane
        "tau_m"      : float, # ms  Membrane time constant
        "tau_refrac" : float, # ms  Duration of refractory period
        "tau_syn_E"  : float, # ms  Decay time of excitatory synaptic curr
        "tau_syn_I"  : float, # ms  Decay time of inhibitory synaptic curr
        "i_offset"   : float, # nA  Offset current
        "v_rest"     : float, # mV  Rest potential
        "v_reset"    : float, # mV  Reset potential after a spike
        "v_thresh"   : float, # mV  Spike threshold
    }.items())


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
    data_attribute_types = {
        "neuron_parameters" : NeuronParameters,
        "calibration" : Calibration,
    }


class PreCalibration(Data):
    """
        Used by the calibration routine to find the suitable slope.
    """
    data_attribute_types = {
        "sim_name" : str,

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


