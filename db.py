#!/usr/bin/env python2
# encoding: utf-8

import json
import os.path as osp
import numpy as np

from .logcfg import log

def setup(*args, **kwargs):
    log.error("The database was discontinued and is without use. Please update "
            "your script accordingly!")

class Data(object):
    """
        Baseclass for all data objects
    """
    # dict mapping data attribute names to types
    data_attribute_types = {}

    @classmethod
    def load(cls, filepath=None):
        with open(filepath) as f:
            datadict = json.load(f)

        if datadict["_type"] != cls.__name__:
            log.warn("Using json data for type {} to create type {}".format(
                datadict["_type"], cls.__name__))
        del datadict["_type"]

        return cls(**datadict)

    def __init__(self, **attributes):
        self._from_dict(attributes)

        # set all those attributes that werent specified
        self._empty()

    def get_dict(self):
        return self._to_dict(with_type=False)

    def write(self, path):
        if osp.splitext(path)[1] != ".json":
            path += ".json"

        with open(path, "w") as f:
            json.dump(self._to_dict(), f,
                    ensure_ascii=False, indent=2)

    def copy(self):
        self.__class__(**self._to_dict())

    def _empty(self):
        """
            Set everything apart from class constants
        """
        for d in self.data_attribute_types:
            if not hasattr(self, d):
                setattr(self, d, None)

    def _to_dict(self, with_type=True):
        dikt = {d: self._convert_attr(d, with_type=with_type) for d in self.data_attribute_types}
        if with_type:
            dikt["_type"] = self.__class__.__name__
        return dikt

    def _convert_attr(self, name, with_type):
        d = getattr(self, name)

        if isinstance(d, Data):
            return d._to_dict(with_type=with_type)

        if isinstance(d, np.ndarray):
            return d.tolist()

        return d

    def _from_dict(self, dikt):
        for name, desired_type in self.data_attribute_types.iteritems():
            if hasattr(self, name):
                # class constants are skipped
                continue

            d = dikt.get(name, None)

            if d is not None and issubclass(desired_type, Data):
                if d["_type"] != desired_type.__class__.__name__:
                    new_desired_type = globals()[d["_type"]]
                    assert issubclass(new_desired_type, desired_type)
                    desired_type = new_desired_type
                del d["_type"]
                d = desired_type(**d)

            elif d is not None and issubclass(desired_type, np.ndarray):
                d = np.array(d)

            setattr(self, name, d)


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


class NeuronParametersConductance(NeuronParameters):
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


class NeuronParametersCurrent(NeuronParameters):
    pynn_model = "IF_cond_exp"

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


class SourceConfig(Data):
    pass


class PoissonSourceConfig(SourceConfig):
    """
        Positive weights: excitatory
        Negative weights: inhibitory
    """
    data_attribute_types = {
            "rates" : np.ndarray,
            "weights" : np.ndarray,
        }


class Fit(Data):
    data_attribute_types = {
        "alpha" : float,
        "v_p05" : float,
    }


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
        "source_config" : SourceConfig,
    }

    def get_samples_v_rest(self):
        return np.linspace(
                self.V_rest_min, self.V_rest_max, self.num_samples,
                endpoint=True)


class ParameterCalibration(Data):
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

        "V_rest_min" : float,
        "V_rest_max" : float,
        "dV" : float,

        "source_config" : SourceConfig,
    }

    def get_samples_v_rest(self):
        return np.linspace(
                self.V_rest_min, self.V_rest_max, self.num_samples,
                endpoint=True)

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


class InitialVmemSearch(Data):
    data_attribute_types = {
        "V_rest_min" : float,
        "V_rest_max" : float,
        "dV" : float,
        "pre_sim_time" : float,

        "lower_bound" : float,
        "upper_bound" : float,

        "max_search_steps" : int,

        "sim_name" : str,

        "duration" : float,
        "dt" : float,
        "burn_in_time" : float,

        "source_config" : SourceConfig,
    }

