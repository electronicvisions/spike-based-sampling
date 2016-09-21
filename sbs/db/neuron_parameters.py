#!/usr/bin/env python2
# encoding: utf-8

from .core import Data
from .. import utils

import copy
import six
import itertools as it

class NeuronParameters(Data):
    data_attribute_types = {
        "pynn_model" : str,
    }

    @property
    def g_l(self):
        "Leak conductance in µS"
        return self.cm / self.tau_m

    def get_pynn_parameters(self, adjusted_parameters=None):
        """
            Get PyNN parameters with the possibility of overriding some (e.g.
            v_rest).
        """
        dikt = self.get_dict()
        if adjusted_parameters is not None:
            dikt.update(adjusted_parameters)
        del dikt["pynn_model"]
        return dikt

    def get_vmem_distribution_theo(self,
            source_parameters, adjusted_parameters=None):
        """
            Has to return a tuple of the following:

            * effective mean membrane potential (v_eff)
            * standard deviation of membrane potential
            * total conductance (g_tot)
            * effective membrane time constant (tau_eff)
        """
        kwargs = self.get_dict()
        kwargs["g_l"] = self.g_l
        kwargs.update(source_parameters)

        if adjusted_parameters is not None:
            kwargs.update(adjusted_parameters)

        return getattr(utils, "{}_distribution".format(self.pynn_model))(
                **kwargs)

    def get_pynn_model_object(self, sim):
        return getattr(sim, self.pynn_model)


class NeuronParametersConductanceExponential(NeuronParameters):
    pynn_model = "IF_cond_exp"

    data_attribute_types = {
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
    }


class NeuronParametersConductanceAlpha(NeuronParameters):
    pynn_model = "IF_cond_alpha"

    data_attribute_types = {
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
    }


class NeuronParametersCurrentExponential(NeuronParameters):
    pynn_model = "IF_curr_exp"

    data_attribute_types = {
        "cm"         : float, # nF  Capacity of the membrane
        "tau_m"      : float, # ms  Membrane time constant
        "tau_refrac" : float, # ms  Duration of refractory period
        "tau_syn_E"  : float, # ms  Decay time of excitatory synaptic curr
        "tau_syn_I"  : float, # ms  Decay time of inhibitory synaptic curr
        "i_offset"   : float, # nA  Offset current
        "v_rest"     : float, # mV  Rest potential
        "v_reset"    : float, # mV  Reset potential after a spike
        "v_thresh"   : float, # mV  Spike threshold
    }


class NeuronParametersCurrentAlpha(NeuronParameters):
    pynn_model = "IF_curr_alpha"

    data_attribute_types = {
        "cm"         : float, # nF  Capacity of the membrane
        "tau_m"      : float, # ms  Membrane time constant
        "tau_refrac" : float, # ms  Duration of refractory period
        "tau_syn_E"  : float, # ms  Decay time of excitatory synaptic curr
        "tau_syn_I"  : float, # ms  Decay time of inhibitory synaptic curr
        "i_offset"   : float, # nA  Offset current
        "v_rest"     : float, # mV  Rest potential
        "v_reset"    : float, # mV  Reset potential after a spike
        "v_thresh"   : float, # mV  Spike threshold
    }


class NativeNestMixin(object):
    """
        Mixin that marks neuron models that are only compatible with nest. 
    """
    data_attribute_types = {
        "nest_model" : str,
    }
    nest_only_attributes = []

    # Can be used to give a tanslations dictionary from PyNN.
    # These are the same translations attributes as in all PyNN standardmodels.
    # If it is a string it is interpreted as standardmodel to take.
    # translations from.
    translations = None

    def get_pynn_parameters(self, adjusted_parameters=None):
        dikt = self.get_dict()

        # apply adjustments before any translation take place
        if adjusted_parameters is not None:
            dikt.update(adjusted_parameters)

        del dikt["pynn_model"]
        del dikt["nest_model"]

        for noa in self.nest_only_attributes:
            del dikt[noa]

        dikt = self.translate_parameters_to_nest(dikt)

        return dikt

    def translate_parameters_to_nest(self, dikt):
        if self.translations is None:
            return dikt

        translations = self.get_translations_pynn()

        dikt = copy.deepcopy(dikt)

        translated = {}

        for k, v in dikt.iteritems():
            if k in translations:
                # if there is a translation, apply it
                # (taken in accordance to PyNN source)
                t = translations[k]
                f = t["forward_transform"]
                if callable(f):
                    t_value = f(**dikt)
                else:
                    t_value = eval(f, globals(), dikt)

                translated[t["translated_name"]] = t_value

            else:
                # if there is no translation, just copy the parameter
                translated[k] = v

        return translated

    def get_nest_parameters(self):
        dikt = self.get_dict()
        nest_params = {k: dikt[k] for k in self.nest_only_attributes}
        return nest_params

    def get_pynn_model_object(self, sim):
        celltype = sim.native_cell_type(self.nest_model)

        translations = self.get_translations_pynn()

        if translations is not None:
            celltype.translations = copy.deepcopy(translations)

        return celltype

    def get_translations_pynn(self):
        if isinstance(self.translations, six.string_types):
            # if translations is a string we copy the translation from the
            # specified standardmodel
            # (we have to do this in order to avoid importing PyNN until we
            # actually create any objects)
            import pyNN.nest as sim
            return getattr(sim, self.translations).translations

        else:
            return self.translations


class RandomRefractoryMixin(NativeNestMixin):
    # don't set anything tau_refrac related via pyNN
    nest_only_attributes = ["tau_refrac", "tau_refrac_dist"] +\
                           NativeNestMixin.nest_only_attributes

    data_attribute_types = dict(it.chain({
        # Dictionary describing one of the random distributions mentioned here:
        # http://www.nest-simulator.org/connection_management/
        # Please note that if the distribution emits float values they are
        # interpreted as ms. On the other hand, integers are interpreted as
        # simulation steps!
        #
        # NOTE: tau_refrac is still used as a mean value for calibration!!
        "tau_refrac_dist" : dict,
        }.iteritems(), NativeNestMixin.data_attribute_types.iteritems()))

    def get_nest_parameters(self):
        params = super(RandomRefractoryMixin, self).get_nest_parameters()
        params["t_ref"] = params["tau_refrac_dist"]
        del params["tau_refrac_dist"]
        del params["tau_refrac"]
        return params


class ConductanceExponentialRandomRefractory(
        RandomRefractoryMixin, NeuronParametersConductanceExponential):

    nest_model = "iaf_cond_exp_rtr"
    translations = "IF_cond_exp"


class CurrentExponentialRandomRefractory(
        RandomRefractoryMixin, NeuronParametersCurrentExponential):

    nest_model = "iaf_psc_exp_rtr"
    translations = "IF_curr_exp"

