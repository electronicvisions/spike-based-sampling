#!/usr/bin/env python2
# encoding: utf-8

from .logcfg import log
import logging
import functools as ft


def pynn_get_analogsignals(segment):
    try:  # pynn 0.9 syntax
        return segment.analogsignals
    except AttributeError:
        return segment.analogsignalarrays


def fix_nest_synapse_defaults(
        synapse_model_to_check="tsodyks2_synapse",
        undesired_keys=[
                "synapse_model",
                "has_delay",
                "requires_symmetric",
                "weight_recorder"
        ]):
    """
        The following is a workaround to the problem that
        get_synapse_defaults from PyNN.nest does not ignore the
        keys "synapse_model", "has_delay", "requires_symmetric"
        and "weight_recorder" used by tsodyks2 in its Defaults.
        These keys hence falsely enter default_params and cause
        some errors. The following aims to achieve the same as
        the present get_synapse_defaults-method in
        PyNN0-0.8.3/pyNN/nest/synapses.py - however, it would be
        much easier if pyNN.nest had a way of augmenting the
        ignore list.

        `synapse_model_to_check` is a synapse type in nest that is used to
        check the default values.

        If any more errors are encountered, the list of undesired keys might
        have to be augmented.
    """
    # First, check if the error is present:
    import importlib
    synapses = importlib.import_module("pyNN.nest.synapses")
    present_defaults = synapses.get_synapse_defaults(synapse_model_to_check)

    if any((k in present_defaults for k in undesired_keys)):
        log.warn("Patching pyNN to work with nest 2.12.0+, "
                 "please consider updating!")

        if log.getEffectiveLevel() <= logging.DEBUG:
            log.debug("Offending keys in default synapse dictionary: " +
                      ", ".join((k for k in undesired_keys
                                 if k in present_defaults)))

        # we need to monkey patch the
        # get_synapse_defaults-method

        original_fct = synapses.get_synapse_defaults

        @ft.wraps(original_fct)
        def patched(modelname):
            defaults = original_fct(modelname)

            return {k: v for k, v in defaults.iteritems()
                    if k not in undesired_keys}

        synapses.get_synapse_defaults = patched


def fix_nest_tsodyks(alternative_name="avoid_pynn_trying_to_be_smart"):
    """
        For reasons that are beyond me, PyNN.nest thinks it is a
        good idea to inject a 'tau_psc' parameter in all
        connections with 'tsodyks' in their name.
        Hence we need to rename the tsodyks2 synapse to something
        else.

        I am at a loss for words..
    """
    log.warn(
        "This is a stupid hack and needs to be fixed in pyNN.nest")

    import nest
    import pyNN.nest as sim
    if alternative_name not in nest.Models():
        sim.nest.CopyModel("tsodyks2_synapse_lbl",
                           alternative_name + "_lbl")
        sim.nest.CopyModel("tsodyks2_synapse",
                           alternative_name)
