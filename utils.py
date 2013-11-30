#!/usr/bin/env python
# encoding: utf-8

import numpy as np
from scipy.special import erf


def IF_cond_exp_distribution(rates_exc, rates_inh, weights_exc, weights_inh,
        e_rev_E, e_rev_I, tau_syn_E, tau_syn_I, g_l, v_rest, cm):
    """
    High Conductance State distribution

    Source parameters are expected to be numpy arrays.
    Unit for rates is Hz!

    All parameters are pynn parameters.

    g_l: leak_conductance
    """
    # convert rates to kHz
    rates_exc /= 1000.
    rates_inh /= 1000.

    # calculate exc, inh and total conductance

    g_exc = np.dot(weights_exc, rates_exc) * tau_syn_E
    g_inh = np.dot(weights_inh, rates_inh) * tau_syn_I
    g_tot = g_exc + g_inh + g_l

    # calculate effective (mean) membrane potential and time constant

    tau_eff = cm / g_tot
    v_eff = (e_rev_E * g_exc + e_rev_I * g_inh + El * g_l) / g_tot

    ####### calculate variance of membrane potential #######

    tau_g_exc = 1. / (1. / tau_syn_E - 1. / tau_eff)
    tau_g_inh = 1. / (1. / tau_syn_I - 1. / tau_eff)
    S_exc = weights_exc * (e_rev_E - v_eff) * tau_g_exc / tau_eff / g_tot
    S_inh = weights_inh * (e_rev_I - v_eff) * tau_g_inh / tau_eff / g_tot

    var = np.dot(rates_exc, S_exc**2)\
              * (tau_g_exc/2. + tau_eff/2.\
                  - 2. * tau_eff * tau_g_exc / (tau_eff + tau_g_exc))\
          + np.dot(rates_inh, S_inh**2)\
              * (tau_g_inh/2. + tau_eff/2.\
                  - 2. * tau_eff * tau_g_inh / (tau_eff + tau_g_inh))

    return v_eff, p.sqrt(var), g_tot


def IF_curr_exp_distribution(rates_exc, rates_inh, weights_exc, weights_inh,
        v_rest, tau_syn_E, tau_syn_I, g_l, cm):
    """
        Vmem distribution
        Unit for rates is Hz!

        All parameters are pynn parameters.

        g_l : leak conductance in µS
    """
    # convert rates to kHz
    rates_exc /= 1000.
    rates_inh /= 1000.

    # calculate total current and conductance

    I_exc = np.dot(weights_exc, rates_exc) * tau_syn_E
    I_inh = np.dot(weights_inh, rates_inh) * tau_syn_I
    g_tot = g_l

    # calculate effective (mean) membrane potential and time constant #######

    tau_eff = cm / g_tot
    v_eff = (I_exc + I_inh) / g_l + v_rest

    # calculate variance of membrane potential

    tau_g_exc = 1. / (1. / tau_syn_E - 1. / tau_eff)
    tau_g_inh = 1. / (1. / tau_syn_I - 1. / tau_eff)
    S_exc = weights_exc * taug_exc / tau_eff / g_tot
    S_inh = weights_inh * taug_inh / tau_eff / g_tot

    var = np.dot(rates_exc, S_exc**2)\
            * (tau_syn_E/2. + tau_eff/2.\
                - 2. * tau_eff * tau_syn_E / (tau_eff + tau_syn_E))\
        + np.dot(rates_inh, S_inh**2)\
            * (tau_syn_I/2. + tau_eff/2.\
                - 2. * tau_eff * tau_syn_I / (tau_eff + tau_syn_I))

    return v_eff, np.sqrt(var), g_tot


def sigmoid(x):
    return 1./(1.+np.exp(-x))


def gauss(x, mean, sigma):
    return 1./np.sqrt(2.*np.pi)/sigma*np.exp(-(x-mean)**2/2./sigma**2)


def erfm(x, mean, sigma):
    return .5*(1.+erf((x-mean)/p.sqrt(2.)/sigma))


def get_all_source_parameters(db_sources):
    """
        Returns a tuple of `np.array`s with source configuration
            (rates_exc, rates_inh, weights_exc, weights_inh)
    """

    num_sources = len(db_sources)
    num_sources_exc = len(filter(lambda x: x.is_exc), db_sources)
    num_sources_inh = num_sources - num_sources_exc

    rates_exc = np.empty((num_sources_exc,))
    weights_exc = np.empty((num_sources_exc,))

    rates_inh = np.empty((num_sources_inh,))
    weights_inh = np.empty((num_sources_inh,))

    i_exc = 0
    i_inh = 0

    for src in db_sources:
        if src.is_exc:
            rates_exc[i_exc] = src.rate
            weights_exc[i_exc] = src.weight
            i_exc += 1
        else:
            rates_inh[i_inh] = src.rate
            weights_inh[i_inh] = src.weight
            i_inh += 1

    return rates_exc, rates_inh, weights_exc, weights_inh

