#!/usr/bin/env python
# encoding: utf-8

import numpy as np
from scipy.special import erf
import string
import hashlib
import collections as c
import time
import itertools as it
import logging
from pprint import pformat as pf
import gzip
import struct
import os
import os.path as osp
try:
    import cPickle as pickle
except ImportError:
    import pickle

from . import cutils

from .logcfg import log

__all__ = [
    "IF_cond_exp_distribution",
    "IF_cond_alpha_distribution",
    "IF_curr_exp_distribution",
    "IF_curr_alpha_distribution",
    "check_list_array",
    "erfm",
    "fill_diagonal",
    "filter_dict",
    "format_time",
    "gauss",
    "get_eta",
    "get_elapsed",
    "get_ordered_spike_idx",
    "get_random_string",
    "get_sha1",
    "get_time_tuple",
    "load_pickle",
    "nest_change_poisson_rate"
    "nest_copy_model",
    "nest_key_connections",
    "save_pickle",
    "run_with_eta",
    "sigmoid",
    "sigmoid_trans",
]

def IF_cond_exp_distribution(rates_exc, rates_inh, weights_exc, weights_inh,
        e_rev_E, e_rev_I, tau_syn_E, tau_syn_I, g_l, v_rest, cm,
        **sink): #sink is just to absorb unused parameter names
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
    weights_exc = np.abs(weights_exc)
    weights_inh = np.abs(weights_inh)

    g_exc = np.dot(weights_exc, rates_exc) * tau_syn_E
    g_inh = np.dot(weights_inh, rates_inh) * tau_syn_I
    g_tot = g_exc + g_inh + g_l

    # calculate effective (mean) membrane potential and time constant

    tau_eff = cm / g_tot
    v_eff = (e_rev_E * g_exc + e_rev_I * g_inh + v_rest * g_l) / g_tot

    log.debug("tau_eff: {:.3f} ms".format(tau_eff))

    ####### calculate variance of membrane potential #######

    tau_g_exc = 1. / (1. / tau_syn_E - 1. / tau_eff)
    tau_g_inh = 1. / (1. / tau_syn_I - 1. / tau_eff)

    S_exc = weights_exc * (e_rev_E - v_eff) * tau_g_exc / tau_eff / g_tot
    S_inh = weights_inh * (e_rev_I - v_eff) * tau_g_inh / tau_eff / g_tot

    var_tau_e = tau_syn_E/2. + tau_eff/2.\
            - 2. * tau_eff * tau_syn_E / (tau_eff + tau_syn_E)

    var_tau_i = tau_syn_I/2. + tau_eff/2.\
            - 2. * tau_eff * tau_syn_I / (tau_eff + tau_syn_I)

    # log.info("v_tau_e/v_tau_i: {} / {}".format(v_tau_e, v_tau_i))

    var = np.dot(rates_exc, S_exc**2) * var_tau_e\
        + np.dot(rates_inh, S_inh**2) * var_tau_i

    return v_eff, np.sqrt(var), g_tot, tau_eff


def IF_cond_alpha_distribution(rates_exc, rates_inh, weights_exc, weights_inh,
        e_rev_E, e_rev_I, tau_syn_E, tau_syn_I, g_l, v_rest, cm,
        **sink): #sink is just to absorb unused parameter names
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
    weights_exc = np.abs(weights_exc)
    weights_inh = np.abs(weights_inh)

    g_exc = np.dot(weights_exc, rates_exc) * tau_syn_E * np.exp(1.)
    g_inh = np.dot(weights_inh, rates_inh) * tau_syn_I * np.exp(1.)
    g_tot = g_exc + g_inh + g_l

    # calculate effective (mean) membrane potential and time constant

    tau_eff = cm / g_tot
    v_eff = (e_rev_E * g_exc + e_rev_I * g_inh + v_rest * g_l) / g_tot

    log.debug("tau_eff: {:.3f} ms".format(tau_eff))

    ####### calculate variance of membrane potential #######

    tau_g_exc = 1. / (1. / tau_syn_E - 1. / tau_eff)
    tau_g_inh = 1. / (1. / tau_syn_I - 1. / tau_eff)

    # s for sum
    tau_s_exc = 1. / (1. / tau_syn_E + 1. / tau_eff)
    tau_s_inh = 1. / (1. / tau_syn_I + 1. / tau_eff)

    S_exc = weights_exc * (e_rev_E - v_eff) * tau_g_exc / tau_eff / g_tot
    S_inh = weights_inh * (e_rev_I - v_eff) * tau_g_inh / tau_eff / g_tot

    S_exc *= np.exp(1.)
    S_inh *= np.exp(1.)

    # var_tau_e = tau_syn_E/2. + tau_eff/2.\
            # - 2. * tau_eff * tau_syn_E / (tau_eff + tau_syn_E)

    # var_tau_i = tau_syn_I/2. + tau_eff/2.\
            # - 2. * tau_eff * tau_syn_I / (tau_eff + tau_syn_I)

    var_tau_exc = tau_syn_E**3 / 4.\
            + 2. * tau_g_exc * (tau_syn_E**2 / 4. - tau_s_exc**2)\
            + tau_g_exc**2 * ((tau_syn_E + tau_eff)/2. - 2*tau_s_exc)
    var_tau_inh = tau_syn_I**3 / 4.\
            + 2. * tau_g_inh * (tau_syn_I**2 / 4. - tau_s_inh**2)\
            + tau_g_inh**2 * ((tau_syn_I + tau_eff)/2. - 2*tau_s_inh)

    # log.info("v_tau_e/v_tau_i: {} / {}".format(v_tau_e, v_tau_i))

    var = np.dot(rates_exc, S_exc**2) * var_tau_exc\
        + np.dot(rates_inh, S_inh**2) * var_tau_inh

    return v_eff, float(np.sqrt(var)), g_tot, tau_eff


def IF_curr_exp_distribution(rates_exc, rates_inh, weights_exc, weights_inh,
        v_rest, tau_syn_E, tau_syn_I, g_l, cm,
        **sink): #sink is just to absorb unused parameter names
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

    log.debug("tau_eff: {:.3f}".format(tau_eff))

    # calculate variance of membrane potential

    tau_g_exc = 1. / (1. / tau_syn_E - 1. / tau_eff)
    tau_g_inh = 1. / (1. / tau_syn_I - 1. / tau_eff)

    S_exc = weights_exc * tau_g_exc / tau_eff / g_tot
    S_inh = weights_inh * tau_g_inh / tau_eff / g_tot

    var = np.dot(rates_exc, S_exc**2)\
            * (tau_syn_E/2. + tau_eff/2.\
                - 2. * tau_eff * tau_syn_E / (tau_eff + tau_syn_E))\
        + np.dot(rates_inh, S_inh**2)\
            * (tau_syn_I/2. + tau_eff/2.\
                - 2. * tau_eff * tau_syn_I / (tau_eff + tau_syn_I))

    return v_eff, float(np.sqrt(var)), g_tot, tau_eff


def IF_curr_alpha_distribution(rates_exc, rates_inh, weights_exc, weights_inh,
        v_rest, tau_syn_E, tau_syn_I, g_l, cm,
        **sink): #sink is just to absorb unused parameter names
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

    I_exc = np.dot(weights_exc, rates_exc) * tau_syn_E * np.exp(1.)
    I_inh = np.dot(weights_inh, rates_inh) * tau_syn_I * np.exp(1.)
    g_tot = g_l

    # calculate effective (mean) membrane potential and time constant #######

    tau_eff = cm / g_tot
    v_eff = (I_exc + I_inh) / g_l + v_rest

    log.debug("tau_eff: {:.3f}".format(tau_eff))

    # calculate variance of membrane potential

    tau_g_exc = 1. / (1. / tau_syn_E - 1. / tau_eff)
    tau_g_inh = 1. / (1. / tau_syn_I - 1. / tau_eff)

    # s for sum
    tau_s_exc = 1. / (1. / tau_syn_E + 1. / tau_eff)
    tau_s_inh = 1. / (1. / tau_syn_I + 1. / tau_eff)

    S_exc = I_exc * tau_g_exc / tau_eff / g_tot
    S_inh = I_inh * tau_g_inh / tau_eff / g_tot

    var_tau_exc = tau_syn_E**3 / 4.\
            + 2. * tau_g_exc * (tau_syn_E**2 / 4. - tau_s_exc**2)\
            + tau_g_exc**2 * ((tau_syn_E + tau_eff)/2. - 2*tau_s_exc)
    var_tau_inh = tau_syn_I**3 / 4.\
            + 2. * tau_g_inh * (tau_syn_I**2 / 4. - tau_s_inh**2)\
            + tau_g_inh**2 * ((tau_syn_I + tau_eff)/2. - 2*tau_s_inh)

    var = np.dot(rates_exc, S_exc**2) * var_tau_exc\
        + np.dot(rates_inh, S_inh**2) * var_tau_inh

    return v_eff, float(np.sqrt(var)), g_tot, tau_eff

# IF_cond_exp_cd_distribution = IF_cond_exp_distribution
# IF_curr_exp_cd_distribution = IF_curr_exp_distribution

# IF_cond_alpha_cd_distribution = IF_cond_alpha_distribution
# IF_curr_alpha_cd_distribution = IF_curr_alpha_distribution


def sigmoid(x):
    return 1./(1.+np.exp(-x))


def sigmoid_trans(x, x_p05, alpha):
    return 1./(1. + np.exp( -(x-x_p05)/alpha))


def gauss(x, mean, sigma):
    return 1./np.sqrt(2.*np.pi)/np.abs(sigma)*np.exp(-(x-mean)**2/2./sigma**2)


def erfm(x, mean, sigma):
    return .5*(1.+erf((x-mean)/np.sqrt(2.)/np.abs(sigma)))


def fill_diagonal(array, value=0):
    """
        Fill the diagonal of `array` with `value`.
    """
    # ensure quadratic form
    for s in array.shape:
        assert s == array.shape[0]

    index = np.arange(array.shape[0], dtype=int)

    indices = [index for s in array.shape]

    array[indices] = value

    return array


def get_urandom_num(n=1, BYTE_LEN=8):
    rand_bytes = os.urandom(BYTE_LEN*n)
    return (struct.unpack("L", rand_bytes[i*BYTE_LEN:(i+1)*BYTE_LEN])[0]
            for i in xrange(n))


def get_random_string(n=32, letters=string.ascii_letters):
    nums = get_urandom_num(n)
    return "".join((letters[i % len(letters)] for i in nums))


def get_sha1(array):
    sha1 = hashlib.sha1()
    sha1.update(array)
    return sha1.hexdigest()


TimeTuple = c.namedtuple("duration", "d h m s ms".split())

# durations in seconds
TIME_DELTAS = TimeTuple(24*3600, 3600, 60, 1, .001)

def make_time_closure_writable(timediff):
    timediff = [timediff]
    def sub(s):
        # closures need mutable objects to write to, but
        # numbers in themselves are immutable
        t = timediff[0]
        timediff[0] = np.mod(t, s)
        return int(np.floor(t/s))
    return sub


def get_time_tuple(timediff):
    return TimeTuple(*map(make_time_closure_writable(timediff), TIME_DELTAS))


def format_time(timediff):
    fmtd = get_time_tuple(timediff)
    return " ".join(
            ("{0}{1}".format(getattr(fmtd, s), s) for s in fmtd._fields
                if getattr(fmtd, s) > 0.))


def get_eta(t_start, current, total):
    """
        Estimate time it takes to finish for simulation of work `total`, if
        the simulation was started at `t_start` and has done work `current`.
    """
    t_elapsed = time.time() - t_start
    if current > 0.:
        return t_elapsed / current * (total - current)
    else:
        return "N/A"


def get_eta_str(t_start, current, total):
    """
       Return the estimated time remaining as pre-formatted string.

       `t_start` is the actual time.time() when operations began, `current` is
       the amount of work already done and `total` is the total amount of work
       to do.

       Example:
           t_start = time.time()
           num_runs = 10
           for i in xrange(num_runs):
               # <do some work>
               get_eta_str(t_start, i, num_runs)
    """
    t_elapsed = time.time() - t_start
    if current > 0 and current < total:
        return format_time(t_elapsed / current * (total - current))
    else:
        return "N/A"


def get_elapsed_str(t_start):
    """
        Return time elapsed from `t_start` as preformatted string.
    """
    return format_time(time.time() - t_start)


def save_pickle(obj, filename, force_extension=False, compresslevel=9):
    """
        Save object in compressed pickle filename.

        By default the extension of the filename will always be replaced by
        "pkl.gz". If you want to force a custom extension, set
        force_extension=True.
    """

    if not force_extension:
        filename = osp.splitext(filename)[0] + ".pkl.gz"

    with gzip.open(filename, "wb", compresslevel=compresslevel) as f:
        pickle.dump(obj, f, protocol=-1)


def load_pickle(filename, force_extension=False):
    """
        Load pickle object from file, if `force_extension` is True, the
        extension will NOT be changed to ".pkl.gz" (the user specified
        extension will be forced).
    """
    if not force_extension:
        filename = osp.splitext(filename)[0] + ".pkl.gz"

    if filename.split(osp.extsep)[-1] == "gz":
        file_opener = gzip.open
    else:
        file_opener = open

    with file_opener(filename) as f:
        return pickle.load(f)


def get_ordered_spike_idx(spiketrains):
    """
        Take spike trains and return a (num_spikes,) record array that contains
        the spike ids ('id') on first and the spike times ('t') on second
        position. The spike times are sorted in ascending order.
    """
    num_spikes = sum((len(st) for st in spiketrains))
    spikes = np.zeros((num_spikes,), dtype=[("id", int), ("t", float)])

    current = 0

    for i,st in enumerate(spiketrains):
        if log.getEffectiveLevel() <= logging.DEBUG:
            log.debug("Raw spikes for #{}: {}".format(i, pf(st)))
        spikes["id"][current:current+len(st)] = i
        spikes["t"][current:current+len(st)] = np.array(st)

        current += len(st)

    sort_idx = np.argsort(spikes["t"])
    sorted_spikes = spikes[sort_idx].copy()

    return sorted_spikes


def check_list_array(obj):
    return isinstance(obj, c.Sequence) or isinstance(obj, np.ndarray)


def dkl(p, q):
    """
        Kullback-Leibler divergence
    """
    idx = (p > 0) * (q > 0)
    p = p[idx]
    q = q[idx]
    return np.sum(p * np.log(p/q))


def dkl_sum_marginals(ps, qs):
    """
        Compute the marginal for each pair of p's and q's and sum the resulting
        DKLs.

        Note that the p' s only consist of a single state (the other will be
        calculated).
    """
    dkl = 0.
    for p, q in it.izip(ps, qs):
        dkl += p * np.log(p/q) + (1. - p) * np.log((1.-p)/(1.-q))
    return dkl


def nest_copy_model(base_model, pynn_compatible=True):
    """
        Make a new random copy of a nest model.

        If `pynn_compatible == True`, the labelled version of the synapse will
        be copied as well.
    """
    import nest
    models = nest.Models()
    while True:
        model_name = base_model + "_" + get_random_string(n=8)
        if model_name not in models:
            break
    nest.CopyModel(base_model, model_name)
    if pynn_compatible:
        # make labelled version available to pyNN
        nest.CopyModel(base_model + "_lbl", model_name + "_lbl")
    return model_name


def nest_key_connections(conn):
    """
        Key for sorting nest connections.

        nest.GetConnections(...) returns connection objects by thread-order,
        hence the ordering of connections is different in the single- and each
        multithreaded case.

        When using this function to sort connections returned by
        nest.GetConnections(...), they will have the same ordering as in the
        single threaded setup.

        The ordering is:
            * source-gid
            * target-gid
            * receptor-port (usually unimpotant as we have at most one synapse
                             between any two nodes)

        We implicitly ignore synapse-model-id as well as thread-id.
    """
    # source-id, target-id, receptor-port
    return conn[0], conn[1], conn[4]


def nest_change_poisson_rate(bm_net, new_rate):
    """
        Convenience function:
        Changes the global rate of the BM `bm_net` to `new_rate`.
    """
    # NOTE: This is a dirty dirty hack ...
    log.info("Changing global Poisson rate to: {} Hz".format(new_rate))
    gid_generators = bm_net._pynn_sources[0][0]["generators"]

    assert len(gid_generators) == 1, "There is more than one Poisson generator currently active! " \
            "We do NOT want to hack that much!"

    import nest
    nest.SetStatus(gid_generators.tolist(), {"rate": new_rate})


def filter_dict(d, f):
    """
        Filter dictionary d based on function f that takes two arguments
        (key and value) and returns a bool to indicate if the key value
        pair should be kept.
    """
    return {k: v for k, v in d.iteritems() if f(k, v)}


def run_with_eta(sim, duration, num_steps=20):
    """
        Replacement for PyNN.run that seperates a pyNN.run-call of `duration`
        into `num_steps` smaller segments and gives an indication of the time
        remaining after each.

        Instead of doing `sim.run(duration)` simply use
        `run_with_eta(sim, duration)`.
    """
    dT = duration / num_steps
    t_start = time.time()
    for i in xrange(num_steps):
        sim.run(dT)
        log.info("Elapsed: {:<32} ETA: {}".format(
            get_elapsed_str(t_start),
            get_eta_str(t_start, i+1, num_steps)))

