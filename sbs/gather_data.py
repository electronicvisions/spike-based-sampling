#!/usr/bin/env python
# encoding: utf-8

"""
    Gather calibration data in another process.
"""

import os
import sys
import os.path as osp
import functools as ft
import numpy as np
import subprocess as sp
import itertools as it
import logging
from pprint import pformat as pf
import time

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

# NOTE: No relative imports here because the file will also be executed as
#       script.
from . import comm
from . import logcfg
from .logcfg import log
from . import buildingblocks as bb
from . import utils
from . import db
from .samplers import LIFsampler

_subprocess_silent = False

def set_subprocess_silent(silent=False):
    global _subprocess_silent
    _subprocess_silent = silent


class SendLogLevelMixin(object):
    def  _send_arguments(self, socket, args, kwargs):
        log.debug("Sending loglevel information.")
        comm.send_object(socket, {log.name : log.getEffectiveLevel()})
        return super(SendLogLevelMixin, self)._send_arguments(
                socket, args, kwargs)

    def _recv_arguments(self, socket):
        log.debug("Receiving loglevel information.")
        loglvls = comm.recv_object(socket)
        for logname, loglevel in loglvls.iteritems():
            logging.getLogger(logname).setLevel(loglevel)
        return super(SendLogLevelMixin, self)._recv_arguments(socket)


class RunInSubprocessWithDatabase(SendLogLevelMixin, comm.RunInSubprocess):
    """
        Send current database information along to the subprocess.


        (So that reading from database works in subprocess.)
    """
    def _spawn_process(self, script_filename):
        log.debug("Spawning subprocess..")
        output = None
        if _subprocess_silent:
            output = open(os.devnull, 'w')
        return sp.Popen([sys.executable, script_filename],
                cwd=self._func_dir, stdout=output, stderr=output)

    def _send_arguments(self, socket, args, kwargs):
        log.debug("Sending database settings.")
        comm.send_object(socket, {"current_basename" : db.current_basename})
        return super(RunInSubprocessWithDatabase, self)\
                ._send_arguments(socket, args, kwargs)

    def _recv_arguments(self, socket):
        assert db.current_basename is None

        log.debug("Receiving database settings.")
        db_data = comm.recv_object(socket)
        current_basename = db_data["current_basename"]
        db.setup(current_basename)

        return super(RunInSubprocessWithDatabase, self)\
                ._recv_arguments(socket)


def eta_from_burnin(t_start, burn_in, duration):
    eta = utils.get_eta(t_start, burn_in, duration+burn_in)
    if not isinstance(eta, basestring):
        eta = utils.format_time(eta)
    log.info("ETA (after burn-in): {}".format(eta))


# make a log function with ETA
def make_log_time(duration, num_steps=10, offset=0):
    increment = duration / num_steps
    duration = duration
    t_start = time.time()

    def log_time(time):
        eta = utils.get_eta(t_start, time, duration + offset)
        if type(eta) is float:
            eta = utils.format_time(eta)

        log.info("{} ms / {} ms. ETA: {}".format(
            time - offset,
            duration,
            eta))
        return time + increment

    return log_time


# get callbacks dependant on backend
def get_callbacks(sim, log_time_params):
    """
        `sim` should be the simulator backend used.
    """

    if sim.__name__.split(".")[-1] == "neuron":
        # neuron does not wörk with callbacks
        callbacks = []
    else:
        callbacks = [make_log_time(**log_time_params)]

    return callbacks


############################
# SAMPLER HELPER FUNCTIONS #
############################

@comm.RunInSubprocess
def gather_calibration_data(
        sampler_config=None):
    """
        This function performs a single calibration run and should normally run
        in a seperate subprocess (which it is when called from LIFsampler).

        It does not fit the sigmoid.
    """
    log.info("Calibration started.")
    log.info("Preparing network.")

    calibration = sampler_config.calibration
    neuron_params = sampler_config.neuron_parameters

    sampler = LIFsampler(sampler_config, sim_name=calibration.sim_name,
            silent=True)

    if calibration.sim_setup_kwargs is None:
        sim_setup_kwargs = {}
    else:
        sim_setup_kwargs = calibration.sim_setup_kwargs

    exec("import {} as sim".format(calibration.sim_name))

    samples_v_rest = calibration.get_samples_v_rest()

    log.info("Gathering {} samples in [{}, {}] mV.".format(
        len(samples_v_rest), samples_v_rest.min(), samples_v_rest.max()))

    burn_in_time = calibration.burn_in_time
    duration = calibration.duration
    total_duration = burn_in_time + duration

    # TODO maybe implement a seed here
    sim.setup(timestep=calibration.dt, **sim_setup_kwargs)

    # create sources
    # sources = bb.create_sources(sim, calibration.source_config, total_duration)

    log.info("Setting up {} samplers.".format(len(samples_v_rest)))
    if log.getEffectiveLevel() <= logging.DEBUG:
        log.debug("Sampler params: {}".format(pf(neuron_params)))

    pop = sampler.create(num_neurons=len(samples_v_rest),
            ignore_calibration=True, duration=total_duration)
    pop.record("spikes")
    pop.initialize(v=samples_v_rest)

    if isinstance(neuron_params, db.NativeNestMixin):
        # the nest-native parameter for v_rest is E_L
        pop.set(E_L=samples_v_rest)
    else:
        pop.set(v_rest=samples_v_rest)

    if log.getEffectiveLevel() <= logging.DEBUG:
        if isinstance(neuron_params, db.NativeNestMixin):
            for i, s in enumerate(pop):
                # the nest-native parameter for v_rest is E_L
                log.debug("v_rest of neuron #{}: {} mV".format(i, s.E_L))
        else:
            for i, s in enumerate(pop):
                log.debug("v_rest of neuron #{}: {} mV".format(i, s.v_rest))

    # connect the two
    # projections = bb.connect_sources(sim, calibration.source_config, sources,
            # samplers)
    #  sources, projection = calibration.source_config.create_connect(sim,
            #  pop, duration=total_duration)

    callbacks = get_callbacks(sim, {
            "duration" : calibration.duration,
            "offset" : burn_in_time,
        })

    # bring samplers into high conductance state
    log.info("Burning in samplers for {} ms".format(burn_in_time))
    t_start = time.time()
    sim.run(burn_in_time)
    eta_from_burnin(t_start, burn_in_time, duration)

    log.info("Generating calibration data..")
    sim.run(duration, callbacks=callbacks)

    log.info("Reading spikes.")
    spiketrains = pop.get_data("spikes").segments[0].spiketrains
    if log.getEffectiveLevel() <= logging.DEBUG:
        for i, st in enumerate(spiketrains):
            log.debug("{}: {}".format(i, pf(st)))
    num_spikes = np.array([(s > burn_in_time).sum() for s in spiketrains],
            dtype=int)

    samples_p_on = num_spikes * neuron_params.tau_refrac / duration

    if log.getEffectiveLevel() <= logging.DEBUG:
        log.debug("Samples p_on:\n{}".format(pf(samples_p_on)))

    # TODO: Put in debug logging
    log.info("Resulting p_on: {}+-{}".format(
        samples_p_on.mean(), samples_p_on.std()))

    sim.end()

    return samples_p_on


@comm.RunInSubprocess
def gather_free_vmem_trace(distribution_params, sampler, adjusted_v_thresh=50.):
    """
        Records a voltage trace of the free membrane potential of the given
        neuron model with the given parameters.

        adjusted_v_tresh is the value the neuron-threshold will be set to
        to avoid spiking.
    """
    dp = distribution_params
    log.info("Preparing to take free Vmem distribution")
    exec("import {} as sim".format(sampler.sim_name))

    if sampler.calibration.sim_setup_kwargs is None:
        sim_setup_kwargs = {}
    else:
        sim_setup_kwargs = sampler.calibration.sim_setup_kwargs

    sim.setup(timestep=dp["dt"], **sim_setup_kwargs)

    total_duration = dp["duration"] + dp["burn_in_time"]

    population = sampler.create(total_duration)

    population.record("v")
    population.initialize(v=sampler.get_pynn_parameters()["v_rest"])
    population.set(v_thresh=adjusted_v_thresh)

    callbacks = get_callbacks(sim, {
            "duration" : dp["duration"],
            "offset" : dp["burn_in_time"],
        })
    log.info("Burning in samplers for {} ms".format(dp["burn_in_time"]))
    t_start = time.time()
    sim.run(dp["burn_in_time"])
    eta_from_burnin(t_start, dp["burn_in_time"], dp["duration"])

    log.info("Starting data gathering run.")
    sim.run(dp["duration"], callbacks=callbacks)

    data = population.get_data("v")

    offset = int(dp["burn_in_time"]/dp["dt"])
    voltage_trace = np.array(data.segments[0].analogsignalarrays[0])[offset:, 0]
    voltage_trace = np.require(voltage_trace, requirements=["C"])

    sim.end()

    return voltage_trace


#####################################
# SAMPLING NETWORK HELPER FUNCTIONS #
#####################################

@comm.RunInSubprocess
def gather_network_spikes(network, duration, dt=0.1, burn_in_time=0.,
        create_kwargs=None, sim_setup_kwargs=None, initial_vmem=None):
    """
        create_kwargs: Extra parameters for the networks creation routine.

        sim_setup_kwargs: Extra parameters for the setup command (random seeds
        etc.).
    """

    if sim_setup_kwargs is None:
        sim_setup_kwargs = {}

    exec "import {} as sim".format(network.sim_name) in globals(), locals()

    sim.setup(timestep=dt, **sim_setup_kwargs)

    if create_kwargs is None:
        create_kwargs = {}
    population, projections = network.create(duration=duration, **create_kwargs)

    if isinstance(population, sim.common.BasePopulation):
        population.record("spikes")
        if initial_vmem is not None:
            population.initialize(v=initial_vmem)
    else:
        for pop in population:
            pop.record("spikes")
        if initial_vmem is not None:
            for pop, v in it.izip(population, initial_vmem):
                pop.initialize(v=v)


    callbacks = get_callbacks(sim, {
            "duration" : duration,
            "offset" : burn_in_time,
        })

    t_start = time.time()
    if burn_in_time > 0.:
        log.info("Burning in samplers for {} ms".format(burn_in_time))
        sim.run(burn_in_time)
        eta_from_burnin(t_start, burn_in_time, duration)

    log.info("Starting data gathering run.")
    sim.run(duration, callbacks=callbacks)

    if isinstance(population, sim.common.BasePopulation):
        spiketrains = population.get_data("spikes").segments[0].spiketrains
    else:
        spiketrains = np.vstack(
                [pop.get_data("spikes").segments[0].spiketrains[0]
                for pop in population])

    # we need to ignore the burn in time
    clean_spiketrains = []
    for st in spiketrains:
        clean_spiketrains.append(np.array(st[st > burn_in_time])-burn_in_time)

    return_data = {
            "spiketrains" : clean_spiketrains,
            "duration" : duration,
            "dt" : dt,
        }
    sim.end()

    return return_data


