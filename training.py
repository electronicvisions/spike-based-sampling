#!/usr/bin/env python2
# encoding: utf-8

# Special training methods
# Make use of some wrappers from SEMf

import itertools as it
import numpy as np
import time
from pprint import pformat as pf

# from SEMf import misc as m

from . import utils
from .logcfg import log
from .network import RapidRBMImprintCurrent

# RapidRBMImprintCurrent = m.ClassInSubprocess(RapidRBMImprintCurrent)

NEST_MAX_TIME = 26843545.5

def train_rbm_cd(
        training_data=None, # binary nd.array
                            # shape (n_labels, n_samples, n_visible)
        num_steps=15000,
        num_hidden=None,
        eta_func=lambda t: 10./(100.+t),
        bm_init_kwargs=None, # dict
        bm_settings=None,    # dict
        sim_setup_kwargs={"spike_precision" : "on_grid", "time_step": 0.1},
        seed=42,
        bm_type=RapidRBMImprintCurrent,
        steps_per_snapshot=0,
    ):
    """
        If steps_per_snapshot > 0, a weight update is taken from the network
        after every `steps_per_snapshot` steps.
    """
    assert len(training_data.shape) == 3

    np.random.seed(seed)

    num_labels, num_samples, num_visible = training_data.shape

    final_bm_settings = {
            # "current_wipe" : 10.,
            # "current_imprint" : 10.,
            # "current_force_spike" : 10.,

            # "time_wipe" : 20.,
            # "time_imprint" : 10.,
            # "time_force_spike" : 10.,
            "time_sim_step" : 10.,
            "time_recon_step" : 10.,
        }

    bm_init_kwargs["num_units_per_layer"] = [num_visible, num_hidden]

    final_bm_settings.update(bm_settings)

    time_recon_step = final_bm_settings["time_recon_step"]

    if steps_per_snapshot > 0:
        num_snapshots = num_steps / steps_per_snapshot + 1
        if num_steps % steps_per_snapshot != 0:
            num_snapshots += 1

        snapshots_weight = np.zeros((num_snapshots, num_visible, num_hidden))
        snapshots_bias = np.zeros((num_snapshots, num_visible + num_hidden))

    def init_net():
        bm = bm_type(**bm_init_kwargs)
        bm.create_no_return(sim_setup_kwargs=sim_setup_kwargs)

        for s in bm.samplers:
            s.silent = True

        bm.auto_sync_biases = False

        for k,v in final_bm_settings.iteritems():
            setattr(bm, k, v)

        # bm.update_weights()

        return bm

    bm = init_net()

    bm.population.record('spikes')

    sample_ids = np.random.randint(num_samples, size=num_steps)

    t_start = time.time()

    # set a random binary state in the beginning
    binary_state = np.ones(bm.num_samplers) + 1 # per default set nothing
    visible_state_model = np.random.randint(2, size=num_visible)

    labels = np.random.randint(num_labels, size=num_steps)

    num_active = np.zeros(num_steps, dtype=int)

    # i_l = 0 # which label
    i_s = 0 # which snapshot
    for i_step, i_samples in enumerate(sample_ids):

        i_l = labels[i_step]

        try:
            if i_step % int(num_steps / 20) == 0:
                log.info("Run #{}. [SimTime: {}ms] ETA: {}".format(i_step,
                    bm.time_current,
                    utils.get_eta_str(t_start, i_step, num_steps)))
        except ZeroDivisionError:
            pass


        visible_state_data = training_data[i_l, i_samples]
        binary_state = np.ones(bm.num_samplers) + 1 # per default set nothing
        binary_state[:num_visible] = visible_state_model
        bm.binary_state = binary_state

        bm.run()
        hidden_state_data = bm.binary_state[num_visible:].copy()
        # log.info("Hidden state: " + pf(hidden_state))
        # log.info("Last spiketimes: " + pf(bm.last_spiketimes))
        binary_state = np.ones(bm.num_samplers) + 1 # per default set nothing
        # binary_state[:num_visible] = visible_state_model

        bm.binary_state = binary_state
        bm.run()

        # hidden_state_model = bm.binary_state[num_visible:].copy()

        bm.continue_run(time_recon_step)
        visible_state_model = bm.binary_state[:num_visible].copy()
        hidden_state_model = bm.binary_state[num_visible:].copy()

        try:
            if i_step % int(num_steps / 20) == 0:
                log.info("Active neurons: {}".format((bm.binary_state == 1).sum()))
        except ZeroDivisionError:
            pass

        num_active[i_step] = (bm.binary_state == 1).sum()
        # bm.continue_run(time_recon_step)
        # log.info("Visible state: " + pf(visible_state))
        # log.info("Last spiketimes: " + pf(bm.last_spiketimes))

        if bm.time_current > 0.95 * NEST_MAX_TIME:
            weights_theo = bm.weights_theo
            bm.kill()
            bm = init_net()
            bm.weights_theo = weights_theo
            # bm.update_weights()

        update_factors = bm.update_factors
        update_factors[:, 0] = np.sqrt(eta_func(i_step))

        update_factors[:num_visible, 1] = visible_state_data
        update_factors[:num_visible, 2] = visible_state_model

        update_factors[num_visible:, 1] = hidden_state_data
        update_factors[num_visible:, 2] = hidden_state_model

        bias_update = eta_func(i_step)\
            * np.r_[visible_state_data - visible_state_model,
                hidden_state_data - hidden_state_model]

        bm.biases_theo = bm.biases_theo + bias_update

        bm.update_factors = update_factors

        bm.queue_update()

        if steps_per_snapshot > 0 and i_step % steps_per_snapshot == 0:
            snapshots_weight[i_s] = bm.weights_theo[0][0, :, :]
            snapshots_bias[i_s] = bm.biases_theo
            i_s += 1

        # i_l = (i_l+1) % num_labels

    # take last snapshot if needed
    if steps_per_snapshot > 0 and i_s < num_snapshots:
        snapshots_weight[i_s] = bm.weights_theo[0][0, :, :]

    bm.binary_state = np.ones(bm.num_samplers, dtype=int)
    bm.run()

    log.info(pf(bm.weights_theo))
    # TODO: Error when querying bio weights
    log.info(pf(bm.biases_theo))
    retval = {
        "final_weights" : bm.weights_theo,
        "final_biases" : bm.biases_theo,
        "spikes" : bm.population.get_data("spikes").segments[0].spiketrains,
        "num_active" : num_active,
    }

    if steps_per_snapshot > 0:
        retval["snapshots_weight"] = snapshots_weight
        retval["snapshots_bias"] = snapshots_bias

    return retval

def train_rbm_ppcd(
        training_data=None, # binary nd.array
                            # shape (n_labels, n_samples, n_visible)
        num_steps=15000,
        num_hidden=None,
        eta_func=lambda t: 10./(100.+t),
        bm_init_kwargs={"data": {}, "model": {}}, # dict of dict
        bm_settings={"data": {}, "model":  {}},    # dict
        sim_setup_kwargs={"spike_precision" : "on_grid", "time_step": 0.1},
        seed=42,
        steps_per_snapshot=0,
        time_step=10.,
        init_weights_theo=None,
        init_biases_theo=None,
    ):
    """
        Parallel PCD training

        If steps_per_snapshot > 0, a weight update is taken from the network
        after every `steps_per_snapshot` steps.
    """
    assert len(training_data.shape) == 3

    np.random.seed(seed)

    num_labels, num_samples, num_visible = training_data.shape

    final_bm_settings = {
            "data" : {
                "current_imprint" : 10.,
                "time_imprint" : time_step,
                "time_sim_step" : time_step,
                "time_recon_step" : time_step,
                "time_wipe" : 0.,
            },
            "model" : {
                "time_wipe" : 0.,
                "time_sim_step" : time_step,
                "time_recon_step" : time_step,
            },
        }

    for k, v in final_bm_settings.iteritems():
        v.update(bm_settings[k])

    if steps_per_snapshot > 0:
        num_snapshots = num_steps / steps_per_snapshot + 1
        if num_steps % steps_per_snapshot != 0:
            num_snapshots += 1

        snapshots_weight = np.zeros((num_snapshots, num_visible, num_hidden))
        snapshots_bias = np.zeros((num_snapshots, num_visible + num_hidden))

    bm = {"data" : None, "model" : None}

    for k in bm.iterkeys():
        bm_init_kwargs[k]["num_units_per_layer"] = [num_visible, num_hidden]


    bm["data"] = RapidRBMImprintCurrent(**bm_init_kwargs["data"])
    bm["model"] = RapidRBMImprintCurrent(**bm_init_kwargs["model"])

    for s in it.chain(bm["data"].samplers, bm["model"].samplers):
        s.silent = True

    bm["data"].create_no_return(sim_setup_kwargs=sim_setup_kwargs)
    bm["model"].create_no_return()

    bm["data"].auto_sync_biases = False
    bm["model"].auto_sync_biases = False

    for k,v in final_bm_settings.iteritems():
        for ki, vi in v.iteritems():
            setattr(bm[k], ki, vi)

    if init_biases_theo is not None:
        for v in bm.itervalues():
            v.biases_theo = init_biases_theo

    if init_weights_theo is not None:
        for v in bm.itervalues():
            v.weights_theo = init_weights_theo

    sample_ids = np.random.randint(num_samples, size=num_steps)

    t_start = time.time()

    # set a random binary state in the beginning
    binary_state = np.ones(bm["data"].num_samplers) + 1 # per default set nothing
    visible_state_model = np.random.randint(2, size=num_visible)

    labels = np.random.randint(num_labels, size=num_steps)

    # i_l = 0 # which label
    i_s = 0 # which snapshot
    for i_step, i_samples in enumerate(sample_ids):

        i_l = labels[i_step]

        try:
            if i_step % int(num_steps / 20) == 0:
                log.info("Run #{}. [SimTime: {}ms] ETA: {}".format(i_step,
                    bm["data"].time_current,
                    utils.get_eta_str(t_start, i_step, num_steps)))
        except ZeroDivisionError:
            pass


        visible_state_data = training_data[i_l, i_samples]
        binary_state = np.ones(bm["data"].num_samplers) + 1 # per default set nothing
        # binary_state[:num_visible] = visible_state_model
        bm["data"].binary_state = binary_state

        bm["data"].run()
        bm["model"].process_run()
        hidden_state_data = bm["data"].binary_state[num_visible:].copy()

        visible_state_model = bm["model"].binary_state[:num_visible].copy()
        hidden_state_model = bm["model"].binary_state[num_visible:].copy()

        update_factors = bm["data"].update_factors
        update_factors[:, 0] = np.sqrt(eta_func(i_step))

        update_factors[:num_visible, 1] = visible_state_data
        update_factors[:num_visible, 2] = visible_state_model

        update_factors[num_visible:, 1] = hidden_state_data
        update_factors[num_visible:, 2] = hidden_state_model

        bias_update = eta_func(i_step)\
            * np.r_[visible_state_data - visible_state_model,
                hidden_state_data - hidden_state_model]

        for v in bm.itervalues():
            v.biases_theo = v.biases_theo + bias_update
            v.update_factors = update_factors
            v.queue_update()

        if steps_per_snapshot > 0 and i_step % steps_per_snapshot == 0:
            snapshots_weight[i_s] = bm["data"].weights_theo[0][0, :, :]
            snapshots_bias[i_s] = bm["data"].biases_theo
            i_s += 1

        # i_l = (i_l+1) % num_labels

    # take last snapshot if needed
    if steps_per_snapshot > 0 and i_s < num_snapshots:
        snapshots_weight[i_s] = bm["data"].weights_theo[0][0, :, :]

    bm["data"].binary_state = np.ones(bm["data"].num_samplers, dtype=int)
    bm["data"].run()

    log.info(
        bm["data"]._sim.nest.GetStatus(bm["data"].population.all_cells.tolist(),
        "updates_queued"))

    log.info(pf(bm["data"].weights_theo))
    # TODO: Error when querying bio weights
    log.info(pf(bm["data"].biases_theo))
    retval = {
        "final_weights" : bm["data"].weights_theo,
        "final_biases" : bm["data"].biases_theo,
        # "spikes" : bm["data"].population.get_data("spikes").segments[0].spiketrains,
    }

    if steps_per_snapshot > 0:
        retval["snapshots_weight"] = snapshots_weight
        retval["snapshots_bias"] = snapshots_bias

    return retval

def train_rbm_ppcd_minibatch(
        training_data=None, # binary nd.array
                            # shape (n_labels, n_samples, n_visible)
        num_steps=15000,
        num_hidden=None,
        eta_func=lambda t: 10./(100.+t),
        bm_init_kwargs={"data": {}, "model": {}}, # dict of dict
        bm_settings={"data": {}, "model":  {}},    # dict
        sim_setup_kwargs={"spike_precision" : "on_grid", "time_step": 0.1},
        seed=42,
        steps_per_snapshot=0,
        time_step=10.,
        init_weights_theo=None,
        init_biases_theo=None,
    ):
    """
        Parallel PCD training

        If steps_per_snapshot > 0, a weight update is taken from the network
        after every `steps_per_snapshot` steps.
    """
    assert len(training_data.shape) == 3

    np.random.seed(seed)

    num_labels, num_samples, num_visible = training_data.shape

    final_bm_settings = {
            "data" : {
                "current_imprint" : 10.,
                "time_imprint" : time_step,
                "time_sim_step" : time_step,
                "time_recon_step" : time_step,
                "time_wipe" : 0.,
            },
            "model" : {
                "time_wipe" : 0.,
                "time_sim_step" : time_step,
                "time_recon_step" : time_step,
            },
        }

    for k, v in final_bm_settings.iteritems():
        v.update(bm_settings[k])

    if steps_per_snapshot > 0:
        num_snapshots = num_steps / steps_per_snapshot + 1
        if num_steps % steps_per_snapshot != 0:
            num_snapshots += 1

        snapshots_weight = np.zeros((num_snapshots, num_visible, num_hidden))
        snapshots_bias = np.zeros((num_snapshots, num_visible + num_hidden))

    all_bms = [{"data" : None, "model" : None} for i in xrange(num_labels)]

    import pyNN.nest as sim
    sim.setup(**sim_setup_kwargs)

    for bm in all_bms:
        for k in bm.iterkeys():
            bm_init_kwargs[k]["num_units_per_layer"] = [num_visible, num_hidden]

        bm["data"] = RapidRBMImprintCurrent(**bm_init_kwargs["data"])
        bm["model"] = RapidRBMImprintCurrent(**bm_init_kwargs["model"])

        for s in it.chain(bm["data"].samplers, bm["model"].samplers):
            s.silent = True

        bm["data"].create_no_return()
        bm["model"].create_no_return()

        bm["data"].auto_sync_biases = False
        bm["model"].auto_sync_biases = False

        for k,v in final_bm_settings.iteritems():
            for ki, vi in v.iteritems():
                setattr(bm[k], ki, vi)

        if init_biases_theo is not None:
            for v in bm.itervalues():
                v.biases_theo = init_biases_theo

        if init_weights_theo is not None:
            for v in bm.itervalues():
                v.weights_theo = init_weights_theo

    sample_ids = np.random.randint(num_samples, size=(num_steps, num_labels))

    t_start = time.time()

    # set a random binary state in the beginning
    binary_state = np.ones(bm["data"].num_samplers) + 1 # per default set nothing
    visible_state_model = np.random.randint(2, size=num_visible)

    # labels = np.random.randint(num_labels, size=num_steps)

    # i_l = 0 # which label
    i_s = 0 # which snapshot
    for i_step, i_samples in enumerate(sample_ids):

        # i_l = labels[i_step]

        try:
            if i_step % int(num_steps / 20) == 0:
                log.info("Run #{}. [SimTime: {}ms] ETA: {}".format(i_step,
                    bm["data"].time_current,
                    utils.get_eta_str(t_start, i_step, num_steps)))

                for i_l, bm in enumerate(all_bms):
                    for k,v in bm.iteritems():
                        v._sim.nest.SetStatus(v._nest_connections,
                                "manual_weight_update", True)

                        # updates_queued = v._sim.nest.GetStatus(
                                # v.population.all_cells.tolist(),
                                # "updates_queued")
                        # log.info("[Updates queued {}:{}] {}".format(
                            # i_l, k, updates_queued))
        except ZeroDivisionError:
            pass

        for i_l, bm in enumerate(all_bms):
            visible_state_data = training_data[i_l, i_samples[i_l]]
            binary_state = np.ones(bm["data"].num_samplers) + 1 # per default set nothing
            binary_state[:num_visible] = visible_state_data
            bm["data"].binary_state = binary_state

            if i_step == 0:
                # initialize model
                bm["model"].binary_state = binary_state

            bm["data"].prepare_run()
            bm["model"].prepare_run()

        sim.run(time_step)

        eta_factor = eta_func(i_step) / num_labels

        for i_l, bm in enumerate(all_bms):
            bm["data"].process_run()
            bm["model"].process_run()

            visible_state_data = training_data[i_l, i_samples[i_l]]
            hidden_state_data = bm["data"].binary_state[num_visible:].copy()

            visible_state_model = bm["model"].binary_state[:num_visible].copy()
            hidden_state_model = bm["model"].binary_state[num_visible:].copy()

            update_factors = bm["data"].update_factors.copy()
            update_factors[:, 0] = eta_factor

            update_factors[:num_visible, 1] = visible_state_data
            update_factors[:num_visible, 2] = visible_state_model

            update_factors[num_visible:, 1] = hidden_state_data
            update_factors[num_visible:, 2] = hidden_state_model

            bias_update = eta_factor\
                * np.r_[visible_state_data - visible_state_model,
                    hidden_state_data - hidden_state_model]

            for other_bm in all_bms:
                for v in other_bm.itervalues():
                    v.biases_theo = v.biases_theo + bias_update
                    v.update_factors = update_factors.copy()
                    v.queue_update()

        if steps_per_snapshot > 0 and i_step % steps_per_snapshot == 0:
            snapshots_weight[i_s] = all_bms[0]["data"].weights_theo[0][0, :, :]
            snapshots_bias[i_s] = all_bms[0]["data"].biases_theo
            i_s += 1

        # i_l = (i_l+1) % num_labels

    # take last snapshot if needed
    if steps_per_snapshot > 0 and i_s < num_snapshots:
        snapshots_weight[i_s] = all_bms[0]["data"].weights_theo[0][0, :, :]

    # all_bms[0]["data"].binary_state = np.ones(all_bms[0]["data"].num_samplers,
        # dtype=int)

    for i_l, bm in enumerate(all_bms):
        for k,v in bm.iteritems():
            v._sim.nest.SetStatus(v._nest_connections,
                    "manual_weight_update", True)

    log.info(pf(bm["model"].weights_theo))
    # TODO: Error when querying bio weights
    log.info(pf(bm["model"].biases_theo))
    retval = {
        "final_weights" : all_bms[0]["model"].weights_theo,
        "final_biases" : all_bms[0]["model"].biases_theo,
        "all_biases" : [{
            "data" : bm["data"].biases_theo,
            "model" : bm["model"].biases_theo,
        } for bm in all_bms],
         "all_weights" : [{
            "data" : bm["data"].weights_theo,
            "model" : bm["model"].weights_theo,
        } for bm in all_bms],
    }

    if steps_per_snapshot > 0:
        retval["snapshots_weight"] = snapshots_weight
        retval["snapshots_bias"] = snapshots_bias

    return retval

def train_rbm_pcd(
        training_data=None, # binary nd.array
                            # shape (n_labels, n_samples, n_visible)
        num_steps=15000,
        num_hidden=None,
        eta_func=lambda t: 10./(100.+t),
        bm_init_kwargs=None, # dict
        bm_settings=None,    # dict
        sim_setup_kwargs={"spike_precision" : "on_grid", "time_step": 0.1},
        seed=42,
        bm_type=RapidRBMImprintCurrent,
        steps_per_snapshot=0,
    ):
    """
        If steps_per_snapshot > 0, a weight update is taken from the network
        after every `steps_per_snapshot` steps.
    """
    assert len(training_data.shape) == 3

    np.random.seed(seed)

    num_labels, num_samples, num_visible = training_data.shape

    final_bm_settings = {
            # "current_wipe" : 10.,
            # "current_imprint" : 10.,
            # "current_force_spike" : 10.,

            # "time_wipe" : 20.,
            # "time_imprint" : 10.,
            # "time_force_spike" : 10.,
            "time_sim_step" : 10.,
            "time_recon_step" : 10.,
        }

    bm_init_kwargs["num_units_per_layer"] = [num_visible, num_hidden]

    final_bm_settings.update(bm_settings)

    time_recon_step = final_bm_settings["time_recon_step"]

    if steps_per_snapshot > 0:
        num_snapshots = num_steps / steps_per_snapshot + 1
        if num_steps % steps_per_snapshot != 0:
            num_snapshots += 1

        snapshots_weight = np.zeros((num_snapshots, num_visible, num_hidden))
        snapshots_bias = np.zeros((num_snapshots, num_visible + num_hidden))

    def init_net():
        bm = bm_type(**bm_init_kwargs)
        bm.create_no_return(sim_setup_kwargs=sim_setup_kwargs)

        for s in bm.samplers:
            s.silent = True

        bm.auto_sync_biases = False

        for k,v in final_bm_settings.iteritems():
            setattr(bm, k, v)

        # bm.update_weights()

        return bm

    bm = init_net()

    sample_ids = np.random.randint(num_samples, size=num_steps)

    t_start = time.time()

    # set a random binary state in the beginning
    binary_state = np.ones(bm.num_samplers) + 1 # per default set nothing
    visible_state_model = np.random.randint(2, size=num_visible)

    labels = np.random.randint(num_labels, size=num_steps)

    # i_l = 0 # which label
    i_s = 0 # which snapshot
    for i_step, i_samples in enumerate(sample_ids):

        i_l = labels[i_step]

        try:
            if i_step % int(num_steps / 20) == 0:
                log.info("Run #{}. [SimTime: {}ms] ETA: {}".format(i_step,
                    bm.time_current,
                    utils.get_eta_str(t_start, i_step, num_steps)))
        except ZeroDivisionError:
            pass


        visible_state_data = training_data[i_l, i_samples]
        binary_state = np.ones(bm.num_samplers) + 1 # per default set nothing
        binary_state[:num_visible] = visible_state_model
        bm.binary_state = binary_state

        bm.run()
        hidden_state_data = bm.binary_state[num_visible:].copy()
        # log.info("Hidden state: " + pf(hidden_state))
        # log.info("Last spiketimes: " + pf(bm.last_spiketimes))
        binary_state = np.ones(bm.num_samplers) + 1 # per default set nothing
        binary_state[:num_visible] = visible_state_model

        bm.binary_state = binary_state
        bm.run()

        hidden_state_model = bm.binary_state[num_visible:].copy()

        bm.continue_run(time_recon_step)
        visible_state_model = bm.binary_state[:num_visible].copy()
        # bm.continue_run(time_recon_step)
        # log.info("Visible state: " + pf(visible_state))
        # log.info("Last spiketimes: " + pf(bm.last_spiketimes))

        if bm.time_current > 0.95 * NEST_MAX_TIME:
            weights_theo = bm.weights_theo
            bm.kill()
            bm = init_net()
            bm.weights_theo = weights_theo
            # bm.update_weights()

        update_factors = bm.update_factors
        update_factors[:, 0] = np.sqrt(eta_func(i_step))

        update_factors[:num_visible, 1] = visible_state_data
        update_factors[:num_visible, 2] = visible_state_model

        update_factors[num_visible:, 1] = hidden_state_data
        update_factors[num_visible:, 2] = hidden_state_model

        bias_update = eta_func(i_step)\
            * np.r_[visible_state_data - visible_state_model,
                hidden_state_data - hidden_state_model]

        bm.biases_theo = bm.biases_theo + bias_update

        bm.update_factors = update_factors

        bm.queue_update()

        if steps_per_snapshot > 0 and i_step % steps_per_snapshot == 0:
            snapshots_weight[i_s] = bm.weights_theo[0][0, :, :]
            snapshots_bias[i_s] = bm.biases_theo
            i_s += 1

        # i_l = (i_l+1) % num_labels

    # take last snapshot if needed
    if steps_per_snapshot > 0 and i_s < num_snapshots:
        snapshots_weight[i_s] = bm.weights_theo[0][0, :, :]

    bm.binary_state = np.ones(bm.num_samplers, dtype=int)
    bm.run()

    log.info(pf(bm.weights_theo))
    # TODO: Error when querying bio weights
    log.info(pf(bm.biases_theo))
    retval = {
        "final_weights" : bm.weights_theo,
        "final_biases" : bm.biases_theo,
    }

    if steps_per_snapshot > 0:
        retval["snapshots_weight"] = snapshots_weight
        retval["snapshots_bias"] = snapshots_bias

    return retval

def train_rbm_pcd_minibatch(
        training_data=None, # binary nd.array
                            # shape (n_labels, n_samples, n_visible)
        num_steps=15000,
        num_hidden=None,
        eta_func=lambda t: 10./(100.+t),
        bm_init_kwargs=None, # dict
        bm_settings=None,    # dict
        sim_setup_kwargs={"spike_precision" : "on_grid", "time_step": 0.1},
        seed=42,
        bm_type=RapidRBMImprintCurrent,
        steps_per_snapshot=0,
    ):
    """
        If steps_per_snapshot > 0, a weight update is taken from the network
        after every `steps_per_snapshot` steps.

        Use a minibatch of one per training label.
    """
    assert len(training_data.shape) == 3

    np.random.seed(seed)

    num_labels, num_samples, num_visible = training_data.shape

    final_bm_settings = {
            "current_wipe" : 10.,
            "current_imprint" : 10.,

            "time_wipe" : 20.,
            "time_imprint" : 10.,
            "time_sim_step" : 10.,
            "time_recon_step" : 10.,
        }

    bm_init_kwargs["num_units_per_layer"] = [num_visible, num_hidden]

    final_bm_settings.update(bm_settings)

    time_sim_step = final_bm_settings["time_sim_step"]
    time_recon_step = final_bm_settings["time_recon_step"]

    if steps_per_snapshot > 0:
        num_snapshots = num_steps / steps_per_snapshot + 1
        if num_steps % steps_per_snapshot != 0:
            num_snapshots += 1

        snapshots_weight = np.zeros((num_snapshots, num_visible, num_hidden))
        snapshots_bias = np.zeros((num_snapshots, num_visible + num_hidden))

    def init_net():
        bms = [bm_type(**bm_init_kwargs) for i in xrange(num_labels)]

        for bm in bms:
            bm.create_no_return(sim_setup_kwargs=sim_setup_kwargs)

            for s in bm.samplers:
                s.silent = True

            # log.info(bm._sim.nest.GetStatus(bm.population.all_cells.tolist(),
                # "num_connections"))

            bm.auto_sync_biases = False

            for k,v in final_bm_settings.iteritems():
                setattr(bm, k, v)

            # bm.update_weights()

        return bms

    bms = init_net()

    sample_ids = np.random.randint(num_samples, size=(num_steps, num_labels))

    t_start = time.time()

    # set a random binary state in the beginning
    binary_state = np.ones((len(bms), bms[0].num_samplers)) + 1
    # per default set nothing
    visible_state_data = np.zeros((len(bms), num_visible), dtype=int)
    visible_state_model = np.random.randint(2, size=(len(bms), num_visible))

    hidden_state_data = np.zeros((len(bms), num_hidden), dtype=int)
    hidden_state_model = np.zeros((len(bms), num_hidden), dtype=int)

    # i_l = 0 # which label
    i_s = 0 # which snapshot
    for i_step, i_samples in enumerate(sample_ids):

        try:
            if i_step % int(num_steps / 20) == 0:
                log.info("Run #{}. [SimTime: {}ms] ETA: {}".format(i_step,
                    bms[0].time_current,
                    utils.get_eta_str(t_start, i_step, num_steps)))
                # for bm in bms:
                    # print bm.last_spiketimes
        except ZeroDivisionError:
            pass

        for i_l, bm in enumerate(bms):
            visible_state_data[i_l] = training_data[i_l, i_samples[i_l]]
            binary_state = np.ones(bm.num_samplers) + 1 # per default set nothing
            binary_state[:num_visible] = visible_state_data[i_l]
            bm.binary_state = binary_state

            # this is the same for all bms
            time_till = bm.prepare_run()

        # log.info("Time till: {}".format(time_till))

        bms[0]._sim.run_until(time_till)

        for i_l, bm in enumerate(bms):
            bm.process_run()

            hidden_state_data[i_l] = bm.binary_state[num_visible:].copy()

            binary_state = np.ones(bm.num_samplers) + 1 # per default set nothing
            binary_state[:num_visible] = visible_state_model[i_l]
            bm.binary_state = binary_state

            time_till = bm.prepare_run()
            # log.info("Hidden state: " + pf(hidden_state))
            # log.info("Last spiketimes: " + pf(bm.last_spiketimes))

        bms[0]._sim.run_until(time_till)
        # bm.continue_run(time_recon_step)

        for i_l, bm in enumerate(bms):
            bm.process_run()
            hidden_state_model[i_l] = bm.binary_state[num_visible:].copy()

        # bms[0]._sim.run_until(time_till)
        bm.continue_run(time_recon_step)

        for i_l, bm in enumerate(bms):
            bm.process_run()
            visible_state_model[i_l] = bm.binary_state[:num_visible].copy()

        # for i_l, bm in enumerate(bms):
            # bm.process_run()
            # log.info("Visible state: " + pf(visible_state))
            # log.info("Last spiketimes: " + pf(bm.last_spiketimes))

        # if bm.time_current > 0.95 * NEST_MAX_TIME:
            # weights_theo = bm.weights_theo
            # bm.kill()
            # bm = init_net()
            # bm.weights_theo = weights_theo
            # # bm.update_weights()

        update_factors = bms[0].update_factors
        update_factors[:, 0] = np.sqrt(eta_func(i_step))

        visible_data_mean = visible_state_data.mean(axis=0) 
        visible_model_mean = visible_state_model.mean(axis=0)
        hidden_data_mean = hidden_state_data.mean(axis=0)
        hidden_model_mean = hidden_state_model.mean(axis=0)

        update_factors[:num_visible, 1] = visible_data_mean
        update_factors[:num_visible, 2] = visible_model_mean

        update_factors[num_visible:, 1] = hidden_data_mean
        update_factors[num_visible:, 2] = hidden_model_mean

        bias_update = eta_func(i_step)\
            * np.r_[visible_data_mean - visible_model_mean,
                hidden_data_mean - hidden_model_mean]

        for i_l, bm in enumerate(bms):
            bm.biases_theo = bm.biases_theo + bias_update
            bm.update_factors = update_factors
            bm.queue_update()

        if steps_per_snapshot > 0 and i_step % steps_per_snapshot == 0:
            snapshots_weight[i_s] = bm.weights_theo[0][0, :, :]
            snapshots_bias[i_s] = bm.biases_theo
            i_s += 1

        # i_l = (i_l+1) % num_labels

    for bm in bms:
        bm._sim.nest.SetStatus(bm.population.all_cells.tolist(), "V_m", 0.)

    bms[0].continue_run(1.)
    for bm in bms:
        bm.process_run()

    # take last snapshot if needed
    if steps_per_snapshot > 0 and i_s < num_snapshots:
        snapshots_weight[i_s] = bm.weights_theo[0][0, :, :]


    log.info(pf(bm.weights_theo))
    log.info(pf(bm.biases_theo))
    retval = {
            "final_weights" : bm.weights_theo,
            "final_biases" : bm.biases_theo,
        }

    if steps_per_snapshot > 0:
        retval["snapshots_weight"] = snapshots_weight
        retval["snapshots_bias"] = snapshots_bias

    return retval

