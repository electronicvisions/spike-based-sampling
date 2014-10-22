#!/usr/bin/env python2
# encoding: utf-8

# Special training methods
# Make use of some wrappers from SEMf

from SEMf import misc as m

from ..logcfg import log
from ..network import RapidRBMCurrentImprint

RapidRBMCurrentImprint = m.ClassInSubprocess(RapidRBMCurrentImprint)


def train_rbm(
        training_data=None, # binary nd.array
                            # shape (n_labels, n_samples, n_visible)
        num_steps=15000,
        bm_init_kwargs=None, # dict
        bm_settings=None,    # dict
    ):
    assert len(training_data.shape) == 3

    bm_default_settings = {
            "current_wipe" : 10.,
            "current_imprint" : 10.,
            "current_force_spike" : 10.,
        }





