#!/usr/bin/env python2
# encoding: utf-8

# Special training methods
# Make use of some wrappers from SEMf

from semf import misc as m


def train_rbm(
        training_data=None, # binary nd.array
                            # shape (n_labels, n_samples, n_visible)
        num_steps=15000,
        bm_init_kwargs=None, # dict
        bm_settings=None,    # dict
    ):

    bm_default_settings = {
            "wipe_current" : 10.,
            "imprint_current" : 10.,
        }





