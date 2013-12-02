#!/usr/bin/env python
# encoding: utf-8

from . import utils

from scipy import optimize as so

def fit_sigmoid(self, x, y, guess_p05, guess_alpha):
    """
        Fits the sigmoid to the x/y data samples.
    """
    opt_vars, cov_vars = so.curve_fit(utils.simoid_norm, x, y,
            p0=[guess_p05, guess_alpha])

    x_p05, alpha = opt_vars

    return x_p05, alpha

