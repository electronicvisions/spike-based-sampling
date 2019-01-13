#!/usr/bin/env python
# encoding: utf-8

from . import utils

from scipy import optimize as so


def fit_sigmoid(x, y, guess_p05, guess_alpha, p_min=0.0, p_max=1.0):
    """
        Fits the sigmoid to the x/y data samples.
        Takes only activity values in [p_min, p_max] into account
    """
    inds = (y > p_min) * (y < p_max)
    x = x[inds]
    y = y[inds]
    opt_vars, cov_vars = so.curve_fit(
            utils.sigmoid_trans, x, y,
            p0=[guess_p05, guess_alpha])

    x_p05, alpha = opt_vars

    return x_p05, alpha
