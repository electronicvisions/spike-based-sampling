#!/usr/bin/env python2
# encoding: utf-8

# custom cell models for usage with PyNN.nest

from .logcfg import log



def patch_pynn():
    from pyNN.nest.standardmodels import cells
    import pyNN.nest

    class IF_cond_exp_cd(cells.IF_cond_exp):
        nest_name = {
                "off_grid" : "iaf_cond_exp",
                "on_grid" : "cd_iaf_cond_exp",
            }

    class IF_curr_exp_cd(cells.IF_curr_exp):
        nest_name = {
                "off_grid" : "iaf_psc_exp",
                "on_grid" : "cd_iaf_psc_exp",
            }

    # monkey patching so much fun, doo dai, doo dai…
    for ct in [IF_cond_exp_cd, IF_curr_exp_cd]:
        log.info("Monkey patching pyNN.nest to include {}…".format(ct.__name__))
        setattr(pyNN.nest, ct.__name__, ct)


