#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import pylab as p
import h5py
import functools as ft

from . import utils
from .logcfg import log

# subgroups are the primary keys for the calibration-rows in the database
# then they have two datasets: v_rest and p_on
data_storage = None # set from db

def create_dataset_compressed(h5grp, *args, **kwargs):
    kwargs.setdefault("compression", "gzip")
    kwargs.setdefault("compression_opts", 9)
    dataset = h5grp.create_dataset(*args, **kwargs)
    h5grp.file.flush() # sync file to avoid corruption
    return dataset


def ensure_group_exists(h5grp, name):
    log.debug("Looking for {} in {}".format(name, h5grp.name))
    if name not in h5grp:
        h5grp.create_group(name)
    return h5grp[name]


def generate_setter(field):
    def setter(self, array):
        h5grp = self.get_storage_group()
        if field in h5grp:
            del h5grp[field]
        create_dataset_compressed(h5grp, name=field, data=array)

    return setter


def generate_getter(field):
    def getter(self):
        h5grp = self.get_storage_group()
        if field in h5grp:
            return h5grp[field]
        else:
            return None

    return getter


def setup_storage_fields(model):
    """
        Decorator to enable storage fields for models.

        Storage fields are defined by `storage_fields` in the original model.
        They can be set and got but not used in any SQL query as they are not
        present in the dataset.

        The model has to be saved to the database for storage fields to work.
    """
    if hasattr(model, "storage_fields"):
        storage_fields = getattr(model, "storage_fields")
        delattr(model, "storage_fields")
    else:
        storage_fields = []

    def get_storage_group(self):
        assert self.get_id() is not None, "Model was not saved in database!"
        return ensure_group_exists(ensure_group_exists(data_storage,
            self.__class__.__name__), str(self.get_id()))

    setattr(model, "get_storage_group", get_storage_group)

    for field in storage_fields:
        setter = generate_setter(field)
        getter = generate_getter(field)
        setattr(model, field, property(getter, setter))

    return model


def plot_function(plotname):
    """
        Wraps a function so that it creates a figure and axes when it is not
        supplied with the kwargs fig/ax.

        Note that fig/ax have to kwargs and not regular args.

        If no figure was supplied, the figure will be shown.
        If `save` was supplied, the figure will be saved instead as `plotname`
        instead.
    """
    def decorator(orig):
        def wrapped(*args, **kwargs):
            show = kwargs.get("show", True)
            save = False

            if "show" in kwargs:
                del kwargs["show"]
            if kwargs.get("fig", None) is None:
                kwargs["fig"] = p.figure()
            else:
                # don't show when user supplies a figure
                show = False

            if kwargs.get("save", False):
                show = False
                save = True
            if "save" in kwargs:
                del kwargs["save"]

            if "plotname" in kwargs:
                local_plotname = kwargs["plotname"]
                del kwargs["plotname"]
            else:
                local_plotname = plotname

            if kwargs.get("ax", None) is None:
                kwargs["ax"] = kwargs["fig"].add_subplot(111)

            log.info("Plotting {}..".format(local_plotname))

            returnval = orig(*args, **kwargs)

            if show:
                kwargs["fig"].show()
            if save:
                kwargs["fig"].savefig(local_plotname)

            return returnval

        return wrapped

    return decorator


def log_exception(f):
    @ft.wraps(f)
    def wrapped(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception, e:
            import traceback as tb, sys
            log.error(tb.format_tb(sys.exc_info()[2])[0])
            log.error(str(e))
            raise e

    return wrapped

