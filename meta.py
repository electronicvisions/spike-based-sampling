#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import pylab as p
import h5py

from . import utils

def create_dataset_compressed(h5grp, *args, **kwargs):
    kwargs.setdefault("compression", "gzip")
    kwargs.setdefault("compression_opts", 9)
    dataset = h5grp.create_dataset(*args, **kwargs)
    h5grp.file.flush() # sync file to avoid corruption
    return dataset


def ensure_group_exists(h5grp, name):
    if name not in h5grp:
        h5grp.create_group(name)
    return h5grp[name]


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
            self.__class__.__name__), self.get_id())

    setattr(model, "get_storage_group", get_storage_group)

    for field in storage_fields:
        def setter(self, array):
            h5grp = self.get_storage_group()
            if field in h5grp:
                del h5grp[field]

            create_dataset_compressed(h5grp, name=field, data=array)

        def getter(self):
            h5grp = self.get_storage_group()
            if field in h5grp:
                return h5grp[field]
            else:
                return None

        setattr(model, field, property(getter, setter))

    return model


def plot_function(orig):
    """
        Wraps a function so that it creates a figure and axes when it is not
        supplied with the kwargs fig/ax.

        Note that fig/ax have to kwargs and not regular args.

        If no figure was supplied, the figure will be shown.
    """
    def wrapped(*args, **kwargs):
        show = kwargs.get("show", True)

        if "show" in kwargs:
            del kwargs["show"]
        if kwargs.get("fig", None) is None:
            kwargs["fig"] = p.figure()
        else:
            # don't show when user supplies a figure
            show = False

        if kwargs.get("ax", None) is None:
            kwargs["ax"] = kwargs["fig"].add_subplot(111)

        returnval = orig(*args, **kwargs)

        if show:
            kwargs["fig"].show()

        return returnval

    return wrapped

