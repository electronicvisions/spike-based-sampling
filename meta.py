#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import pylab as p
import h5py
import functools as ft
import peewee as pw
import logging

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

def delete_dataset(h5grp, name):
    h5file = h5grp.file
    del h5grp[name]
    h5file.flush()


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
        setattr(self, field + "_sha1", utils.get_sha1(array))

    return setter


def generate_getter(field):
    def getter(self):
        h5grp = self.get_storage_group()
        if field in h5grp:
            return np.array(h5grp[field])
        else:
            return None

    return getter


class StorageFields(pw.BaseModel):
    """
        Meta class to enable storage fields for models.

        Storage fields are defined by `storage_fields` in the original model.
        They can be set and got but not used in any SQL query as they are not
        present in the dataset.

        The model has to be saved to the database for storage fields to work.

        For each storage field, there will be `name`_sha1 char field added so
        that there is an indication when storage contents change (e.g. this is
        used to distinguish sources with different spike times).
    """
    def __new__(mcs, name, bases, dcts):
        storage_fields = dcts.get("_storage_fields", tuple())

        def get_storage_group(self):
            assert self.get_id() is not None, "Model was not saved in database!"
            return ensure_group_exists(ensure_group_exists(data_storage,
                self.__class__.__name__), str(self.get_id()))

        dcts["get_storage_group"] = get_storage_group

        for field in storage_fields:
            setter = generate_setter(field)
            getter = generate_getter(field)
            dcts[field] = property(getter, setter)
            dcts[field + "_sha1"] = pw.CharField(null=True, max_length=40)

        cls = super(StorageFields, mcs).__new__(mcs, name, bases, dcts)

        # we need to overwrite the delete_instance method
        def delete_instance(self):
            h5grp = self.get_storage_group()
            for sf in storage_fields:
                delete_dataset(h5grp, sf)

            return super(cls, self).delete_instance()

        setattr(cls, "delete_instance", delete_instance)
        return cls


class DependsOn(object):
    """
        Descriptor dealing with dependencies to other values.

        The function passed to the decorator should accept one argument (self)
        for computing nodes that update their value when their dependencies
        change and two arguments (self, value) for nodes that get set.
    """

    def __init__(self, *dependencies):
        if dependencies is None:
            dependencies = []

        # the names of variables we dependend on
        self._dependencies = dependencies
        # set of all variables that need to be updated if we are updated
        self._influences = []

        # dependencies will be propagated the first time a descriptor function
        # is accessed
        self._propagated_dependencies = False

        self._func = None
        self.value_name = "_unnamed"
        self.attr_name = "unamed"

    def __call__(self, func):
        self.attr_name = func.func_name
        self.value_name = "_" + self.attr_name
        self._func = func
        self.__doc__ = func.func_doc
        return self

    def __get__(self, instance, owner):
        log.debug("Getting {}.".format(self.attr_name))

        if self.needs_update(instance):
            if log.getEffectiveLevel() <= logging.DEBUG:
                log.debug("{} needs update.".format(self.attr_name))
            setattr(instance, self.value_name, self._func(instance))

        return getattr(instance, self.value_name)

    def __set__(self, instance, value):
        self._propagate_dependencies(type(instance))
        self.wipe(instance)
        log.debug("Setting {}.".format(self.attr_name))
        setattr(instance, self.value_name, self._func(instance, value))

    def _propagate_dependencies(self, klass):
        if self._propagated_dependencies:
            return
        log.debug("Propagating dependencies for {}.".format(self.attr_name))
        for dep in self._dependencies:
            klass.__dict__[dep]._influences.append(self)
        self._propagated_dependencies = True

    def wipe(self, instance):
        """
            Mark the values in instances as invalid/wipe them.
        """
        if self.needs_update(instance):
            # we have already wiped this instance
            return
        log.debug("Wiping {}.".format(self.attr_name))
        setattr(instance, self.value_name, None)
        for influence in self._influences:
            influence.wipe(instance)

    def needs_update(self, instance):
        return getattr(instance, self.value_name, None) is None


def HasDependencies(klass):
    """
        Decorator that is needed for dependency relations to be maintained.
    """
    for attr in vars(klass).itervalues():
        if isinstance(attr, DependsOn):
            attr._propagate_dependencies(klass)

    # add wipe function
    def wipe(self, name):
        descriptor = self.__class__.__dict__.get(name, None)

        assert descriptor is not None, "{} not found".format(name)
        assert isinstance(descriptor, DependsOn),\
                "{} is no dependency type".format(name)

        descriptor.wipe(self)

    setattr(klass, "wipe", wipe)

    return klass




def plot_function(plotname):
    """
        Wraps a function so that it creates a figure and axes when it is not
        supplied with the kwargs fig/ax.

        Note that fig/ax have to be kwargs and not regular args.

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

