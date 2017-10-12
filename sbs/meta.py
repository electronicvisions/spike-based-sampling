#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import functools as ft
#  import peewee as pw
import logging
import os.path as osp

from . import utils
from .logcfg import log

# subgroups are the primary keys for the calibration-rows in the database
# then they have two datasets: v_rest and p_on
data_storage_filename = None # set from db


_dependency_checked_token = "_depcheck_done_26bslash9"


class DependsOn(object):
    """
        Descriptor dealing with dependencies to other values.

        The function passed to the decorator should accept one argument (self)
        for computing nodes that update their value when their dependencies
        change and two arguments (self, value) for nodes that get set.

        NOTE: When this descriptor is used, the corresponding class needs to be
        decorated with the HasDependencies-decorator.
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
        if self.needs_update(instance):
            #  if log.getEffectiveLevel() <= logging.DEBUG:
                #  log.debug("{} needs update.".format(self.attr_name))
            setattr(instance, self.value_name, self._func(instance))

        return getattr(instance, self.value_name)

    def __set__(self, instance, value):
        # self._propagate_dependencies(type(instance))
        self.wipe(instance, force=True)
        setattr(instance, self.value_name, self._func(instance, value))

    def _propagate_dependencies(self, klass):
        if self._propagated_dependencies:
            return
        self._class = klass
        log.debug("Propagating dependencies for {}.".format(self.attr_name))
        for dep in self._dependencies:
            log.debug("{} <- {}".format(self.attr_name, dep))
            to_search = [klass]
            while len(to_search) > 0:
                k = to_search.pop(0)
                if dep in k.__dict__:
                    k.__dict__[dep]._influences.append(self)
                    break
                else:
                    to_search.extend(k.__bases__)
            else:
                log.error("Could not add dependency {} for class {}.".format(
                    dep.__name__, klass.__name__))

        self._propagated_dependencies = True

    def wipe(self, instance, force=False):
        """
            Mark the values in instances as invalid/wipe them.
        """
        if self.needs_update(instance) and not force:
            # we have already wiped this instance
            return
        #  log.debug("Wiping {}.".format(self.attr_name))
        setattr(instance, self.value_name, None)
        for influence in self._influences:
            influence.wipe(instance)

    def needs_update(self, instance):
        return isinstance(instance, self._class)\
                and getattr(instance, self.value_name, None) is None

def HasDependencies(klass):
    """
        Decorator that is needed for dependency relations to be maintained.
    """
    klass_vars = vars(klass)
    if _dependency_checked_token in klass_vars:
        return klass

    found_dependencies = False

    for attr in klass_vars.itervalues():
        if isinstance(attr, DependsOn):
            found_dependencies = True
            attr._propagate_dependencies(klass)

    # add wipe function
    def wipe(self, name):
        descriptor = self.__class__.__dict__.get(name, None)

        assert descriptor is not None, "{} not found".format(name)
        assert isinstance(descriptor, DependsOn),\
                "{} is no dependency type".format(name)

        descriptor.wipe(self)

    if found_dependencies:
        setattr(klass, "wipe", wipe)
        setattr(klass, _dependency_checked_token, True)

    # catch mixins (they can be modified in place)
    # note: only direct mixins are caught as long as each class in the hierachy
    # has at least one dependson object
    if found_dependencies:
        for b in klass.__bases__:
            HasDependencies(b)

    return klass


def plot_function(plotname, dpi=300):
    """
        Wraps a function so that it creates a figure and axes when it is not
        supplied with the kwargs fig/ax.

        Note that fig/ax have to be kwargs and not regular args.

        If no figure was supplied, the figure will be shown.
        If `save` was supplied, the figure will be saved instead as `plotname`
        instead.

        tight_layout=True can be set to force tight layout.
    """
    def decorator(orig):
        def wrapped(*args, **kwargs):
            show = kwargs.get("show", True)
            save = False

            if "show" in kwargs:
                del kwargs["show"]
            if kwargs.get("fig", None) is None:
                kwargs["fig"] = plt.figure()
            else:
                # don't show when user supplies a figure
                show = False

            if kwargs.get("save", False):
                show = False
                save = True
            if "save" in kwargs:
                del kwargs["save"]

            if kwargs.get("plotname", None) is not None:
                local_plotname = kwargs["plotname"]
                del kwargs["plotname"]
            else:
                local_plotname = plotname

            if "tight_layout" in kwargs:
                tight_layout = kwargs["tight_layout"]
                del kwargs["tight_layout"]
            else:
                tight_layout = False

            if kwargs.get("plotfolder", None) is not None:
                local_plotname = osp.join(kwargs["plotfolder"], local_plotname)
                del kwargs["plotfolder"]

            if kwargs.get("ax", None) is None:
                kwargs["ax"] = kwargs["fig"].add_subplot(111)

            log.info("Plotting {}..".format(local_plotname))

            returnval = orig(*args, **kwargs)

            if tight_layout:
                kwargs["fig"].tight_layout()

            if show:
                kwargs["fig"].show()
            if save:
                kwargs["fig"].savefig(local_plotname, dpi=dpi)
                plt.close(kwargs["fig"])

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

