#!/usr/bin/env python
# encoding: utf-8

"""
    Defines all data models and db interactions.
"""

from .logcfg import log
from . import utils
from . import meta

from pprint import pformat as pf
import logging
import peewee as pw
import numpy as np
import h5py
import datetime
import shelve
import collections as c
import tempfile
tempfile.gettempdir()
import os
import os.path as osp
import shutil

# database for parameters
database = pw.SqliteDatabase(None)

# classes for datastorage
current_basename = None

def setup(basename="database"):
    basename = osp.abspath(osp.expanduser(osp.expandvars(basename)))
    # avoid redundant ".sql.sql" or ".h5.h5" file endings
    base, ext = osp.splitext(basename)
    if ext in [".h5", ".sql"]:
        basename = base

    db_name = "{}.sql".format(basename)
    ds_name = "{}.h5".format(basename)

    log.info("Setting up database: {}".format(db_name))
    if osp.isfile(ds_name):
        log.info("Backing up HDF5 file..")
        shutil.copyfile(ds_name, ds_name + ".backup")
    else:
        with h5py.File(ds_name, "a"):
            log.info("Creating HDF5 file: {}".format(ds_name))

    if not database.deferred:
        database.close()
    database.init(db_name)
    database.connect()

    log.info("Setting up storage for datasets: {}".format(ds_name))
    meta.data_storage_filename = ds_name

    for model in _merge_order:
        if not model.table_exists():
            log.info("Creating table: {}".format(model.__name__))
            model.create_table()
    global current_basename
    current_basename = basename


def filter_incomplete_calibrations(query):
    return query.where(
            (Calibration.alpha >> None)\
            | (Calibration.alpha_theo >> None) \
            | (Calibration.v_p05 >> None) \
            | (Calibration.mean >> None )\
            | (Calibration.std >> None )\
            | (Calibration.g_tot >> None)
        )


def purge_incomplete_calibrations():
    """
        Calibration objects get written to the database prior to calibrating 
        because the id needs to be known for the samples to be written to 

        NOTE: Currently this method SHOULD NOT be used in a multi-user
        environment!
    """

    dq = filter_incomplete_calibrations(Calibration.select())

    to_delete = dq.execute()
    # we delete them by hand so that corresponding storage gets deleted as well
    # for larger applications this would be very inefficient
    num_deleted = 0
    for instance in to_delete:
        instance.delete_instance()
        num_deleted += 1

    log.info("Purged {} partial calibration{}..".format(num_deleted,
        "s" if num_deleted != 1 else ""))


def get_incomplete_calibration_ids():
    """
        List of ids of incomplete calibrations.
    """
    return list(filter_incomplete_calibrations(
        Calibration.select(Calibration.id)))



# NOTE: Only ever delete instances with `delete_instance` instead of a delete
# query.
class BaseModel(pw.Model):
    """
        The base model that sets the database.
    """
    def get_non_null_fields(self):
        """
            Gets all fields that 
            a) Are not None/null
            b) Are not simply sha1 sums.
            c) Are not the `date` attribute.
        """
        filter_func = lambda k: not k.endswith("_sha1") and k != "date"\
                and isinstance(getattr(self.__class__, k, None), pw.Field)
        field_names = filter(filter_func, dir(self))\
                + getattr(self.__class__, "_storage_fields")
        return {k: getattr(self, k) for k in field_names\
                if getattr(self, k) is not None}

    class Meta(object):
        database = database


##################################
# Actual models used in database #
##################################

class NeuronParameters(BaseModel):
    pynn_model = pw.CharField(max_length=48) # name of the pyNN model

    cm         = pw.DoubleField() # nF  Capacity of the membrane
    tau_m      = pw.DoubleField() # ms  Membrane time constant
    tau_refrac = pw.DoubleField() # ms  Duration of refractory period
    tau_syn_E  = pw.DoubleField() # ms  Decay time of excitatory synaptic curr
    tau_syn_I  = pw.DoubleField() # ms  Decay time of inhibitory synaptic curr
    e_rev_E    = pw.DoubleField(null=True) # mV  Reversal potential for exc inpt
    e_rev_I    = pw.DoubleField(null=True) # mV  Reversal potential for inh inpt
    i_offset   = pw.DoubleField() # nA  Offset current
    v_rest     = pw.DoubleField() # mV  Rest potential
    v_reset    = pw.DoubleField() # mV  Reset potential after a spike
    v_thresh   = pw.DoubleField() # mV  Spike threshold

    # for book keeping
    date = pw.DateTimeField(default=datetime.datetime.now)

    @property
    def g_l(self):
        "Leak conductance in µS"
        return self.cm / self.tau_m

    def get_pynn_parameters(self):
        """
            Returns a dictionary with all pyNN parameters to be passed to the
            model.
        """
        ignored_fields = ["pynn_model", "id", "date"]
        params = {k: v for k,v in self._data.iteritems()\
                if v is not None and k not in ignored_fields}
        if log.getEffectiveLevel() <= logging.DEBUG:
            log.debug("PyNN parameters: {}".format(pf(params)))
        return params

    class Meta:
        order_by = ("-date",)
        indexes = (
                ((
                    'pynn_model',
                    'cm',
                    'tau_m',
                    'tau_refrac',
                    'tau_syn_E',
                    'tau_syn_I',
                    'e_rev_E',
                    'e_rev_I',
                    'i_offset',
                    'v_rest',
                    'v_reset',
                    'v_thresh',
                ), True),
            )


class Calibration(BaseModel):
    """
        Represents a calibration result.
    """
    __metaclass__ = meta.StorageFields

    # configuration
    duration = pw.DoubleField(default=1e6)
    num_samples = pw.IntegerField(default=100)
    std_range = pw.DoubleField(default=4.)
    burn_in_time = pw.DoubleField(default=200.)
    dt = pw.DoubleField(default=0.01)

    # We do not check if simulators match when loading (because it should not
    # matter). It is only kept as information.
    simulator = pw.CharField(default="pynn.nest", max_length=48)

    used_parameters = pw.ForeignKeyField(NeuronParameters,
            related_name="calibrations", index=True, cascade=True)

    # results (may be null because before calibration only the arguments above
    # are known)
    alpha = pw.DoubleField(null=True)
    alpha_theo = pw.DoubleField(null=True) # theoretical estimate
    v_p05 = pw.DoubleField(null=True)
    mean = pw.DoubleField(null=True)
    std = pw.DoubleField(null=True)
    g_tot = pw.DoubleField(null=True)

    # for book keeping
    date = pw.DateTimeField(default=datetime.datetime.now)

    _storage_fields = ["samples_v_rest", "samples_p_on"]

    def link_sources(self, sources):
        """
            Note: SourceCFGs have to be present in the database already!
                  Also, they have to be all linked at the same time (but do not
                  have to be unique).
        """
        log.debug("Linking {} sources.".format(len(sources)))
        assert(not any((src.get_id() is None for src in sources)))
        # delete any sources previously linked to this node
        # node we do allow sources to be specified again

        dq = SourceCFGInCalibration.select().where(
                SourceCFGInCalibration.calibration==self)

        num_deleted = dq.execute()

        log.debug("Deleted {} previoiusly linked sources.".format(num_deleted))

        for src in sources:
            SourceCFGInCalibration.create(source=src, calibration=self)

    @property
    def is_complete(self):
        return self.alpha is not None\
            and self.v_p05 is not None

    class Meta:
        order_by = ("-date",)


class SourceCFG(BaseModel):
    __metaclass__ = meta.StorageFields
    # if rate is None we have spike times specified in the storage!
    rate = pw.DoubleField(null=True)
    weight = pw.DoubleField()
    is_exc = pw.BooleanField()

    _storage_fields = ["spike_times"]

    @property
    def has_spikes(self):
        return self.spike_times_sha1 is not None


class VmemDistribution(BaseModel):
    __metaclass__ = meta.StorageFields
    # config
    dt = pw.DoubleField(default=0.1)
    duration = pw.DoubleField(default=100000.0)
    burn_in_time = pw.DoubleField(default=200.0)
    used_parameters = pw.ForeignKeyField(rel_model=NeuronParameters,
            related_name="distributions")

    # misc
    date = pw.DateTimeField(default=datetime.datetime.now)

    # results
    mean = pw.DoubleField(null=True)
    std = pw.DoubleField(null=True)

    _storage_fields = ["voltage_trace"]

    class Meta:
        order_by = ("-date",)


class SourceCFGInCalibration(BaseModel):
    source = pw.ForeignKeyField(SourceCFG, related_name="calibrations",
            cascade=True)
    calibration = pw.ForeignKeyField(Calibration, related_name="sources",
            cascade=True)


###################################
# Utility functions for DB-access #
###################################

def sync_params_to_db(neuron_parameters):
    """
        Returns a (maybe newly created) NeuronParameters model that corresponds
        to this parameter set (allows access to calibrations).

        Note: If not all parameter names are specified, only the first matching
              element will be returned.
    """
    try:
        params = NeuronParameters.get(*(getattr(NeuronParameters, k) == v\
                    for k,v in neuron_parameters.iteritems()))
        log.debug("Parameters found in database, loaded.")
    except NeuronParameters.DoesNotExist:
        log.info("Parameters not found in database, creating new entry.")
        params = NeuronParameters.create(**neuron_parameters)

    return params


def create_source_cfg(rate, weight, is_excitatory=True):
    """
        Returns a SourceCFG-object that has been synched to the database with
        the specified rate/weight combination as well as information whether
        the source is excitatory/inhibitory.
    """
    try:
        src_cfg = SourceCFG.get(SourceCFG.rate == rate,
                SourceCFG.weight == weight,
                SourceCFG.is_exc == is_excitatory )
        log.debug("Source configuration loaded from database.")
    except SourceCFG.DoesNotExist:
        log.info("Source configuration not present in database, creating..")
        src_cfg = SourceCFG.create(rate=rate, weight=weight,
                is_exc=is_excitatory)

    return src_cfg


_merge_order = [
        NeuronParameters,
        SourceCFG,
        Calibration,
        SourceCFGInCalibration,
        VmemDistribution,
    ]

def merge_databases(db_name_source, db_name_target):
    """
        When doing several simulations at once, each can only write one database
        file.

        The easiest way afterwards is to merge databases using this function.

        NOTE: The database will be the target database afterwards.
    """
    src = db_name_source
    tgt = db_name_target

    log.info("Merging from {} into {}".format(src, tgt))

    tmp_filename = "{}/sbs_{}.shelve".format(tempfile.tempdir,
            utils.get_random_string())

    log.debug("Temporary storage set up in {}.".format(tmp_filename))
    tmp_storage = shelve.open(tmp_filename)

    id_mapping = c.defaultdict(lambda :c.defaultdict(lambda : {}))

    log.info("Reading from {}..".format(src))
    setup(src)
    for model in _merge_order:
        model_name = model.__name__
        log.info(".. {}".format(model_name))

        model_storage = {}

        # get all original fields
        model_fields = []
        for a in dir(model):
            # we dont want to sync dates
            if a == "date":
                continue
            attr = getattr(model, a)
            if isinstance(attr, pw.Field)\
                    and not isinstance(attr, pw.PrimaryKeyField):
                model_fields.append(a)

        storage_fields = getattr(model, "_storage_fields", tuple())

        all_fields = model_fields + list(storage_fields)

        for entry in model.select():
            model_storage[entry.get_id()] = entry_storage = {}

            for f in all_fields:
                if isinstance(getattr(model, f), pw.ForeignKeyField):
                    try:
                        entry_storage[f] = getattr(entry, f).get_id()
                    except AttributeError:
                        entry_storage[f] = None
                else:
                    entry_storage[f] = getattr(entry, f)

        tmp_storage[model_name] = model_storage

    if log.getEffectiveLevel() <= logging.DEBUG:
        log.debug("Source database contents.")
        log.debug(pf(tmp_storage))


    log.info("Writing to {}..".format(tgt))
    # after reading all entries from source, insert them into the target
    setup(tgt)
    for model in _merge_order:
        model_name = model.__name__
        log.info(".. {}".format(model_name))

        storage_fields = getattr(model, "_storage_fields", tuple())

        # we need to update ForeignKeys
        fk_fields = [getattr(model, a) for a in dir(model)
                if isinstance(getattr(model, a), pw.ForeignKeyField)]

        for old_id, attributes in tmp_storage[model_name].iteritems():
            # the attributes that are actually in the sql part of the database
            entry_db_attributes = {k:v for k,v in attributes.iteritems()\
                    if k not in storage_fields and v is not None}

            # update the old ids of foreignkeys with the new ones
            # (this is the only point where _merge_order is of importance)
            for fk in fk_fields:
                log.debug("Remapping {}..".format(fk.name))
                entry_db_attributes[fk.name] = \
                        id_mapping[fk.rel_model.__name__]\
                        [entry_db_attributes[fk.name]]

            if log.getEffectiveLevel() <= logging.DEBUG:
                log.debug("Entry attributes:" + pf(entry_db_attributes))

            # check if entry is already present in database
            try:
                new_entry = model.get(**entry_db_attributes)
                log.debug("Found entry.")
            except model.DoesNotExist:
                log.debug("Inserting element into database.")
                new_entry = model.create(**entry_db_attributes)
                if len(storage_fields) > 0:
                    for sf in storage_fields:
                        if attributes[sf] is not None:
                            setattr(new_entry, sf, attributes[sf])

                    # always need to call save when updating storage fields
                    new_entry.save()

            id_mapping[model_name][old_id] = new_entry.get_id()

    if log.getEffectiveLevel() <= logging.DEBUG:
        log.debug("id_mapping: " + pf(id_mapping))

    tmp_storage.clear()
    tmp_storage.close()
    try:
        os.remove(tmp_filename)
    except OSError:
        log.warn("Could not delete temporary storage {}".format(tmp_filename))

