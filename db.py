#!/usr/bin/env python
# encoding: utf-8

"""
    Defines all data models and db interactions.
"""

from .logcfg import log

import peewee as pw
import numpy as np
import h5py
import datetime

# database for parameters
database = pw.SqliteDatabase(None)

# storage for calibration data (subgroup '/calibration')
# subgroups are the primary keys for the calibration-rows in the database
# then they have two datasets: v_rest and p_on
data_storage = None

# classes for datastorage

def setup_database(basename="database"):

    db_name = "{}.sql".format(basename)
    ds_name = "{}.h5".format(basename)

    log.info("Setting up database: {}".format(db_name))
    database.init(db_name)

    log.info("Setting up storage for datasets: {}".format(ds_name))
    global data_storage
    data_storage = h5py.File(ds_name, "a")

    for model in [NeuronParameters, Calibration, SourceCFG, SourceCFGInCalibration]:
        if not model.table_exists():
            log.info("Creating table: {}".format(model.__name__))
            model.create_table()


def create_dataset_compressed(h5grp, *args, **kwargs):
    kwargs.setdefault("compression", "gzip")
    kwargs.setdefault("compression_opts", 9)
    return h5grp.create_dataset(*args, **kwargs)


def ensure_group_exists(h5grp, name):
    if name not in h5grp:
        h5grp.create_group(name)
    return h5grp[name]


class BaseModel(pw.Model):
    """
        The base model that sets the database.
    """
    class Meta(object):
        database = database


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
        return params

    class Meta:
        order_by = ("-date",)
        indexes = (
                ((
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


@setup_storage_fields
class Calibration(BaseModel):
    """
        Represents a calibration result.
    """
    # configuration
    duration = pw.DoubleField(default=10000.)
    num_samples = pw.IntegerField(default=1000)
    std_range = pw.DoubleField(default=4.)

    used_parameters = pw.ForeignKeyField(NeuronParameters,
            related_name="calibrations", index=True)

    # results
    alpha = pw.DoubleField()
    v_p05 = pw.DoubleField()
    mean = pw.DoubleField()
    std = pw.DoubleField()
    g_tot = pw.DoubleField()

    storage_fields = ["samples_v_rest", "samples_p_on"]

    # for book keeping
    date = pw.DateTimeField(default=datetime.datetime.now)

    def link_sources(self, sources):
        """
            Note: SourceCFGs have to be present in the database already!
        """
        assert(not any((src.get_id() is None for src in sources)))
        # see if there already are any sources linked
        present_sources = list(SourceCFGs.select(SourceCFGs.id)\
                .join(SourceCFGInCalibration)\
                .join(Calibration).where(Calibration == self))

        for src in filter(lambda x: x not in present_sources, sources):
            SourceCFGInCalibration.create(source=src, calibration=self)

    class Meta:
        order_by = ("-date",)


class SourceCFG(BaseModel):
    rate = pw.DoubleField()
    weight = pw.DoubleField()
    is_exc = pw.BooleanField()


class SourceCFGInCalibration(BaseModel):
    source = pw.ForeignKeyField(SourceCFG, related_name="calibrations")
    calibration = pw.ForeignKeyField(Calibration, related_name="sources")


def sync_params_to_db(neuron_parameters):
    """
        Returns a (maybe newly created) NeuronParameters model that corresponds
        to this parameter set (allows access to calibrations).

        Note: If not all parameter names are specified, only the first matching
              element will be returned.
    """
    try:
        params = NeuronParameters.get(*(getattr(NeuronParams, k) == v\
                    for k,v in neuron_parameters.iterkeys()))
        log.info("Parameters found in database, loaded.")
    except NeuronParams.DoesNotExist:
        log.info("Parameters not found in database, creating new entry.")
        params = NeuronParams.create(**neuron_parameters)

    return params


def create_source_cfg(rate, weight, is_excitatory=True):
    """
        Returns a SourceCFGs-object that has been synched to the database with
        the specified rate/weight combination as well as information whether
        the source is excitatory/inhibitory.
    """
    try:
        src_cfg = SourceCFG.get(SourceCFG.rate == rate,
                SourceCFG.weight == weight,
                SourceCFG.is_exc == is_excitatory )
        log.info("SourceCFG configuration loaded from database.")
    except SourceCFG.DoesNotExist:
        log.info("SourceCFG configuration not present in database, creating..")
        src_cfg = SourceCFG.create(rate=rate, weight=weight,
                is_exc=is_excitatory)

    return src_cfg




