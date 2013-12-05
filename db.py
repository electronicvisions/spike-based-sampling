#!/usr/bin/env python
# encoding: utf-8

"""
    Defines all data models and db interactions.
"""

from .logcfg import log
from . import utils
from . import meta

import peewee as pw
import numpy as np
import h5py
import datetime

# database for parameters
database = pw.SqliteDatabase(None)

# classes for datastorage

def setup_database(basename="database"):

    db_name = "{}.sql".format(basename)
    ds_name = "{}.h5".format(basename)

    log.info("Setting up database: {}".format(db_name))
    database.init(db_name)

    log.info("Setting up storage for datasets: {}".format(ds_name))
    meta.data_storage = h5py.File(ds_name, "a")

    for model in [NeuronParameters, Calibration, SourceCFG,
            SourceCFGInCalibration]:
        if not model.table_exists():
            log.info("Creating table: {}".format(model.__name__))
            model.create_table()


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

    dq = filter_incomplete_calibrations(Calibration.delete())

    num_deleted = dq.execute()
    log.info("Purged {} partial calibration{}..".format(num_deleted,
        "s" if num_deleted > 1 else ""))


def get_incomplete_calibration_ids():
    """
        List of ids of incomplete calibrations.
    """
    return list(filter_incomplete_calibrations(
        Calibration.select(Calibration.id)))



class BaseModel(pw.Model):
    """
        The base model that sets the database.
    """
    class Meta(object):
        database = database


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


@meta.setup_storage_fields
class Calibration(BaseModel):
    """
        Represents a calibration result.
    """
    cfg_fields = ["duration", "num_samples", "std_range"]
    # configuration
    duration = pw.DoubleField(default=10000.)
    num_samples = pw.IntegerField(default=1000)
    std_range = pw.DoubleField(default=4.)
    burn_in_time = pw.DoubleField(default=100.)
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

    storage_fields = ["samples_v_rest", "samples_p_on"]

    def link_sources(self, sources):
        """
            Note: SourceCFGs have to be present in the database already!
                  Also, they have to be all linked at the same time (but do not
                  have to be unique).
        """
        assert(not any((src.get_id() is None for src in sources)))
        # delete any sources previously linked to this node
        # node we do allow sources to be specified again

        dq = SourceCFGInCalibration.delete().where(
                SourceCFGInCalibration.calibration==self)

        num_deleted = dq.execute()

        log.debug("Deleted {} previoiusly linked sources.".format(num_deleted))

        for src in sources:
            SourceCFGInCalibration.create(source=src, calibration=self)

    class Meta:
        order_by = ("-date",)


@meta.setup_storage_fields
class SourceCFG(BaseModel):
    # if rate is None we have spike times specified in the storage!
    rate = pw.DoubleField(null=True)
    weight = pw.DoubleField()
    is_exc = pw.BooleanField()

    storage_fields = ["spike_times"]

    @property
    def has_spikes(self):
        return self.rate is None


@meta.setup_storage_fields
class VmemDistribution(BaseModel):
    # config
    dt = pw.DoubleField(default=0.1)

    # results
    mean = pw.DoubleField(null=True)
    std = pw.DoubleField(null=True)

    storage_fields = ["voltage_trace"]


class SourceCFGInCalibration(BaseModel):
    source = pw.ForeignKeyField(SourceCFG, related_name="calibrations",
            cascade=True)
    calibration = pw.ForeignKeyField(Calibration, related_name="sources",
            cascade=True)


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
        log.info("Parameters found in database, loaded.")
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
        log.info("Source configuration loaded from database.")
    except SourceCFG.DoesNotExist:
        log.info("Source configuration not present in database, creating..")
        src_cfg = SourceCFG.create(rate=rate, weight=weight,
                is_exc=is_excitatory)

    return src_cfg




