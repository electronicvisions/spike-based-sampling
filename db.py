#!/usr/bin/env python
# encoding: utf-8

"""
    Defines all data models and db interactions.
"""

from .logcfg import log

import peewee as pw
import h5py

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

    if "calibration" not in data_storage:
        data_storage.create_group("calibration")

    for model in [NeuronParameters, Calibration]:
        if not model.table_exists():
            log.info("Creating table: {}".format(model.__name__))
            model.create_table()


def create_dataset_compressed(h5grp, *args, **kwargs):
    kwargs.setdefault("compression", "gzip")
    kwargs.setdefault("compression_opts", 9)
    return h5grp.create_dataset(*args, **kwargs)


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
    v_reset    = pw.DoubleField() # mV  Reset potential after a spike
    v_thresh   = pw.DoubleField() # mV  Spike threshold

    @property
    def gl(self):
        "Leak conductance in µS"
        return self.cm / self.tau_m

    def add_calibration(self, alpha, v_p05, samples_v_rest, samples_p_on,
            sources):
        """
            Add a new calibration and sync the samples_* arrays to the storage.
        """
        new_calib = Calibration.create(used_parameters=self, alpha=alpha,
                v_p05=v_p05)

        storage = data_storage["calibration"].create_group(str(new_calib.id))
        create_dataset_compressed(storage, name="v_rest", data=samples_v_rest)
        create_dataset_compressed(storage, name="p_on", data=samples_p_on)

        # indicate which sources were used
        for source in sources:
            SourceInCalibration.create(source=source, calibration=new_calib)

        return new_calib


class Calibration(BaseModel):
    """
        Represents a calibration result.
    """
    used_parameters = pw.ForeignKeyField(NeuronParameters,
            related_name="calibrations")
    alpha = pw.DoubleField()
    v_p05 = pw.DoubleField()
    duration = pw.DoubleField()
    num_samples = pw.IntegerField()

    def get_storage_group(self):
        return data_storage["calibration"][str(self.id)]

    @property
    def samples_v_rest(self):
        return np.array(self.get_storage_group()["v_rest"])

    @property
    def samples_p_on(self):
        return np.array(self.get_storage_group()["p_on"])


class Source(BaseModel):
    rate = pw.DoubleField()
    weight = pw.DoubleField()
    is_exc = pw.BooleanField()


class SourceInCalibration(BaseModel):
    source = pw.ForeignKeyField(Source, related_name="calibrations")
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


