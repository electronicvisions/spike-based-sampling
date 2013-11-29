#!/usr/bin/env python
# encoding: utf-8

from .logcfg import log

from . import db

class LIFsampler(object):

    supported_pynn_neuronmodels = ["IF_curr_exp", "IF_cond_exp"]

    def __init__(self, sim, pynn_neuronmodel=None, neuron_parameters=None,
            index=-1):
        """
            `sim`: The used simulator.

            `neuron_pynnmodel`: String specifying which pyNN model to use.

            `neuron_parameters`: Parameters to pyNN model.

            Alternatively: If both `pynn_neuronmodel` and `neuron_parameters`
                are None, the stored parameters of `index` are loaded.
        """
        self.sim = sim

        if pynn_neuronmodel is not None and neuron_parameters is not None:
            if pynn_neuronmodel not in self.supported_pynn_neuronmodels:
                raise Exception("Neuron model not supported!")
            log.info("Checking parameters in database..")
            neuron_parameters["pynn_model"] = pynn_neuronmodel
            self.db_params = db.sync_params_to_db(neuron_parameters)

        elif pynn_neuronmodel is None and neuron_parameters is None:
            log.info("Getting parameters with index {}".format(index))
            query = db.NeuronParameters.select()
            if index < 0:
                offset = query.count() + index

            if offset > 0:
                query.offset(offset)

            self.db_params = query.get()

        else:
            raise Exception("Please specify both model and parameters or "
                            "neither.")

        self.bias = 0.
        self.mu = 0.
        self.sigma = 1.

        # the loaded calibration object from the database
        # which will be used for adjusting neuron parameters
        # as well as determining weights
        self.db_calibration = None

        # the sources used in calibration
        self.db_sources = None

        self.population = None


    def set_bias(self, bias):
        self.bias = bias


    @property
    def is_calibrated(self):
        return self.db_calibration is not None


    def load_calibration(self, index=-1):
        """
            Attempt to load an existing calibration.

            `index` can be used to specify which configuration to use when more
            are available (per default the latest will be used)

            Returns a bool to indicate whether or not the calibration data
            was successfully loaded.
        """
        query = db.Calibration.select()\
                .where(db.Calibration.used_parameters == self.db_params)

        if index != 0:
            if index > 0:
                offset = index
            else:
                offset = query.count() + index

            # account for out of bounds index
            if offset > 0 and offset < query.count():
                query.offset(offset)

        try:
            self.db_calibration = query.get()
        except db.Calibration.DoesNotExist:
            return False

        self._load_sources()
        self._calc_distribution()

        return True


    def _calc_distribution(self):
        raise NotImplementedError


    def _load_sources(self):
        assert(self.is_calibrated)
        self.db_sources = list(db.Source.select()\
                .join(db.SourceInCalibration).join(db.Calibration)\
                .where(db.Calibration.id == self.calibration).naive())


    def create(self, population=None, create_sources=True):
        """
            Actually configures the supplied pyNN-object `popluation`
            (Population, PopulationView etc.) to have the loaded parameters and
            calibration.

            If no `pynn_object` is supplied a Population of size 1 will be
            created.

            Usually, the same source-configuration that was present during
            calibration would be created and used, this can be disabled via
            `create_sources`.
        """
        assert(self.is_calibrated)

        self.population = population
        if self.population is None:
            self.population = sim.Population(1,
                    getattr(self.sim, self.db_params.pynn_model)())

        # get all parameters needed for instance
        params = self.get_parameters()

        self.population.set(**params)

        if create_sources == True:
            self._create_sources()

        return self.population

    def _create_sources(self):
        assert(self.is_calibrated)
        raise NotImplementedError


    def get_v_rest_from_bias(self):
        assert(self.is_calibrated)
        return self.calibration.v_p05 + self.calibration.alpha * self.bias


    @property
    def pynn_model(self):
        return self.db_parmas.pynn_model


    def get_parameters(self):
        ignored_fields = ["neuron_pynnmodel", "id"]
        params = {k: v for k,v in self.db_params._data.iteritems()\
                if v is not None and k not in ignored_fields}
        params["v_rest"] = self.get_v_rest_from_bias()
        return params


    @property
    def is_created(self):
        return self.population is not None


    def do_calibration(self, duration,
            source_weights_exc, source_weights_inh,
            source_rates_exc, source_rates_inh,
            num_samples=1000):
        """
            Calibrate the sampler, using the specified source parameters.

            Both excitatory/inhibitory weights/rates are expected to be lists!
        """
        assert(len(source_weights_exc) == len(source_rates_exc))
        assert(len(source_weights_inh) == len(source_rates_inh))

        raise NotImplementedError


