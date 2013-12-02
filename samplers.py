#!/usr/bin/env python
# encoding: utf-8

from .logcfg import log
from . import utils
from . import db
from . import meta

import itertools as it

class LIFsampler(object):

    supported_pynn_neuronmodels = ["IF_curr_exp", "IF_cond_exp"]

    def __init__(self, sim_name="pyNN.neuron",
            pynn_neuronmodel=None, neuron_parameters=None, id=None):
        """
            `sim_name`: Name of the used simulator.

            `neuron_pynnmodel`: String specifying which pyNN model to use.

            `neuron_parameters`: Parameters to pyNN model.

            Alternatively: If both `pynn_neuronmodel` and `neuron_parameters`
                are None, the stored parameters of `id` are loaded.
        """
        self.sim_name = sim_name

        if pynn_neuronmodel is not None and neuron_parameters is not None:
            self._ensure_model_is_supported(pynn_neuronmodel)
            log.info("Checking parameters in database..")
            neuron_parameters["pynn_model"] = pynn_neuronmodel
            self.db_params = db.sync_params_to_db(neuron_parameters)

        elif pynn_neuronmodel is None and neuron_parameters is None:
            log.info("Getting parameters with index {}".format(index))
            query = db.NeuronParameters.select()

            if id is not None:
                query.where(db.NeuronParameters.id == id)

            try:
                self.db_params = query.get()
            except db.NeuronParameters.DoesNotExist:
                raise Exception("No neuron parameters found!")

        else:
            raise Exception("Please specify both model and parameters or "
                            "neither.")

        self.bias_theo = 0.

        # the loaded calibration object from the database
        # which will be used for adjusting neuron parameters
        # as well as determining weights
        self.db_calibration = None

        # the sources used in calibration
        self.db_sources = None

        self.population = None


    # implement bias_bio and bias_theo as properties so
    # the user can assign either and query the other automatically
    # (once the sampler is calibrated)
    @property
    def bias_theo(self):
        if not self.bias_is_theo:
            # if the bias is in bio units we need calibration to give the bio
            # equivalent
            assert(self.is_calibrated)
            return self._bias / self.db_calibration.alpha
        else:
            return self._bias

    @property
    def bias_bio(self):
        if self.bias_is_theo:
            # if the bias is theo we need calibration to give the bio equivalent
            assert(self.is_calibrated)
            return self._bias * self.db_calibration.alpha
        else:
            return self._bias


    @bias_theo.setter
    def bias_theo(self, bias):
        self._bias = bias
        self.bias_is_theo = True


    @bias_bio.setter
    def bias_bio(self, bias):
        self._bias = bias
        self.bias_is_theo = False


    @property
    def mean(self):
        assert(self.is_calibrated)
        return self.db_calibration.mean

    @property
    def std(self):
        assert(self.is_calibrated)
        return self.db_calibration.std

    @property
    def g_tot(self):
        assert(self.is_calibrated)
        return self.db_calibration.g_tot

    def _ensure_model_is_supported(self, pynn_neuronmodel=None):
        if pynn_neuronmodel is None:
            pynn_neuronmodel = self.pynn_model
        if pynn_neuronmodel not in self.supported_pynn_neuronmodels:
            raise Exception("Neuron model not supported!")


    @property
    def is_calibrated(self):
        return self.db_calibration is not None


    def get_calibration_ids(self):
        """
            Lists all available calibration ids for this set of neuron
            parameters, newest first.
        """
        query = db.Calibration.select(db.Calibration.id)\
                .where(db.Calibration.used_parameters == self.db_params)


    def load_calibration(self, id=None):
        """
            Attempt to load an existing calibration.

            `id` can be used to specify a calibration event from the database
            which has to have been done with this set of neuron parameters.

            Returns a bool to indicate whether or not the calibration data
            was successfully loaded.
        """
        query = db.Calibration.select()\
                .where(db.Calibration.used_parameters == self.db_params)

        if id is not None:
            query.where(db.Calibration.id == id)

        try:
            self.db_calibration = query.get()
        except db.Calibration.DoesNotExist:
            return False

        self._load_sources()
        self._post_calibration()

        return True


    def _load_sources(self):
        assert(self.is_calibrated)
        self.db_sources = list(db.SourceCFG.select()\
                .join(db.SourceCFGInCalibration).join(db.Calibration)\
                .where(db.Calibration.id == self.calibration).naive())


    def create(self, population=None, create_pynn_sources=True):
        """
            Actually configures the supplied pyNN-object `popluation`
            (Population, PopulationView etc.) to have the loaded parameters and
            calibration.

            If no `pynn_object` is supplied a Population of size 1 will be
            created.

            Usually, the same source-configuration that was present during
            calibration would be created and used, this can be disabled via
            `create_pynn_sources`.
        """
        assert(self.is_calibrated)

        exec("import {} as sim".format(self.sim_string))
        self.sim = sim

        self.population = population
        if self.population is None:
            self.population = sim.Population(1,
                    getattr(self.sim, self.db_params.pynn_model)())

        # get all parameters needed for instance
        params = self.get_parameters()

        self.population.set(**params)

        if create_pynn_sources == True:
            self._create_pynn_sources()

        return self.population

    def _create_pynn_sources(self):
        assert(self.is_calibrated)
        raise NotImplementedError


    def get_v_rest_from_bias(self):
        assert(self.is_calibrated)
        return self.calibration.v_p05 + self.bias_bio


    @property
    def pynn_model(self):
        return self.db_parmas.pynn_model


    def get_pynn_parameters(self):
        """
            Returns dictionary with all needed pynn parameters to implement
            the sampler. Note: The resting potential will be set according to
            the specified bias.
        """
        assert(self.is_calibrated)
        params = self.db_params.get_pynn_parameters()
        params["v_rest"] = self.get_v_rest_from_bias()
        return params


    @property
    def is_created(self):
        return self.population is not None


    @property
    def sources_configured(self):
        return self.db_sources is not None


    def forget_calibration(self):
        """
            Unsets the calibration for this sampler.
        """
        self.db_calibration = None


    def set_source_cfg(self,
            weights_exc, weights_inh,
            rates_exc, rates_inh):
        assert not self.is_calibrated, "Sources already loaded from calibration"
        assert(len(weights_exc) == len(rates_exc))
        assert(len(weights_inh) == len(rates_inh))

        self.db_sources = []
        for weight, rate, is_exc in\
                it.chain(
                    (it.izip(*a) for a in [
                            (weights_exc, rates_exc, it.repeat(True)),
                            (weights_inh, rates_inh, it.repeat(False)),
                        ]
                    )
                ):
            self.db_sources.append(db.create_source_cfg(rate, weight, is_exc))


    def get_source_cfg(self, is_exc=True):
        """
            Return excitatory (default) or inhibitory (`is_exc==False`) source
            configuration.
        """
        return filter(lambda x: x.is_exc==is_exc, self.db_sources)


    def get_source_rates(self, is_exc=True):
        return [src.rate for src in self.get_source_cfg(is_exc)]


    def get_source_weights(self, is_exc=True):
        return [src.weight for src in self.get_source_cfg(is_exc)]


    def calibrate(self, duration=10000., num_samples=1000, std_range=4.):
        """
            Calibrate the sampler, using the specified source parameters.
        """
        assert self.sources_configured, "Please use `set_source_cfg` prior to "\
                "calibrating."

        self.db_calibration = db.Calibration(duration=duration,
                num_samples=num_samples, std_range=std_range)

        # sync to db because the gathering function writes to it
        self.db_calibration.save()

        self._calc_distribution()
        self._estimate_alpha()

        # by importing here we avoid importing networking stuff until we have to
        from .gather_calibration_data import gather_calibration_data
        gather_calibration_data(
                db_neuron_params=self.db_params,
                db_partial_calibration=self.db_calibration,
                db_sources=self.db_sources,
                sim_name=self.sim_name)

        self.db_calibration.v_p05, self.db_calibration.alpha = fit.fit_sigmoid(
            np.array(self.db_calibration.samples_v_rest),
            np.array(self.db_calibration.samples_p_on),
            guess_p05=self.db_params.v_thresh,
            guess_alpha=self.db_calibration.alpha_theo)

        self.db_calibration.save()
        self.db_calibration.link_sources(self.db_sources)


    def get_all_source_parameters(self):
        """
            Returns a tuple of `np.array`s with source configuration
                (rates_exc, rates_inh, weights_exc, weights_inh)
        """
        assert(self.sources_configured)

        return utils.get_all_source_parameters(self.db_sources)


    def _calc_distribution(self):
        dbc = self.db_calibration

        dist = getattr(utils, "{}_distribution".format(self.pynn_model))

        args = self.get_all_source_parameters()
        kwargs = self.db_params.get_pynn_parameters()
        # gl is not a pynn parameter
        kwargs["gl"] = self.db_params.gl

        dbc.mean, dbc.std, dbc.g_tot = dist(*args, **kwargs)


    def _estimate_alpha(self):
        dbc = self.db_calibration
        # estimate for syn weight factor from theo to LIF
        dbc.alpha_theo = .25 * np.sqrt(2. * np.pi) * dbc.std

