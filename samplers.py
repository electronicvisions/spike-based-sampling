#!/usr/bin/env python
# encoding: utf-8

from .logcfg import log
from . import utils
from . import db
from . import fit
from . import meta
from . import buildingblocks as bb
from . import cutils

import itertools as it
import numpy as np


@meta.HasDependencies
class LIFsampler(object):

    supported_pynn_neuron_models = ["IF_curr_exp", "IF_cond_exp"]

    def __init__(self, sim_name="pyNN.nest",
            pynn_model=None, neuron_parameters=None, id=None, silent=False):
        """
            `sim_name`: Name of the used simulator.

            `neuron_pynnmodel`: String specifying which pyNN model to use.

            `neuron_parameters`: Parameters to pyNN model.

            Alternatively: If both `pynn_model` and `neuron_parameters`
                are None, the stored parameters of `id` are loaded.
        """
        self.silent = silent

        if not self.silent:
            log.info("Setting up sampler.")
        self.sim_name = sim_name

        if pynn_model is not None and neuron_parameters is not None:
            self._ensure_model_is_supported(pynn_model)
            if not self.silent:
                log.info("Checking parameters in database..")
            neuron_parameters["pynn_model"] = pynn_model
            self.db_params = db.sync_params_to_db(neuron_parameters)

        elif pynn_model is None and neuron_parameters is None:
            if not self.silent:
                log.info("Getting parameters with id {}.".format(id))
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

        self.free_vmem_trace = None

        self.population = None
        self.sources = None
        self.source_populations = None

    @meta.DependsOn()
    def sim_name(self, name):
        """
            The full simulator name.
        """
        if not name.startswith("pyNN."):
            name = "pyNN." + name
        return name

    # implement bias_bio and bias_theo as properties so
    # the user can assign either and query the other automatically
    # (once the sampler is calibrated)
    @meta.DependsOn("bias_bio")
    def bias_theo(self, value=None):
        """
            Get or set the bias in theoretical units.

            Automatic conversion:
            After the bias has been set in either biological or theoretical
            units both can be retrieved and the conversion will be done when
            needed.
        """
        if value is None:
            # if the bias is in bio units we need calibration to give the
            # theoretical equivalent
            assert(self.is_calibrated)
            return self.bias_bio / self.db_calibration.alpha
        else:
            return value

    @meta.DependsOn("bias_theo")
    def bias_bio(self, value=None):
        """
            Get or set the bias in biological units (mV).

            Automatic conversion:
            After the bias has been set in either biological or theoretical
            units both can be retrieved and the conversion will be done when
            needed.
        """
        if value is None:
            # if the bias is theo we need calibration to give the bio equivalent
            assert(self.is_calibrated)
            return self.bias_theo * self.db_calibration.alpha
        else:
            return value

    def sync_bias_to_pynn(self):
        assert self.is_created
        self.population.set(v_rest=self.get_v_rest_from_bias())

    @meta.DependsOn("db_calibration")
    def vmem_mean_theo(self):
        assert(self.is_calibrated)
        return self.db_calibration.mean

    @meta.DependsOn("db_calibration")
    def vmem_std_theo(self):
        assert(self.is_calibrated)
        return self.db_calibration.std

    @meta.DependsOn("db_calibration")
    def g_tot(self):
        assert(self.is_calibrated)
        return self.db_calibration.g_tot

    @meta.DependsOn("db_calibration")
    def alpha_fitted(self):
        assert(self.is_completely_calibrated)
        return self.db_calibration.alpha

    @meta.DependsOn("db_calibration")
    def alpha_theo(self):
        assert(self.is_calibrated)
        return self.db_calibration.alpha_theo

    @meta.DependsOn("db_calibration")
    def v_p05(self):
        assert(self.is_calibrated)
        return self.db_calibration.v_p05

    @meta.DependsOn()
    def db_calibration(self, value=None):
        """
            The database calibration object.
        """
        return value

    @property
    def is_calibrated(self):
        return self.db_calibration is not None

    @property
    def is_completely_calibrated(self):
        return self.is_calibrated and self.db_calibration.is_complete

    @property
    def pynn_model(self):
        return self.db_params.pynn_model

    @property
    def has_free_vmem_trace(self):
        return self.free_vmem_trace is not None

    def get_pynn_parameters(self, adjust_vrest=True):
        """
            Returns dictionary with all needed pynn parameters to implement
            the sampler. Note: The resting potential will be set according to
            the specified bias.

            `adjust_vrest` can be set to False to maintain the original rest
            value. This is only really useful during calibration as there are
            not values yet with which to update.
        """
        assert(self.is_calibrated)
        params = self.db_params.get_pynn_parameters()
        if adjust_vrest:
            params["v_rest"] = self.get_v_rest_from_bias()
        return params

    def get_parameters_id(self):
        """
            Return the id of the parameters in the database.
        """
        return self.db_params.id

    def get_calibration_id(self):
        """
            Return the id of the calibration used currently.

            Returns None if the sampler has not been calibrated.
        """
        if self.is_calibrated:
            return self.db_calibration.id
        else:
            return None

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

    def get_calibration_ids(self):
        """
            Lists all available calibration ids for this set of neuron
            parameters, newest first.
        """
        query = db.Calibration.select(db.Calibration.id)\
                .where(db.Calibration.used_parameters == self.db_params)
        return [calib.id for calib in query]

    def get_v_rest_from_bias(self):
        assert(self.is_calibrated)
        return self.db_calibration.v_p05 + self.bias_bio


    def list_calibrations(self):
        """
            Return a list of ids of calibrations for this sampler.
        """
        query = db.Calibration.select()\
                .where(db.Calibration.used_parameters == self.db_params)

        return [calib.id for calib in query]


    def load_calibration(self, id=None):
        """
            Attempt to load an existing calibration.

            `id` can be used to specify a calibration event from the database
            which has to have been done with this set of neuron parameters.

            Returns a bool to indicate whether or not the calibration data
            was successfully loaded.
        """
        if not self.silent:
            log.info("Attempting to load calibration.")
        query = db.Calibration.select()\
                .where(db.Calibration.used_parameters == self.db_params)

        if id is not None:
            query.where(db.Calibration.id == id)

        try:
            self.db_calibration = query.get()
        except db.Calibration.DoesNotExist:
            if not self.silent:
                log.info("No calibration present.")
            return False

        self._load_sources()

        if not self.silent:
            log.info("Calibration with id {} loaded.".format(
            self.db_calibration.id))
        return True

    def set_source_cfg(self,
            weights_exc, weights_inh,
            rates_exc, rates_inh):
        assert not self.is_calibrated, "Sources already loaded from calibration"
        assert(len(weights_exc) == len(rates_exc))
        assert(len(weights_inh) == len(rates_inh))

        self.db_sources = []
        for weight, rate, is_exc in\
                it.chain(
                    *(it.izip(*a) for a in [
                            (weights_exc, rates_exc, it.repeat(True)),
                            (weights_inh, rates_inh, it.repeat(False)),
                        ]
                    )
                ):
            self.db_sources.append(db.create_source_cfg(rate, weight, is_exc))

    def calibrate(self, **calibration_params):
        """
            Calibrate the sampler, using the specified source parameters.
        """
        assert self.sources_configured, "Please use `set_source_cfg` prior to "\
                "calibrating."

        calibration_params["simulator"] = self.sim_name
        calibration_params["used_parameters"] = self.db_params

        # sync to db because the gathering function writes to it

        self.db_calibration = db.Calibration(**calibration_params)

        self._calc_distribution_theo()
        self._estimate_alpha()

        self.db_calibration.save()

        # by importing here we avoid importing networking stuff until we have to
        from .gather_data import gather_calibration_data
        self.db_calibration.samples_v_rest, self.db_calibration.samples_p_on =\
                gather_calibration_data(
                    sim_name=self.sim_name,
                    calib_cfg=self.db_calibration.get_non_null_fields(),
                    pynn_model=self.pynn_model,
                    neuron_params=self.db_params.get_pynn_parameters(),
                    sources_cfg=self.get_sources_cfg_lod()
                )

        if not self.silent:
            log.info("Calibration data gathered, performing fit.")
        self.db_calibration.v_p05, self.db_calibration.alpha = fit.fit_sigmoid(
            self.db_calibration.samples_v_rest,
            self.db_calibration.samples_p_on,
            guess_p05=self.db_params.v_thresh,
            guess_alpha=self.db_calibration.alpha_theo)
        self.db_calibration.save()

        if not self.silent:
            log.info("Fitted alpha: {:.3f}".format(self.alpha_fitted))
            log.info("Fitted v_p05: {:.3f} mV".format(self.v_p05))

        self.db_calibration.link_sources(self.db_sources)

    def measure_free_vmem_dist(self, duration=100000., dt=0.1, burn_in_time=200.):
        """
            Measure the distribution of the free membrane potential, given
            the parameters (attributes of VmemDistribution).
        """
        assert self.is_calibrated

        from .gather_data import gather_free_vmem_trace

        self.free_vmem_trace = gather_free_vmem_trace(
                distribution_params={
                        "duration": duration,
                        "dt": dt,
                        "burn_in_time": burn_in_time,
                    },
                pynn_model=self.pynn_model,
                neuron_params=self.get_pynn_parameters(),
                sources_cfg=self.get_sources_cfg_lod(),
                sim_name=self.sim_name
            )

    def get_sources_cfg_lod(self):
        """
            Get source parameters as List-Of-Dictionaries
        """
        needed_attributes = [
                "rate",
                "weight",
                "is_exc",
                "has_spikes",
                "spike_times",
            ]
        return [{k: getattr(src, k) for k in needed_attributes}
                for src in self.db_sources]

    def get_all_source_parameters(self):
        """
            Returns a tuple of `np.array`s with source configuration
                (rates_exc, rates_inh, weights_exc, weights_inh)
        """
        assert(self.sources_configured)

        return utils.get_all_source_parameters(self.db_sources)

    def get_vmem_dist_theo(self):
        dist = getattr(utils, "{}_distribution".format(self.pynn_model))
        args = self.get_all_source_parameters()
        if not self.is_completely_calibrated:
            if not self.silent:
                log.info("Computing vmem distribution ONLY from supplied neuron "
                     "parameters!")
            kwargs = self.db_params.get_pynn_parameters()
        else:
            if not self.silent:
                log.info("Computing vmem distribution with bias set to {}.".format(
                self.bias_theo))
            kwargs = self.get_pynn_parameters()

        # g_l is not a pynn parameter
        kwargs["g_l"] = self.db_params.g_l

        return dist(*args, **kwargs)

    # this is just kept for now to make sure the other factor is correct --obreitwi, 14-02-14 20:45:13
    # def factor_weights_bio_to_theo(self):
        # if self.pynn_model == "IF_cond_exp":
            # # from minimization of L2(PSP-rect) -> no more blue sky!!!
            # # (original comment from v1 code --obreitwi, 19-12-13 21:10:08)
            # delta_E = np.array([
                    # self.db_calibration.mean - self.db_params.e_rev_I,
                    # self.db_params.e_rev_E - self.db_calibration.mean
                # ])
            # theo_weights = weights * delta_E[is_exc_int] /\
                # (cm - g_tot * tau[is_exc_int]) *\
                # (- cm / g_tot * (np.exp(- tau[is_exc_int] * g_tot / cm) - 1.)\
                    # + tau[is_exc_int] * (np.exp(-1.) - 1.)
                # ) / self.alpha_fitted * g_tot / self.db_params.g_l 
        # elif self.pynn_model == "IF_curr_exp":
            # theo_weights = weights / (cm - g_tot * tau[is_exc_int]) *\
                # (- cm / g_tot * (np.exp(- tau[is_exc_int] * g_tot / cm) - 1.)\
                    # + tau[is_exc_int] * (np.exp(-1.) - 1.)
                # ) / self.alpha_fiited

    @meta.DependsOn("db_calibration", "bias_theo", "bias_bio")
    def factor_weights_theo_to_bio_exc(self):
        return self._calc_factor_weights_theo_to_bio(
                is_excitatory=True,
                tau=self.db_params.tau_syn_E
            )

    @meta.DependsOn("db_calibration", "bias_theo", "bias_bio")
    def factor_weights_theo_to_bio_inh(self):
        mean, std, g_tot = self.get_vmem_dist_theo()
        return self._calc_factor_weights_theo_to_bio(
                is_excitatory=False,
                tau=self.db_params.tau_syn_I
            )

    def convert_weights_theo_to_bio(self, weights):
        """
            Convert a theoretical boltzmann weight array to biological units
            (dependening on calibration).

            We assume a excitatory target for weights => 0 and inhibitory for
            weights < 0.!

            NOTE: These weights should be inbound for this sampler!
        """
        assert self.is_calibrated
        weights = np.array(weights)

        is_exc = weights >= 0.

        # make an integer array to select one of two values based on the bools
        is_exc_int = np.array(is_exc, dtype=int)

        factor = np.array([
                self.factor_weights_theo_to_bio_inh,
                self.factor_weights_theo_to_bio_exc,
            ])

        nnweights = weights * factor[is_exc_int]

        return nnweights

    def convert_weights_bio_to_theo(self, weights):
        """
            Convert a biological weight array to theoretical (Boltzmann) units
            (dependening on calibration).

            We assume a excitatory target for weights => 0 and inhibitory for
            weights < 0.!

            NOTE: These weights should be inbound for this sampler!
        """
        assert self.is_calibrated
        weights = np.array(weights)

        is_exc = weights >= 0.

        # make an integer array to select one of two values based on the bools
        is_exc_int = np.array(is_exc, dtype=int)

        factor = np.array([
                self.factor_weights_theo_to_bio_inh,
                self.factor_weights_theo_to_bio_exc,
            ])

        theo_weights = weights / factor[is_exc_int]

        return theo_weights

    ##################
    # PYNN methods #
    ##################

    def create(self, duration=None, population=None, create_pynn_sources=True):
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

        exec("import {} as sim".format(self.sim_name))
        self.sim = sim

        self.population = population
        if self.population is None:
            self.population = sim.Population(1,
                    getattr(self.sim, self.pynn_model)())

        # get all parameters needed for instance
        params = self.get_pynn_parameters()

        self.population.set(**params)

        if create_pynn_sources == True:
            assert duration is not None, "Instructed to create sources "\
                    "without duration!"
            sources_cfg = self.get_sources_cfg_lod()
            self.sources = bb.create_sources(sim, sources_cfg, duration)
            self.source_projections = bb.connect_sources(sim, sources_cfg,
                    self.sources, self.population)

        return self.population

    ##################
    # PLOT FUNCTIONS #
    ##################

    @meta.plot_function("calibration")
    def plot_calibration(self, plot_v_dist_theo=False, plot_vlines=True,
            fig=None, ax=None):
        assert self.is_calibrated

        samples_v_rest = self.db_calibration.samples_v_rest
        samples_p_on = self.db_calibration.samples_p_on

        v_thresh = self.db_params.v_thresh
        v_p05 = self.db_calibration.v_p05
        std = self.db_calibration.std

        xdata = np.linspace(v_thresh-4.*std, v_thresh+4.*std, 500)

        if plot_v_dist_theo:
            estim_dist_v_thresh = utils.gauss(xdata, v_thresh, std)
        estim_cdf_v_thresh = utils.erfm(xdata, v_thresh, std)
        estim_sigmoid = utils.sigmoid_trans(xdata, v_thresh, self.alpha_theo)

        fitted_p_on = utils.sigmoid_trans(samples_v_rest, v_p05,
                self.alpha_fitted)

        if plot_v_dist_theo:
            ax.plot(xdata, estim_dist_v_thresh,
                    label="est. $V_{mem}$ distribution @ $\mu = v_{thresh}$")
        ax.plot(xdata, estim_cdf_v_thresh,
                label="est. CDF of $V_{mem}$ @ $\mu =v_{thresh}$")
        ax.plot(xdata, estim_sigmoid,
                label="est. trf sigm w/ $p(V > V_{thresh} = p_{ON})$")
        ax.plot(samples_v_rest, fitted_p_on, label="fitted $p_{ON}$")
        ax.plot(samples_v_rest, samples_p_on, marker="x", ls="", c="b",
                label="measured $p_{ON}$")

        if plot_vlines:
            ax.axvline(v_thresh, ls="--", label="$v_{thresh}$", c="r")
            ax.axvline(v_p05, ls="--", label="$v_{p=0.5}$", c="b")

        ax.set_xlabel("$V_{rest}$")
        ax.set_ylabel("$p_{ON}$")

        ax.legend(bbox_to_anchor=(1.15, .5))

    @meta.plot_function("free_vmem_dist")
    def plot_free_vmem(self, num_bins=200, plot_vlines=True, fig=None, ax=None):
        assert self.has_free_vmem_trace
        assert self.is_calibrated

        volttrace = self.free_vmem_trace

        counts, bins, patches = ax.hist(volttrace, bins=num_bins, normed=True,
                fc="None")

        ax.set_xlim(volttrace.min(), volttrace.max())

        v_thresh = self.db_params.v_thresh
        v_p05 = self.db_calibration.v_p05
        if plot_vlines:
            ax.axvline(v_thresh, ls="--", label="$v_{thresh}$", c="r")
            ax.axvline(v_p05, ls="--", label="$v_{p=0.5}$", c="b")

        mean, std, g_tot = self.get_vmem_dist_theo()
        max_bin = counts.max()

        ax.axvline(mean, ls="-", c="r", label="$\\bar{v}_{theo}$")
        ax.arrow(x=mean, dx=std, y=np.exp(-.5)*max_bin, dy=0.,
                label="$\\sigma_{v_{theo}}$")
        ax.arrow(x=mean, dx=-std, y=np.exp(-.5)*max_bin, dy=0.)

        ax.ticklabel_format(axis="x", style='sci', useOffset=False)

        ax.set_xlabel("$V_{mem}$")
        ax.set_ylabel("$p(V_{mem,free})$")

        ax.legend(bbox_to_anchor=(0.35, 1.))

    @meta.plot_function("free_vmem_autocorr")
    def plot_free_vmem_autocorr(self, max_step_diff=1000, fig=None, ax=None):
        """
            Plot the autocorrelation of the free membrane potential.

            max_step_diff: What is the maximum difference in steps for which
                           the autocorrelation should be calculated.
        """
        assert self.has_free_vmem_trace
        autocorr = cutils.autocorr(self.free_vmem_trace, max_step_diff)

        ax.plot(autocorr)

        ax.set_xlabel("$\Delta$ T [steps]")
        ax.set_ylabel("Correlation")
        log.info("Done")


    ###########################
    # INTERNALLY USED METHODS #
    ###########################

    def _calc_factor_weights_theo_to_bio(self, is_excitatory, tau):
        mean, std, g_tot = self.get_vmem_dist_theo()
        cm = self.db_params.cm

        if self.pynn_model == "IF_cond_exp":
            if is_excitatory:
                delta_E = self.db_params.e_rev_E - mean
            else:
                delta_E = mean - self.db_params.e_rev_I
            # from minimization of L2(PSP-rect) -> no more blue sky!!! (comment
            # from v1 code, --obreitwi, 19-12-13 19:44:27)
            factor = self.alpha_fitted * self.db_params.g_l\
                    / self.db_calibration.g_tot /\
                (delta_E / (cm - g_tot * tau) *\
                    (- cm / g_tot * (np.exp(- tau * g_tot / cm)-1.)\
                        + tau * (np.exp(-1.) - 1.)\
                    )\
                )

        elif self.pynn_model == "IF_curr_exp":
            factor = self.alpha_fitted /\
                (1. / (cm - g_tot * tau) *\
                    (- cm / g_tot * (np.exp(-tau*g_tot/cm) - 1.)\
                        + tau * (np.exp(-1.) - 1.)
                    )
                )
        return factor

    def _calc_distribution_theo(self):
        dbc = self.db_calibration

        dbc.mean, dbc.std, dbc.g_tot = self.get_vmem_dist_theo()
        if not self.silent:
            log.info(u"Theoretical membrane distribution: {:.3f}±{:.3f}mV".format(
            dbc.mean, dbc.std))

    def _estimate_alpha(self):
        dbc = self.db_calibration
        # estimate for syn weight factor from theo to LIF
        dbc.alpha_theo = .25 * np.sqrt(2. * np.pi) * dbc.std

    def _load_sources(self):
        assert(self.is_calibrated)
        self.db_sources = list(db.SourceCFG.select()\
                .join(db.SourceCFGInCalibration).join(db.Calibration)\
                .where(db.Calibration.id == self.db_calibration).naive())

    def _ensure_model_is_supported(self, pynn_neuron_model=None):
        if pynn_neuron_model is None:
            pynn_neuron_model = self.pynn_model
        if pynn_neuron_model not in self.supported_pynn_neuron_models:
            raise Exception("Neuron model not supported!")

