#!/usr/bin/env python
# encoding: utf-8

from .logcfg import log
from . import utils
from . import db
from . import fit
from . import meta
from . import buildingblocks as bb
from . import cutils
from . import cells

import itertools as it
import numpy as np


@meta.HasDependencies
class LIFsampler(object):

    supported_pynn_neuron_models = [
            "IF_curr_exp",
            "IF_cond_exp",
            # "IF_cond_exp_cd",
            # "IF_curr_exp_cd",
            "IF_curr_alpha",
            "IF_cond_alpha",
            # "IF_cond_alpha_cd",
            # "IF_curr_alpha_cd",
        ]

    def __init__(self, sampler_config, sim_name="pyNN.nest", silent=False):
        """
            sampler_config:
                sbs.db.NeuronParameters or sbs.db.SamplerConfiguration
        """
        self.silent = silent

        if not self.silent:
            log.info("Setting up sampler.")

        self.sim_name = sim_name

        if isinstance(sampler_config, db.SamplerConfiguration):
            self.calibration = sampler_config.calibration
            self.neuron_parameters = sampler_config.neuron_parameters
            self.source_config = sampler_config.source_config

        elif isinstance(sampler_config, db.NeuronParameters):
            self.neuron_parameters = sampler_config
            self.calibration = None

        else:
            raise Exception("Invalid sampler_config supplied.")

        self._ensure_model_is_supported(self.neuron_parameters.pynn_model)

        self.bias_theo = 0.

        self.free_vmem = None

        # population or population slice describing this particular
        self.population = None

        # the network in which this sampler is embedded, this is set from the
        # Boltzmann Machines
        self.network = {'population': None, 'index': None}

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
            return self.bias_bio / self.calibration.fit.alpha
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
            return self.bias_theo_to_bio(self.bias_theo)
        else:
            return value

    def bias_theo_to_bio(self, bias):
        return bias * self.calibration.fit.alpha

    def bias_bio_to_theo(self, bias):
        return bias / self.calibration.alpha

    def sync_bias_to_pynn(self):
        assert self.is_created
        self.population.set(v_rest=self.get_v_rest_from_bias())

    @meta.DependsOn()
    def dist_theo(self, value=None):
        return value

    @meta.DependsOn("dist_theo")
    def alpha_theo(self):
        # estimate for syn weight factor from theo to LIF
        alpha_theo = .25 * np.sqrt(2. * np.pi) * self.dist_theo.std
        return alpha_theo

    @meta.DependsOn()
    def calibration(self, value=None):
        """
            The database calibration object.
        """
        return value

    @meta.DependsOn()
    def source_config(self, value=None):
        """
            Can only be set by modifying the calibration.
        """
        if value is None:
            return self.calibration.source_config
        else:
            return value

    @property
    def is_calibrated(self):
        return self.calibration is not None

    @property
    def pynn_model(self):
        return self.neuron_parameters.pynn_model

    @property
    def has_free_vmem_trace(self):
        return self.free_vmem is not None

    def get_pynn_parameters(self, adjust_v_rest=True):
        """
            Returns dictionary with all needed pynn parameters to implement
            the sampler. Note: The resting potential will be set according to
            the specified bias.

            `adjust_v_rest` can be set to False to maintain the original rest
            value. This is only really useful during calibration as there are
            not values yet with which to update.
        """
        assert(self.is_calibrated)
        params = self.neuron_parameters.get_pynn_parameters()
        if adjust_v_rest:
            params["v_rest"] = self.get_v_rest_from_bias()
        return params

    def get_parameters_id(self):
        """
            Return the id of the parameters in the database.
        """
        return self.neuron_parameters.id

    def get_calibration_id(self):
        """
            Return the id of the calibration used currently.

            Returns None if the sampler has not been calibrated.
        """
        if self.is_calibrated:
            return self.calibration.id
        else:
            return None

    @property
    def is_created(self):
        return self.population is not None

    def forget_calibration(self):
        """
            Unsets the calibration for this sampler.
        """
        self.calibration = None

    def get_v_rest_from_bias(self, bias=None):
        assert(self.is_calibrated)
        if bias is None:
            return self.calibration.fit.v_p05 + self.bias_bio
        else:
            return self.calibration.fit.v_p05 + self.bias_theo_to_bio(bias)

    def calibrate(self, calibration=None, **pre_calibration_parameters):
        """
            Calibrate the sampler, using the configuration from the provided
            calibration object.

            If no calibration object is given, self.calibration will be used.
            In this case no pre-calibration is performed.

            pre_calibration_parameters can be used to alter the parameters of
            the initial slope search (see sbs.db.PreCalibration).

            Valid pre-calibration parameters are:
                - lower_bound
                - upper_bound
                - V_rest_min
                - V_rest_max
                - dV
        """
        if calibration is None:
            do_pre_calibration = False
            calibration = self.calibration
        else:
            do_pre_calibration = True

        calibration.sim_name = self.sim_name

        # by importing here we avoid importing networking stuff until we have to
        from .gather_data import gather_calibration_data

        if do_pre_calibration:
            final_pre_calib = self._do_pre_calibration(calibration,
                    **pre_calibration_parameters)

            # copy the final V_rest ranges
            calibration.V_rest_min = final_pre_calib.V_rest_min
            calibration.V_rest_max = final_pre_calib.V_rest_max

        log.info("Taking {} samples from {:.3f}mV to {:.3f}mV…".format(
                calibration.num_samples,
                calibration.V_rest_min,
                calibration.V_rest_max
            ))

        # do final, proper calibration
        calibparams = db.SamplerConfiguration(
                calibration=calibration,
                neuron_parameters=self.neuron_parameters)

        self.calibration = calibration
        self.calibration.samples_p_on = gather_calibration_data(calibparams)

        if not self.silent:
            log.info("Calibration data gathered, performing fit.")

        self._calc_distribution_theo()

        # self._estimate_alpha()

        self.calibration.fit = db.Fit()
        self.calibration.fit.v_p05, self.calibration.fit.alpha =\
            fit.fit_sigmoid(
                self.calibration.get_samples_v_rest(),
                self.calibration.samples_p_on,
                guess_p05=self.neuron_parameters.v_thresh,
                guess_alpha=self.alpha_theo)


        if not self.silent:
            log.info("Fitted alpha: {:.3f}".format(self.calibration.fit.alpha))
            log.info("Fitted v_p05: {:.3f} mV".format(
                self.calibration.fit.v_p05))

    def measure_free_vmem_dist(self, duration=100000., dt=0.1, burn_in_time=200.):
        """
            Measure the distribution of the free membrane potential, given
            the parameters (attributes of VmemDistribution).
        """
        assert self.is_calibrated

        from .gather_data import gather_free_vmem_trace

        self.free_vmem = {
                "trace" : gather_free_vmem_trace(
                    distribution_params={
                        "duration": duration,
                        "dt": dt,
                        "burn_in_time": burn_in_time,
                        },
                    sampler=self),
                "dt" : dt
            }

    def get_calibration_source_parameters(self):
        """
            Returns a tuple of `np.array`s with calibration source configuration
                (rates_exc, rates_inh, weights_exc, weights_inh)
        """
        src_cfg = self.calibration.source_config
        is_exc = src_cfg.weights > 0.
        is_inh = np.logical_not(is_exc)

        rates_exc = src_cfg.rates[is_exc]
        rates_inh = src_cfg.rates[is_inh]
        weights_exc = src_cfg.weights[is_exc]
        weights_inh = src_cfg.weights[is_inh]

        return rates_exc, rates_inh, weights_exc, weights_inh

    def get_vmem_dist_theo(self):
        dist = getattr(utils, "{}_distribution".format(self.pynn_model))
        # TODO:
        args = self.get_calibration_source_parameters()
        if self.calibration.fit is None or  not self.calibration.fit.is_valid():
            if not self.silent:
                log.info("Computing vmem distribution ONLY from supplied neuron "
                     "parameters!")
            kwargs = self.neuron_parameters.get_pynn_parameters()
        else:
            if not self.silent:
                log.info("Computing vmem distribution with bias set to {}.".format(
                self.bias_theo))
            kwargs = self.get_pynn_parameters()

        # g_l is not a pynn parameter
        kwargs["g_l"] = self.neuron_parameters.g_l

        return dist(*args, **kwargs)

    # this is just kept for now to make sure the other factor is correct --obreitwi, 14-02-14 20:45:13
    # def factor_weights_bio_to_theo(self):
        # if self.pynn_model == "IF_cond_exp":
            # # from minimization of L2(PSP-rect) -> no more blue sky!!!
            # # (original comment from v1 code --obreitwi, 19-12-13 21:10:08)
            # delta_E = np.array([
                    # self.calibration.mean - self.neuron_params.e_rev_I,
                    # self.neuron_params.e_rev_E - self.calibration.mean
                # ])
            # theo_weights = weights * delta_E[is_exc_int] /\
                # (cm - g_tot * tau[is_exc_int]) *\
                # (- cm / g_tot * (np.exp(- tau[is_exc_int] * g_tot / cm) - 1.)\
                    # + tau[is_exc_int] * (np.exp(-1.) - 1.)
                # ) / self.calibration.fit.alpha * g_tot / self.neuron_params.g_l
        # elif self.pynn_model == "IF_curr_exp":
            # theo_weights = weights / (cm - g_tot * tau[is_exc_int]) *\
                # (- cm / g_tot * (np.exp(- tau[is_exc_int] * g_tot / cm) - 1.)\
                    # + tau[is_exc_int] * (np.exp(-1.) - 1.)
                # ) / self.alpha_fiited

    @meta.DependsOn("calibration", "bias_theo", "bias_bio")
    def factor_weights_theo_to_bio_exc(self):
        return self._calc_factor_weights_theo_to_bio(
                is_excitatory=True,
                tau=self.neuron_parameters.tau_syn_E
            )

    @meta.DependsOn("calibration", "bias_theo", "bias_bio")
    def factor_weights_theo_to_bio_inh(self):
        mean, std, g_tot, tau_eff = self.get_vmem_dist_theo()
        return self._calc_factor_weights_theo_to_bio(
                is_excitatory=False,
                tau=self.neuron_parameters.tau_syn_I
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

    def write_config(self, filename):
        if not self.silent:
            log.info("Writing sampler configuration…")
        db.SamplerConfiguration(calibration=self.calibration,
            neuron_parameters=self.neuron_parameters).write(filename)


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
            self.source_config.create_connect(
                    sim, [self], duration=duration)

        return self.population

    ##################
    # PLOT FUNCTIONS #
    ##################

    @meta.plot_function("calibration")
    def plot_calibration(self, plot_v_dist_theo=False, plot_vlines=True,
            fig=None, ax=None):
        assert self.is_calibrated

        self._calc_distribution_theo()

        samples_v_rest = self.calibration.get_samples_v_rest()
        samples_p_on = self.calibration.samples_p_on

        v_thresh = self.neuron_parameters.v_thresh
        v_p05 = self.calibration.fit.v_p05
        std = self.dist_theo.std

        xdata = np.linspace(v_thresh-4.*std, v_thresh+4.*std, 500)

        if plot_v_dist_theo:
            estim_dist_v_thresh = utils.gauss(xdata, v_thresh, std)
        estim_cdf_v_thresh = utils.erfm(xdata, v_thresh, std)
        estim_sigmoid = utils.sigmoid_trans(xdata, v_thresh, self.alpha_theo)

        fitted_p_on = utils.sigmoid_trans(samples_v_rest, v_p05,
                self.calibration.fit.alpha)

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

    @meta.plot_function("calibration_residuals")
    def plot_calibration_residuals(self, plot_v_dist_theo=False, plot_vlines=True,
            width=4., npoints=500, fig=None, ax=None):
        assert self.is_calibrated

        self._calc_distribution_theo()

        samples_v_rest = self.calibration.get_samples_v_rest()
        samples_p_on = self.calibration.samples_p_on

        v_thresh = self.neuron_parameters.v_thresh
        v_p05 = self.calibration.fit.v_p05
        std = self.dist_theo.std

        xdata = np.linspace(v_thresh-width*std, v_thresh+width*std, npoints)


        fitted_p_on = utils.sigmoid_trans(samples_v_rest, v_p05,
                self.calibration.fit.alpha)

        ax.plot(samples_v_rest, samples_p_on-fitted_p_on, label="residuals")

        ax.set_xlabel("$V_{rest}$")
        ax.set_ylabel("$p_{ON}$")

        ax.legend(bbox_to_anchor=(1.15, .5))

    @meta.plot_function("free_vmem_dist")
    def plot_free_vmem(self, num_bins=200, plot_vlines=True, fig=None, ax=None):
        assert self.has_free_vmem_trace
        assert self.is_calibrated

        volttrace = self.free_vmem["trace"]

        counts, bins, patches = ax.hist(volttrace, bins=num_bins, normed=True,
                fc="None")

        ax.set_xlim(volttrace.min(), volttrace.max())

        v_thresh = self.neuron_parameters.v_thresh
        v_p05 = self.calibration.fit.v_p05
        if plot_vlines:
            ax.axvline(v_thresh, ls="--", label="$v_{thresh}$", c="r")
            ax.axvline(v_p05, ls="--", label="$v_{p=0.5}$", c="b")

        mean, std, g_tot, tau_eff = self.get_vmem_dist_theo()
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
        autocorr = cutils.autocorr(self.free_vmem["trace"], max_step_diff)

        ax.plot(np.arange(1, max_step_diff+1)
                * self.free_vmem["dt"], autocorr)

        ax.set_xlabel("$\Delta$ T [ms]")
        ax.set_ylabel("Correlation")
        log.info("Done")


    ###########################
    # INTERNALLY USED METHODS #
    ###########################

    def _calc_factor_weights_theo_to_bio(self, is_excitatory, tau):
        mean, std, g_tot, tau_eff = self.get_vmem_dist_theo()
        cm = self.neuron_parameters.cm

        if self.pynn_model.startswith("IF_cond_exp"):
            tau_r = self.neuron_parameters.tau_refrac
            if is_excitatory:
                delta_E = self.neuron_parameters.e_rev_E - mean
            else:
                delta_E = mean - self.neuron_parameters.e_rev_I
            # from minimization of L2(PSP-rect) -> no more blue sky!!! (comment
            # from v1 code, --obreitwi, 19-12-13 19:44:27)
            factor = self.calibration.fit.alpha * self.neuron_parameters.g_l\
                    / g_tot * tau_r / tau /\
                (delta_E / (cm - g_tot * tau) *\
                    (- cm / g_tot * (np.exp(- tau_r * g_tot / cm)-1.)\
                        + tau * (np.exp(-tau_r / tau) - 1.)\
                    )\
                )

        elif self.pynn_model.startswith("IF_curr_exp"):
            tau_r = self.neuron_parameters.tau_refrac
            factor = self.calibration.fit.alpha * tau_r / tau /\
                (1. / (cm - g_tot * tau) *\
                    (- cm / g_tot * (np.exp(-tau_r*g_tot/cm) - 1.)\
                        + tau * (np.exp(-tau_r / tau) - 1.)
                    )
                )

        elif self.pynn_model.startswith("IF_cond_alpha"):
            tau_r = self.neuron_parameters.tau_refrac
            if is_excitatory:
                delta_E = self.neuron_parameters.e_rev_E - mean
            else:
                delta_E = mean - self.neuron_parameters.e_rev_I

            tau_c = 1. / (1. / tau - 1. / tau_eff)

            factor = -self.calibration.fit.alpha * self.neuron_parameters.g_l\
                    / np.exp(1) / tau_c * tau_r * tau * tau_eff / delta_E / (
                          tau**2 * (1- np.exp(-tau_r/tau))
                        - tau_r * tau * np.exp(-tau_r/tau)
                        + tau_c * (
                              tau_eff * (np.exp(-tau_r/tau_eff)- 1 )
                            - tau * (np.exp(-tau_r / tau) - 1)
                          )
                    )

        elif self.pynn_model.startswith("IF_curr_alpha"):
            tau_r = self.neuron_parameters.tau_refrac
            tau_m = self.neuron_parameters.tau_m
            tau_c = 1. / (1. / tau - 1. / tau_m)

            factor = -self.calibration.fit.alpha * self.neuron_parameters.g_l\
                    / np.exp(1) / tau_c * tau * tau_m * tau_r / (
                          tau**2 * (1- np.exp(-tau_r/tau))
                        - tau_r * tau * np.exp(-tau_r/tau)
                        + tau_c * (
                              tau_m * (np.exp(-tau_r/tau_m) - 1 )
                            - tau * (np.exp(-tau_r / tau) - 1)
                          )
                    )

        return factor

    def _calc_distribution_theo(self):
        dbc = self.dist_theo = db.VmemDistribution()

        dbc.mean, dbc.std, dbc.g_tot, dbc.tau_eff = self.get_vmem_dist_theo()
        if not self.silent:
            log.info(u"Theoretical Vmem distribution: {:.3f}±{:.3f}mV".format(
            dbc.mean, dbc.std))

    def _ensure_model_is_supported(self, pynn_neuron_model=None):
        if pynn_neuron_model is None:
            pynn_neuron_model = self.pynn_model
        if pynn_neuron_model not in self.supported_pynn_neuron_models:
            raise Exception("Neuron model not supported!")

    def _do_pre_calibration(self, calibration, **pre_calibration_parameters):
        pre_calib = db.PreCalibration(
            V_rest_min=-80., V_rest_max=-20.,
            dV=0.2,
            lower_bound=0.05, upper_bound=0.95,
            duration=1000., #  time spent when scanning for the sigmoid
            max_search_steps=100,
        )
        for k in ["sim_name", "sim_setup_kwargs", "burn_in_time", "dt",
                "source_config"]:
            setattr(pre_calib, k, getattr(calibration, k))

        for k,v in pre_calibration_parameters.iteritems():
            setattr(pre_calib, k, v)

        orig_pre_calib = pre_calib.copy()

        upper_bound_found = lower_bound_found = False

        samples_v_rest = []
        samples_p_on = []

        pre_sampler_config = db.SamplerConfiguration(
                neuron_parameters=self.neuron_parameters,
                calibration=pre_calib)

        # by importing here we avoid importing networking stuff until we have to
        from .gather_data import gather_calibration_data

        V_range = pre_calib.V_rest_max - pre_calib.V_rest_min

        search_steps = 0
        while search_steps < pre_calib.max_search_steps:
            if not upper_bound_found:
                samples_p_on.append(gather_calibration_data(pre_sampler_config))
                samples_v_rest.append(pre_calib.get_samples_v_rest())

            else:
                samples_p_on.insert(0, gather_calibration_data(pre_sampler_config))
                samples_v_rest.insert(0, pre_calib.get_samples_v_rest())

            upper_bound_found = any(((spon>pre_calib.upper_bound).any()
                for spon in reversed(samples_p_on)))

            lower_bound_found = any(((spon<pre_calib.lower_bound).any()
                for spon in samples_p_on))

            if upper_bound_found and lower_bound_found:
                break

            # adjust the next search range and make sure we are scanning nothing
            # twice
            if not upper_bound_found:
                pre_calib.V_rest_min = max(orig_pre_calib.V_rest_min,
                        pre_calib.V_rest_min) + V_range
                pre_calib.V_rest_max = max(orig_pre_calib.V_rest_max,
                        pre_calib.V_rest_max) + V_range
            else:
                pre_calib.V_rest_min = min(orig_pre_calib.V_rest_min,
                        pre_calib.V_rest_min) - V_range
                pre_calib.V_rest_max = min(orig_pre_calib.V_rest_max,
                        pre_calib.V_rest_max) - V_range

            search_steps += 1

        samples_p_on = np.hstack(samples_p_on)
        samples_v_rest = np.hstack(samples_v_rest)

        idx = np.where((samples_p_on < pre_calib.upper_bound)
                * (samples_p_on > pre_calib.lower_bound))[0]

        if len(idx) > 0:
            idx_low = max(0, idx[0]-1)
            idx_high = min(samples_p_on.size-1, idx[-1]+1)

        else:
            idx_high = np.where(samples_p_on > pre_calib.upper_bound)[0][0]
            idx_low = idx_high - 1

        pre_calib.V_rest_min = samples_v_rest[idx_low]
        pre_calib.V_rest_max = samples_v_rest[idx_high]

        return pre_calib

