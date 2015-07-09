

from .logcfg import log
from . import db
from . import network
from . import utils

import numpy as np
from pprint import pformat as pf

def sample_network(database, num_samplers=5, weights=None, biases=None,
        neuron_id=1, sim_name="pyNN.nest", duration=1e5, savefilename=None,
        numpy_seed=42):
    """
        Run and plot a sample network wit the given weights and biases
    """
    db.setup(database)
    db.purge_incomplete_calibrations()

    np.random.seed(numpy_seed)

    if savefilename is not None:
        bm = network.BoltzmannMachine.load(savefilename)
    else:
        bm = None

    if bm is None:
        # no network loaded, we need to create it
        bm = network.BoltzmannMachine(num_samplers=num_samplers,
                sim_name=sim_name, neuron_parameters_db_ids=neuron_id)

        bm.load_calibration()

        if weights is None:
            weights = np.random.randn(bm.num_samplers, bm.num_samplers)
            weights = (weights + weights.T) / 2.
        bm.weights_theo = weights

        if biases is None:
            bm.biases_theo = np.random.randn(bm.num_samplers)
        else:
            bm.biases_theo = biases

        bm.saturating_synapses_enabled = True

        bm.gather_spikes(duration=duration, dt=0.1, burn_in_time=500.)
        if savefilename is not None:
            bm.save(savefilename)

    log.info("Weights (theo):\n" + pf(bm.weights_theo))
    log.info("Biases (theo):\n" + pf(bm.biases_theo))

    log.info("Weights (bio):\n" + pf(bm.weights_bio))
    log.info("Biases (bio):\n" + pf(bm.biases_bio))

    log.info("Spikes: {}".format(pf(bm.ordered_spikes)))

    log.info("Spike-data: {}".format(pf(bm.spike_data)))

    bm.selected_sampler_idx = range(bm.num_samplers)

    log.info("Marginal prob (sim):\n" + pf(bm.dist_marginal_sim))

    log.info("Joint prob (sim):\n" + pf(list(np.ndenumerate(bm.dist_joint_sim))))

    # log.info("Joint prob (sim, exclusive):\n"\
            # + pf(list(bm.dist_joint_exclusive_sim.all_items())))

    log.info("Marginal prob (theo):\n" + pf(bm.dist_marginal_theo))

    log.info("Joint prob (theo):\n"\
            + pf(list(np.ndenumerate(bm.dist_joint_theo))))

    log.info("DKL marginal: {}".format(utils.dkl_sum_marginals(
        bm.dist_marginal_theo, bm.dist_marginal_sim)))

    log.info("DKL joint: {}".format(utils.dkl(
            bm.dist_joint_theo.flatten(), bm.dist_joint_sim.flatten())))

    bm.plot_dist_marginal(save=True)
    bm.plot_dist_joint(save=True)

