
from cpython cimport bool

import numpy as np
cimport numpy as np

import time

cimport cython

ctypedef unsigned long uint

cdef inline double get_bm_prob(
        uint num_dims,
        long* state,
        double* weights,
        double* biases
    ):
    cdef uint i, j
    cdef double exponent = 0.
    # print "State (pre prob)",
    # for i in range(num_dims):
        # print state[i],
    # print ""
    for i in range(num_dims):
        for j in range(num_dims):
            exponent += .5 * state[i] * weights[i*num_dims+j] * state[j]
        exponent += state[i] * biases[i]
    return np.exp(exponent)


cdef bool check_value_in_uint_array(
        uint value, uint num_vector_elem, uint* vector):
    for i in range(num_vector_elem):
        if vector[i] == value:
            return True
    return False

cdef bool check_value_in_long_array(
        uint value, uint num_vector_elem, long* vector):
    for i in range(num_vector_elem):
        if vector[i] == value:
            return True
    return False

@cython.boundscheck(False)
cdef double get_bm_partition_theo_for_fixed(
        np.ndarray[np.int_t, ndim=1] state,
        np.ndarray[np.float64_t, ndim=2] weights,
        np.ndarray[np.float64_t, ndim=1] biases,
        np.ndarray[np.int_t, ndim=1] fixed,
    ):
    cdef uint num_dims = state.shape[0]
    cdef uint num_fixed = fixed.shape[0]
    # just extract the arguments
    return _get_bm_partition_theo_for_fixed(
            num_dims,
            <long*> state.data,
            <double*> weights.data,
            <double*> biases.data,
            num_fixed,
            <uint*> fixed.data,
        )


# returns whether the state looped around
cdef inline bool advance_state(
        uint num_dims,
        long* state,
        uint num_fixed,
        uint* fixed,
    ):
    cdef long carry = 0
    cdef long add = 1

    cdef long i
    for i in range(num_dims-1, -1, -1):
        if num_fixed > 0 and check_value_in_uint_array(i, num_fixed, fixed):
            continue

        carry = (state[i] + add) == 2
        state[i] = state[i] ^ add
        add = carry

    # if the carry contains 1 after we looped over all dimensions, we fully
    # wrapped around and reached the beginning state again
    return carry > 0

cdef inline double _get_bm_partition_theo_for_fixed(
        uint num_dims,
        long* state,
        double* weights,
        double* biases,
        uint num_fixed,
        uint* fixed,
    ):
    cdef uint i

    # init the state variable to be zero everywhere it isnt fixed
    for i in range(num_dims):
        if num_fixed > 0 and check_value_in_uint_array(i, num_fixed, fixed):
            continue
        else:
            state[i] = 0

    cdef double partition = 0.
    cdef bool final_state_reached = False

    while True:
        partition += get_bm_prob(num_dims, state, weights, biases)

        # advance the state
        if advance_state(num_dims, state, num_fixed, fixed):
            break

    return partition


@cython.boundscheck(False)
def get_bm_partition_theo(np.ndarray[np.float64_t, ndim=2] weights,
                 np.ndarray[np.float64_t, ndim=1] biases):
    assert weights.shape[0] == weights.shape[1], "Weights must be quadratic"
    assert weights.shape[0] == biases.shape[0], "Biases and weights must match"

    cdef uint num_dims = weights.shape[0]

    # no fixed indices
    cdef np.ndarray[np.int_t, ndim=1] no_fixed = np.zeros((0,), dtype=np.int)
    cdef np.ndarray[np.int_t, ndim=1] state = np.zeros((num_dims,),
            dtype=np.int)

    cdef double partition = get_bm_partition_theo_for_fixed(
            state, weights, biases, no_fixed)

    return partition


@cython.boundscheck(False)
def get_bm_marginal_theo(np.ndarray[np.float64_t, ndim=2] weights,
                 np.ndarray[np.float64_t, ndim=1] biases,
                 np.ndarray[np.int_t, ndim=1] selected_idx):
    """
        Get theoretical marginal distribution for Boltzmann distribution.

        This does not calculate the joint probability explicitly and is
        therefore able to compute the marginal for higher dimensions.
    """
    assert weights.shape[0] == weights.shape[1], "Weights must be quadratic"
    assert weights.shape[0] == biases.shape[0], "Biases and weights must match"

    cdef uint num_selected = selected_idx.shape[0]

    cdef uint num_dims = weights.shape[0]
    cdef np.ndarray[np.int_t, ndim=1] state = np.zeros((num_dims,),
            dtype=np.int)
    cdef np.ndarray[np.int_t, ndim=1] fixed = np.zeros((1,), dtype=np.int)

    cdef np.ndarray[np.float64_t, ndim=1] probs = np.zeros((num_selected,),
            dtype=np.float64)

    cdef uint i,j

    cdef double partition = get_bm_partition_theo(weights, biases)

    for i in range(num_selected):
        fixed[0] = selected_idx[i]
        state[selected_idx[i]] = 1
        probs[i] = get_bm_partition_theo_for_fixed(
                state,
                weights,
                biases,
                fixed
                ) / partition

    return probs


@cython.boundscheck(False)
def get_bm_joint_theo(np.ndarray[np.float64_t, ndim=2] weights,
                 np.ndarray[np.float64_t, ndim=1] biases):
    """
        Get theoretical marginal distribution for Boltzmann distribution.
    """
    assert weights.shape[0] == weights.shape[1], "Weights must be quadratic"
    assert weights.shape[0] == biases.shape[0], "Biases and weights must match"

    cdef uint num_dims = weights.shape[0]
    cdef uint i
    cdef uint num_total = (1 << num_dims)

    cdef double* weights_ptr = <double*> weights.data
    cdef double* biases_ptr = <double*> biases.data

    cdef np.ndarray[np.float64_t, ndim=1] joints = np.zeros((num_total,),
            dtype=np.float64)
    cdef np.ndarray[np.int_t, ndim=1] state = np.zeros((num_dims,),
            dtype=np.int)

    cdef long* state_ptr = <long*> state.data

    for i in range(num_total):
        joints[i] = get_bm_prob(num_dims, state_ptr, weights_ptr, biases_ptr)
        advance_state(num_dims, state_ptr, 0, NULL)

    joints /= joints.sum()

    return joints.reshape([2 for i in range(num_dims)])


cdef inline uint get_current_state(uint num_selected, double* tau_sampler_ptr):

    cdef uint current_state = 0

    for i in range(num_selected):
        if tau_sampler_ptr[i] > 0.:
            current_state += (1 << (num_selected - 1 - i))

    return current_state


def autocorr(np.ndarray[np.float64_t, ndim=1] array, uint max_step_diff):
    cdef np.ndarray[np.float64_t, ndim=1] joints = np.zeros((max_step_diff,),
            dtype=np.float64)

    cdef uint i

    joints[0] = 1.

    for i in range(1,max_step_diff):
        joints[i] = np.corrcoef(array[:-i], array[i:])[0,1]

    return joints

@cython.boundscheck(False)
def get_bm_joint_sim(
        np.ndarray[np.int_t, ndim=1] spike_ids,
        np.ndarray[np.float64_t, ndim=1] spike_times,
        np.ndarray[np.int_t, ndim=1] sampler_idx,
        np.ndarray[np.float64_t, ndim=1] tau_refrac_pss, # per selected sampler
        double duration,
    ):
    sampler_idx.sort()
    assert spike_ids.shape[0] == spike_times.shape[0]

    cdef double current_time = 0.
    cdef uint i_spike = 0
    cdef uint i, sampler_id
    cdef double next_inactivation, next_spike, time_step
    cdef uint num_spikes = spike_ids.shape[0]

    cdef uint num_selected = sampler_idx.shape[0]
    cdef long* sampler_idx_ptr = <long*> sampler_idx.data 

    cdef np.ndarray[np.float64_t, ndim=1] tau_sampler =\
            np.zeros((num_selected,), dtype=np.float64)
    cdef double* tau_sampler_ptr = <double*> tau_sampler.data

    cdef uint num_total = (1 << num_selected)
    cdef np.ndarray[np.float64_t, ndim=1] joints = np.zeros((num_total,),
            dtype=np.float64)

    cdef bool is_spike

    # since getting the joint for too many samplers is infeasable
    # we can store 
    cdef uint current_state

    # skip all spikes from samplers we do not care about
    while i_spike < num_spikes and not check_value_in_long_array(
            spike_ids[i_spike], num_selected, sampler_idx_ptr):
        i_spike += 1

    while current_time < duration:

        # find next activation
        next_inactivation = np.inf
        for i in range(num_selected):
            if tau_sampler_ptr[i] > 0.\
                    and tau_sampler_ptr[i] < next_inactivation:
                next_inactivation = tau_sampler_ptr[i]

        if i_spike < num_spikes:
            next_spike = spike_times[i_spike] - current_time
        else:
            next_spike = duration - current_time

        # check out if the next event is a spike or a simple inactivation of
        # a sampler
        if next_inactivation > next_spike:
            is_spike = i_spike < num_spikes
            time_step = next_spike

        else:
            is_spike = False
            time_step = next_inactivation

        current_state = get_current_state(num_selected, tau_sampler_ptr)

        # note that the current state is on for the next time
        joints[current_state] += time_step

        for i in range(num_selected):
            tau_sampler[i] -= time_step

        if is_spike:
            sampler_id = spike_ids[i_spike]

            for i in range(num_selected):
                if sampler_idx_ptr[i] == sampler_id:
                    sampler_id = i
                    break
            # adjust current spike
            tau_sampler[sampler_id] = tau_refrac_pss[sampler_id]

            # find next spike
            i_spike += 1
            # skip all spikes from samplers we do not care about
            while i_spike < num_spikes and not check_value_in_long_array(
                    spike_ids[i_spike], num_selected, sampler_idx_ptr):
                i_spike += 1

        current_time += time_step
        # print "tau",
        # for i in range(num_selected):
            # print tau_sampler_ptr[i],
        # print ""


    joints /= duration
    return joints.reshape([2 for i in range(num_selected)])


cdef _gibbs_sampling(
            np.ndarray[np.int_t, ndim=2] init_state,
            np.ndarray[np.int_t, ndim=2] weights,
            np.ndarray[np.int_t, ndim=1] biases,
            np.ndarray[np.int_t, ndim=1] selected_idx,
            bool compute_joint,
            bool record_z_vectors,
            uint num_steps,
            ):

    cdef uint num_chains = init_state.shape[0]
    cdef uint num_samplers = init_state.shape[1]

    for i_step in range(num_steps):
        pass

