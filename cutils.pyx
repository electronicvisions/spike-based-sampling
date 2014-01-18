
from cpython cimport bool

import numpy as np
cimport numpy as np

import time

cimport cython

__all__ = [
        "get_marginal",
        "get_partition",
    ]

ctypedef unsigned int uint

cdef double get_prob(
        np.ndarray[np.int_t, ndim=1] state,
        np.ndarray[np.float64_t, ndim=2] weights,
        np.ndarray[np.float64_t, ndim=1] bias
    ):
    cdef double prob = np.exp(.5 * state.T.dot(weights.dot(state))\
            + bias.dot(state))
    return prob


@cython.boundscheck(False) 
cdef bool check_value_in_array(int value, np.ndarray[np.int_t, ndim=1] vector):
    for i in range(vector.shape[0]):
        if vector[i] == value:
            return True
    return False


@cython.boundscheck(False) 
cdef double get_partition_for_active(
        np.ndarray[np.int_t, ndim=1] state,
        np.ndarray[np.float64_t, ndim=2] weights,
        np.ndarray[np.float64_t, ndim=1] biases,
        np.ndarray[np.int_t, ndim=1] active,
        uint current_idx=0
    ):
    # just skip active samplers, their values are fixed
    if check_value_in_array(current_idx, active):
        if current_idx == state.shape[0] - 1:
            # if we are at the last sampler, just return the probability for
            # this state
            return get_prob(state, weights, biases)
        else:
            # else just skip it
            return get_partition_for_active(
                    state, weights, biases, active, current_idx+1)

    cdef uint local_state
    cdef double partition = 0.

    if current_idx < state.shape[0] - 1:
        for local_state in range(2):
            state[current_idx] = local_state
            partition += get_partition_for_active(
                    state, weights, biases, active, current_idx+1)
    else:
        for local_state in range(2):
            state[current_idx] = local_state
            partition += get_prob(state, weights, biases)

    return partition


@cython.boundscheck(False) 
def get_partition(np.ndarray[np.float64_t, ndim=2] weights,
                 np.ndarray[np.float64_t, ndim=1] biases):
    assert weights.shape[0] == weights.shape[1], "Weights must be quadratic"
    assert weights.shape[0] == biases.shape[0], "Biases and weights must match"

    cdef uint num_vars = weights.shape[0]

    # no active indices
    cdef np.ndarray[np.int_t, ndim=1] no_active = np.zeros((0,), dtype=np.int)
    cdef np.ndarray[np.int_t, ndim=1] state = np.zeros((num_vars,),dtype=np.int)

    cdef double partition = get_partition_for_active(
            state, weights, biases, no_active)

    return partition


@cython.boundscheck(False) 
def get_marginal(np.ndarray[np.float64_t, ndim=2] weights,
                 np.ndarray[np.float64_t, ndim=1] biases):
    """
        The diagonal of weights should contain the biases.
    """
    assert weights.shape[0] == weights.shape[1], "Weights must be quadratic"
    assert weights.shape[0] == biases.shape[0], "Biases and weights must match"

    cdef uint num_vars = weights.shape[0]
    cdef np.ndarray[np.int_t, ndim=1] state = np.zeros((num_vars,), dtype=np.int)
    cdef np.ndarray[np.float64_t, ndim=1] probs = np.zeros((num_vars,),
            dtype=np.float64)
    cdef np.ndarray[np.int_t, ndim=1] active = np.zeros((1,), dtype=np.int)

    cdef uint i,j

    cdef double partition = get_partition(weights, biases)

    for i in range(num_vars):
        active[0] = i
        state[i] = 1
        probs[i] = get_partition_for_active(
                state,
                weights,
                biases,
                active,
                0
                ) / partition

        return probs

