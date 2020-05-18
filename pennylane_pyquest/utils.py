import numpy as np
import math
import functools


def reverseBits(num, max_num):
    total_len = len(bin(max_num)) - 2
    num_bin = bin(num)[2:].zfill(total_len)
    return int(num_bin[::-1], 2)


@functools.lru_cache()
def reversed_indices(n):
    indices = []
    for i in range(n + 1):
        indices.append(reverseBits(i, n))

    return np.array(indices)


def reorder_state2(state):
    N = len(state)

    return state[reversed_indices(N - 1)]

def reorder_state(state):
    N = int(math.log2(len(state)))

    state = state.reshape([2] * N)
    state = np.moveaxis(state, list(range(N)), list(reversed(range(N))))

    return state.ravel()

def reorder_matrix(matrix):
    N = int(math.log2(matrix.shape[0]))

    matrix = matrix.reshape([2] * (2 * N))
    src = np.array(list(range(N)))
    dest = np.array(list(reversed(range(N))))
    matrix = np.moveaxis(matrix, src, dest)
    matrix = np.moveaxis(matrix, src + N, dest + N)

    return matrix.reshape((2**N, 2**N))