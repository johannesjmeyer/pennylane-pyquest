import numpy as np
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


def reorder_state(state):
    N = len(state)

    return state[reversed_indices(N - 1)]
