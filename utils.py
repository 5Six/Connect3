import numpy as np


def search_sequence_numpy(arr, seq):
    Na, Nseq = arr.size, seq.size

    r_seq = np.arange(Nseq)

    M = (arr[np.arange(Na-Nseq+1)[:, None] + r_seq] == seq).all(1)

    return np.any(M)


def print_grid(grid):
    print(grid[::-1, ])



