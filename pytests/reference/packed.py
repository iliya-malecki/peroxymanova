from typing import Sequence, Collection
from numba import njit
import numpy as np

@njit
def get_ss_w(sqdistances: np.ndarray, labels: np.ndarray, bc: np.ndarray):
    sum = np.zeros_like(bc, dtype=np.float64)

    for d, lbs in zip(sqdistances, labels):
        if lbs[0] == lbs[1]:
            sum[lbs[0]] += d
    return np.sum(sum/bc)


def get_f(ss_t: np.float64, ss_w: np.float64, a: int, n: int):
    ss_a = ss_t - ss_w
    f = (ss_a/(a-1))/(ss_w/(n-a))
    return f


def permanova(sqdistances: np.ndarray, fastlabels: np.ndarray):

    packed_labels = np.empty(((len(fastlabels)-1)*len(fastlabels)//2, 2), dtype=np.int16)
    pc = 0
    for i in range(len(fastlabels)):
        for j in range(i+1, len(fastlabels)):
            packed_labels[pc][:] = fastlabels[i], fastlabels[j]
            pc+=1



    packed_dists = sqdistances[np.triu_indices_from(sqdistances, 1)]

    bc = np.bincount(fastlabels)
    ss_t = np.sum(packed_dists)/len(fastlabels)
    ss_w = get_ss_w(packed_dists, packed_labels, bc)

    f = get_f(ss_t, ss_w, len(bc), len(fastlabels))
    other_fs = []
    for _ in range(1000):
        np.random.shuffle(packed_dists)
        other_fs.append(
            get_f(
                ss_t,
                get_ss_w(packed_dists, packed_labels, bc),
                len(bc),
                len(fastlabels)
            )
        )

    return f, (np.array(other_fs) >= f).mean()
