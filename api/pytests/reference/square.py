from typing import Sequence, Collection
from numba import njit
import numpy as np

@njit
def get_ss_w(sqdistances: np.ndarray, labels: np.ndarray, bc: np.ndarray):
    sum = np.zeros_like(bc, dtype=np.float64)
    for i in range(sqdistances.shape[0]):
        for j in range(sqdistances.shape[0]):
            if labels[i] == labels[j] and i != j:
                sum[labels[i]] += sqdistances[i, j]
    return np.sum(sum/bc/2)


def get_ss_t(sqdistances: np.ndarray):
    sum = np.float64(0)
    for i in range(sqdistances.shape[0]):
        for j in range(sqdistances.shape[0]):
            if i != j:
                sum += sqdistances[i, j]

    return sum/sqdistances.shape[0]/2


def get_f(ss_t: np.float64, ss_w: np.float64, a: int, n: int):
    ss_a = ss_t - ss_w
    f = (ss_a/(a-1))/(ss_w/(n-a))
    return f


def permanova(sqdistances: np.ndarray, fastlabels: np.ndarray):

    bc = np.bincount(fastlabels)
    ss_t = get_ss_t(sqdistances)
    ss_w = get_ss_w(sqdistances, fastlabels, bc)

    f = get_f(ss_t, ss_w, len(bc), len(fastlabels))
    other_fs = []
    for _ in range(1000):
        np.random.shuffle(fastlabels)
        other_fs.append(
            get_f(
                ss_t,
                get_ss_w(sqdistances, fastlabels, bc),
                len(bc),
                len(fastlabels)
            )
        )

    return f, (np.array(other_fs) >= f).mean()
