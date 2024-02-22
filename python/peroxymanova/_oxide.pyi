import numpy as np
from typing import Any

def permanova(
    sqdistances: np.ndarray[Any, np.dtype[np.float64]],
    labels: np.ndarray[Any, np.dtype[np.uint]],
    permutations=1000,
) -> tuple[float, float]:
    """
    ### Notes
    Run a highly optimized permanova implementation written in rust and compiled for your architecture.
    The question the algorithm answers is "are these observations from different
    groups coming from the same distribution?", or, in simpler terms,
    "is there a real difference between these groups?"
    This algorithm relies on the fact that the sum of squared distances to
    a centroid of a group of measurements equals the average distance between
    every measurement in that group. To see the full explanation refer to the original paper.

    ### Parameters

    sqdistances: a 2d distance matrix that is already squared (as in, literally, x**2 kind of squared)

    labels: a 1d array of uint category labels, where a given `labels[i]`\
        refers to a category label of `sqdistances[i,:]` and `sqdistances[:, i]`

    permutations: amount of permutations for calculating p-value

    ### Returns
    The first float of the tuple is the statistic, a deterministic number that represents how
    extreme the difference between groups is. It is kind of impossible to interpret
    but its useful to compare different runs on the same data.
    The second float is the p-value, a permutation-based approximation of the
    probability that the null hypothesis should be accepted. Since the null hypothesis here
    is that there is no difference between groups, a low p-value (typically below 0.05)
    means that groups are likely different.
    """

ordinal_encoding_dtypes = np.str_ | np.int64 | np.int32 | np.int16
