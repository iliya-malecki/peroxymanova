import numpy as np
from numpy.typing import NDArray

def permanova(
    sqdistances: NDArray[np.float64], labels: NDArray[np.int_]
) -> tuple[float, float]: ...
