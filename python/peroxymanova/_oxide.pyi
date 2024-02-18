import numpy as np
from typing import Any

def permanova(
    sqdistances: np.ndarray[Any, np.dtype[np.float64]],
    labels: np.ndarray[Any, np.dtype[np.uint]],
) -> tuple[float, float]: ...

ordinal_encoding_dtypes = np.str_ | np.int64
