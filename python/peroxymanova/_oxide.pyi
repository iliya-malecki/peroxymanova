import numpy as np
from typing import Any

def permanova(
    sqdistances: np.ndarray[Any, np.dtype[np.float64]],
    labels: np.ndarray[Any, np.dtype[np.uint]],
) -> tuple[float, float]: ...
def ordinal_encoding(
    arr: np.ndarray[Any, np.dtype[np.str_]],
) -> np.ndarray[Any, np.dtype[np.uint]]: ...
