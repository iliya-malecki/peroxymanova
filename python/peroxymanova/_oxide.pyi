import numpy as np
from typing import Any

def permanova(
    sqdistances: np.ndarray[np.float64, Any], labels: np.ndarray[np.uint, Any]
) -> tuple[float, float]: ...
def ordinal_encoding(arr: np.ndarray[np.str_, Any]) -> np.ndarray[np.uint, Any]: ...
