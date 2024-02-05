import numpy as np
from typing import Any

def permanova(
    sqdistances: np.ndarray[np.float64, Any], labels: np.ndarray[np.int_, Any]
) -> tuple[float, float]: ...
