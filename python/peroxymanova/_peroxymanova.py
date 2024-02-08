from __future__ import annotations
from typing import TYPE_CHECKING, TypeVar, cast
from ._oxide import permanova, ordinal_encoding
import numpy as np
from typing import NamedTuple

if TYPE_CHECKING:
    from typing import Collection, Callable, Literal, Any


class PermanovaResults(NamedTuple):
    statistic: float
    pvalue: float


T = TypeVar("T")


def get_distance_matrix(things: Collection[T], distance: Callable[[T, T], np.float64]):
    dists = np.empty((len(things), len(things)), dtype=np.float64)
    for i, a in enumerate(things):
        for j, b in enumerate(things):
            if i != j:
                dists[i, j] = np.float64(distance(a, b))
    return dists


def calculate_distances(
    things: Collection[T],
    distance: Callable[[T, T], np.float64],
    engine: Literal["python", "numba"],
) -> np.ndarray[np.floating[Any], Any]:
    if len(things) < 2:
        raise ValueError("len(things) < 2")

    if not callable(distance):
        raise ValueError("distance wrong")

    if engine == "python":
        return get_distance_matrix(things, distance)

    elif engine == "numba":
        import numba

        return numba.jit(get_distance_matrix)(things, distance)  # type: ignore

    raise ValueError("engine wrong")


def run(
    things: Collection[T],
    distance: Callable[[T, T], np.float64],
    labels: np.ndarray[np.str_ | np.int_, Any],
    engine: Literal["python", "numba"],
    already_squared=False,
) -> PermanovaResults:
    if labels.dtype is np.dtype(str):
        fastlabels = ordinal_encoding(cast("np.ndarray[np.str_, Any]", labels))
    else:
        if not (min(labels) == 0 and max(labels) == len(np.unique(labels)) - 1):
            raise ValueError(
                "in case of integer array it must be an ordinal encoding"
            )  # TODO: do ordinal encoding regardless of type
        fastlabels = cast("np.ndarray[np.uint, Any]", labels)
    dist = calculate_distances(things, distance, engine)
    if not already_squared:
        dist **= 2
    return PermanovaResults(*permanova(dist, fastlabels))
