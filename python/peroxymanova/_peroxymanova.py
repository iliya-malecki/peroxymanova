from __future__ import annotations
from typing import TYPE_CHECKING, TypeVar, cast, Sequence
from ._oxide import permanova, ordinal_encoding
import numpy as np
from typing import NamedTuple
from concurrent.futures import ProcessPoolExecutor

if TYPE_CHECKING:
    from typing import Collection, Callable, Literal, Any


class PermanovaResults(NamedTuple):
    statistic: float
    pvalue: float


T = TypeVar("T")


def get_distance_matrix_parallel(
    things: Sequence[T], distance: Callable[[T, T], np.float64], workers: int
):
    def access_helper(i: int):
        """
        used for retrieving the `things` elements one by one
        regardless of how the tasks were sent to the worker,
        since the tasks are just ints
        """
        thing = things[i]  # run __getitem__ once
        return [
            distance(thing, things[j]) if i != j else 0.0 for j in range(len(things))
        ]

    with ProcessPoolExecutor(max_workers=workers) as ppe:
        return np.array(ppe.map(access_helper, range(len(things))), dtype=np.float64)


def get_distance_matrix(things: Collection[T], distance: Callable[[T, T], np.float64]):
    dists = np.empty((len(things), len(things)), dtype=np.float64)
    for i, a in enumerate(things):
        for j, b in enumerate(things):
            if i != j:
                dists[i, j] = np.float64(distance(a, b))
    return dists


def calculate_distances(
    things: Collection[T] | Sequence[T],
    distance: Callable[[T, T], np.float64],
    engine: Literal["python", "concurrent.futures", "numba"],
) -> np.ndarray[np.floating[Any], Any]:
    if len(things) < 2:
        raise ValueError("len(things) < 2")

    if not callable(distance):
        raise ValueError("distance wrong")

    if engine == "python":
        return get_distance_matrix(things, distance)

    elif engine == "concurrent.futures":
        if not isinstance(things, Sequence):
            # TODO: support Iterable, since each worker is chopping off a head from the Iterable of things and then iterates through all the Iterable again
            raise ValueError(
                "`things` must be a `Sequence` for engine == 'concurrent.futures'"
            )
        return get_distance_matrix_parallel(things, distance, 2)

    elif engine == "numba":
        import numba

        return numba.jit(get_distance_matrix)(things, distance)  # type: ignore

    raise ValueError("engine wrong")


def run(
    things: Collection[T] | Sequence[T],
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
