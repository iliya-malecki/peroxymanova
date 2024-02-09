from __future__ import annotations
from typing import (
    TypeVar,
    Protocol,
    Iterator,
    Iterable,
    Collection,
    Callable,
    Literal,
    Any,
    cast,
    runtime_checkable,
    overload,
)
from ._oxide import permanova, ordinal_encoding
import numpy as np
from typing import NamedTuple
from concurrent.futures import ProcessPoolExecutor


class PermanovaResults(NamedTuple):
    statistic: float
    pvalue: float


T = TypeVar("T")
Tc = TypeVar("Tc", covariant=True)


@runtime_checkable
class AnySequence(Collection, Protocol[Tc]):
    def __getitem__(self, key: int, /) -> Tc:
        ...


def get_distance_matrix_parallel(
    things: AnySequence[T], distance: Callable[[T, T], np.float64], workers: int | None
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


def get_distance_matrix(things: Iterable[T], distance: Callable[[T, T], np.float64]):
    dists = []
    for i, a in enumerate(things):
        row = []
        for j, b in enumerate(things):
            if i == j:
                row.append(0.0)
            else:
                row.append(distance(a, b))
        dists.append(row)
    return np.array(dists)


@overload
def calculate_distances(
    things: Iterable[T],
    distance: Callable[[T, T], np.float64],
    engine: Literal["python", "numba"],
) -> np.ndarray[np.floating[Any], Any]:
    ...


@overload
def calculate_distances(
    things: Iterable[T] | AnySequence[T],
    distance: Callable[[T, T], np.float64],
    engine: Literal["concurrent.futures"],
    workers: int,
) -> np.ndarray[np.floating[Any], Any]:
    ...


def calculate_distances(
    things: Iterable[T] | AnySequence[T],
    distance: Callable[[T, T], np.float64],
    engine: Literal["python", "numba", "concurrent.futures"],
    workers: int | None = None,
) -> np.ndarray[np.floating[Any], Any]:
    return _calculate_distances(
        things=things, distance=distance, engine=engine, workers=workers
    )


def _calculate_distances(
    things: Iterable[T] | AnySequence[T],
    distance: Callable[[T, T], np.float64],
    engine: Literal["python", "numba", "concurrent.futures"],
    workers: int | None = None,
) -> np.ndarray[np.floating[Any], Any]:
    if isinstance(things, Iterator) and engine in ["python", "numba"]:
        raise ValueError(
            "`things` should be immutable on read, i.e. be an `Iterable` and not `Iterator`"
        )
    if workers is not None and engine in ["python", "numba"]:
        raise ValueError("`workers` is an option for concurrent.futures")

    if not callable(distance):
        raise ValueError("distance wrong")

    if engine == "python":
        return get_distance_matrix(things, distance)

    elif engine == "concurrent.futures":
        if not isinstance(things, AnySequence):
            # TODO: support Iterable, since each worker is chopping off a head from the Iterable of things and then iterates through all the Iterable again
            raise ValueError(
                "`things` must be a `Sequence` for engine == 'concurrent.futures'"
            )
        return get_distance_matrix_parallel(things, distance, workers)

    elif engine == "numba":
        import numba

        return numba.jit(get_distance_matrix)(things, distance)  # type: ignore

    raise ValueError("engine wrong")


@overload
def run(
    things: Iterable[T],
    distance: Callable[[T, T], np.float64],
    labels: np.ndarray[Any, np.dtype[np.str_ | np.int_]],
    engine: Literal["python", "numba"],
    already_squared=False,
) -> PermanovaResults:
    ...


@overload
def run(
    things: AnySequence[T],
    distance: Callable[[T, T], np.float64],
    labels: np.ndarray[Any, np.dtype[np.str_ | np.int_]],
    engine: Literal["concurrent.futures"],
    already_squared=False,
) -> PermanovaResults:
    ...


def run(
    things: Iterable[T] | AnySequence[T],
    distance: Callable[[T, T], np.float64],
    labels: np.ndarray[Any, np.dtype[np.str_ | np.int_]],
    engine: Literal["python", "numba", "concurrent.futures"],
    already_squared=False,
    workers: int | None = None,
) -> PermanovaResults:
    if labels.dtype is np.dtype(str):
        fastlabels = ordinal_encoding(cast("np.ndarray[np.str_, Any]", labels))
    else:
        if not (min(labels) == 0 and max(labels) == len(np.unique(labels)) - 1):
            raise ValueError(
                "in case of integer array it must be an ordinal encoding"
            )  # TODO: do ordinal encoding regardless of type
        fastlabels = cast("np.ndarray[np.uint, Any]", labels)
    dist = _calculate_distances(things, distance, engine, workers)
    if not already_squared:
        dist **= 2
    return PermanovaResults(*permanova(dist, fastlabels))
