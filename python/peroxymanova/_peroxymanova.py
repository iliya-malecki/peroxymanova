from __future__ import annotations
from typing import (
    TypeVar,
    Protocol,
    Iterator,
    Iterable,
    Callable,
    Literal,
    Any,
    runtime_checkable,
    overload,
)
from ._oxide import permanova
from . import _oxide
import numpy as np
from typing import NamedTuple
from concurrent.futures import ProcessPoolExecutor
from functools import partial


T = TypeVar("T")
Tc = TypeVar("Tc", covariant=True)


def ordinal_encoding(
    arr: np.ndarray[Any, np.dtype[_oxide.ordinal_encoding_dtypes]],
) -> np.ndarray[Any, np.dtype[np.uint]]:
    if not isinstance(arr, np.ndarray):
        raise TypeError("input should be a `np.ndarray`")
    if arr.dtype.name.startswith(("str", "bytes", "void")):
        suffix = "".join(x for x in arr.dtype.name if x.isalpha())
    else:
        suffix = arr.dtype.name
    try:
        func = getattr(_oxide, f"ordinal_encoding_{suffix}")
    except NameError:
        raise TypeError(f"input dtype {arr.dtype} not understood")
    return func(arr)


class PermanovaResults(NamedTuple):
    statistic: float
    pvalue: float


@runtime_checkable
class AnySequence(Protocol[Tc]):
    def __getitem__(self, __key: int) -> Tc:
        ...

    def __len__(self) -> int:
        ...


def _access_helper(
    i: int, distance: Callable[[T, T], np.float64], things: AnySequence[T]
):
    """
    used for retrieving the `things` elements one by one
    regardless of how the tasks were sent to the worker,
    since the tasks are just ints
    """
    thing = things[i]  # run __getitem__ once on the worker
    return [distance(thing, things[j]) if i != j else 0.0 for j in range(len(things))]


def get_distance_matrix_parallel(
    things: AnySequence[T], distance: Callable[[T, T], np.float64], workers: int | None
):
    with ProcessPoolExecutor(max_workers=workers) as ppe:
        return np.array(
            list(
                ppe.map(
                    partial(_access_helper, distance=distance, things=things),
                    range(len(things)),
                )
            ),
            dtype=np.float64,
        )


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


def get_distance_matrix_numba(
    things: AnySequence[T], distance: Callable[[T, T], np.float64]
):
    dists = np.empty((len(things), len(things)), dtype=np.float64)
    for i in range(len(things)):
        for j in range(len(things)):
            if i == j:
                dists[j, i] = 0.0
            else:
                dists[j, i] = distance(things[i], things[j])
    return dists


@overload
def calculate_distances(
    things: Iterable[T],
    distance: Callable[[T, T], np.float64],
    engine: Literal["python"],
) -> np.ndarray[Any, np.dtype[np.floating[Any]]]:
    ...


@overload
def calculate_distances(
    things: AnySequence[T],
    distance: Callable[[T, T], np.float64],
    engine: Literal["concurrent.futures", "numba"],
    workers: int,
) -> np.ndarray[Any, np.dtype[np.floating[Any]]]:
    ...


def calculate_distances(
    things: Iterable[T] | AnySequence[T],
    distance: Callable[[T, T], np.float64],
    engine: Literal["python", "numba", "concurrent.futures"],
    workers: int | None = None,
) -> np.ndarray[Any, np.dtype[np.floating[Any]]]:
    return _calculate_distances(
        things=things, distance=distance, engine=engine, workers=workers
    )


def _calculate_distances(
    things: Iterable[T] | AnySequence[T],
    distance: Callable[[T, T], np.float64],
    engine: Literal["python", "numba", "concurrent.futures"],
    workers: int | None = None,
) -> np.ndarray[Any, np.dtype[np.floating[Any]]]:
    if isinstance(things, Iterator):
        raise ValueError(
            "`things` should be immutable on read, i.e. be an `Iterable` and not `Iterator`"
        )
    if workers is not None and engine in ["python", "numba"]:
        raise ValueError("`workers` is an option for engine='concurrent.futures'")

    if not callable(distance):
        raise ValueError("distance must be a callable")

    if engine == "python":
        if not isinstance(things, Iterable):
            raise ValueError("`things` must be a `Iterable` for engine == 'python'")
        return get_distance_matrix(things, distance)

    elif engine == "concurrent.futures":
        if not isinstance(things, AnySequence):
            raise ValueError(
                "`things` must be a `Sequence` for engine == 'concurrent.futures'"
            )
        return get_distance_matrix_parallel(things, distance, workers)

    elif engine == "numba":
        import numba

        return numba.njit(get_distance_matrix_numba)(things, numba.njit(distance))  # type: ignore

    raise ValueError("engine isnt in the list of allowed ones, consult the type hints")


@overload
def permanova_pipeline(
    things: Iterable[T],
    distance: Callable[[T, T], np.float64],
    labels: np.ndarray[Any, np.dtype[_oxide.ordinal_encoding_dtypes]],
    engine: Literal["python"],
    already_squared=False,
) -> PermanovaResults:
    ...


@overload
def permanova_pipeline(
    things: AnySequence[T],
    distance: Callable[[T, T], np.float64],
    labels: np.ndarray[Any, np.dtype[_oxide.ordinal_encoding_dtypes]],
    engine: Literal["concurrent.futures", "numba"],
    already_squared=False,
) -> PermanovaResults:
    ...


def permanova_pipeline(
    things: Iterable[T] | AnySequence[T],
    distance: Callable[[T, T], np.float64],
    labels: np.ndarray[Any, np.dtype[_oxide.ordinal_encoding_dtypes]],
    engine: Literal["python", "numba", "concurrent.futures"],
    symmetrification: Literal["roundtrip", "one-sided"],
    already_squared=False,
    workers: int | None = None,
) -> PermanovaResults:
    """
    ### Run the full pipeline:
    1. Compute pairwise distances between `things` and resolve inconsistencies*
    2. Perform ordinal encoding of lables for the `peroxymanova.permanova`
    3. Run highly optimized compiled `peroxymanova.permanova`*

    ### Notes:
    - "inconsistencies" in the pairwise distances refer to the fact that the
    properties of symmetry (i.e. `distance(a, b) == distance(b, a)`)
    and zero distance to self (i.e. `distance(a, a) == 0`) dont necessarily hold
    an arbitrary user-defined function `distance(a, b)`, and those are important for metrics.
    While violating zero distance to self happens to not affect one-way permutational anova,
    violating symmetry does affect it, so `symmetrification` parameter controls how it is achieved.
    - the result of running permanova is the p-value for the
    hypothesis that different groups of `things` (encoded in `labels`)
    are identical, as far as the `distance` metric is concerned.
    Hence, it can be interpreted in the same way a p-value for anova is interpreted.
    - The key difference between permanova and anova is that this algorithm only requires
    a `distance` to be defined, while anova needs all operators on numbers (like addition
    and division and whatnot)

    ## Parameters:
    for the types please consult the type annotations

    things: a set of (any) things of some type `T` to run permanova for

    labels: an array of labels of the same length as `things`
    (even if `__len__` is not defined for `things`, there must be a correspondence by order)

    distance: a function that accepts two objects of type `T` and returns
    a float "distance" between them. This can be anything, really,
    but the closer it is in spirit to the Eucledian (L2) distance, the better
    (but then again, if an honest Eucledian distance can actually be defined,
    then `T` is just some kind of number and you should use anova from scipy)

    engine: the engine that will calculate the distance matrix
    - python: the most flexible one, only requires `things` to have `__iter__` method.
    It is implied that iterating over `things` doesnt mutate them.
    - numba: uses numba's just-in-time compilation to calculate the distance matrix faster,
    but requires `things` to have `__getitem__` method and be numba-friendly,
    along with the `distance` function. If you arent sure if your objects are numba-friendly,
    prepare for numba errors.
    - concurrent.futures: uses concurrent.futures to run the distance computation in parallel,
    requires `things` to have `__getitem__` method. This can be used for relatively fast computation
    of potentially large objects, as the `things` can be a lazy dataset that loads huge things
    with its `__getitem__`.

    symmetrification: a strategy for ensuring symmetric distance matrix
    - roundtrip: each `distance(a, b)` is summed with its counterpart `distance(b, a)`
    - one-sided: only compute one `distance(a, b)`
    for `a` and `b` such that `a` comes before `b` in the `things`

    already_squared: permanova algorithm requires distances to be squared,
    but for speed one can provide a `distance` function that returns a squared result
    (it makes sense if `distance` was supposed to take a square root)

    workers: amount of workers for concurrent.futures, only makes sense for that engine
    """
    if (
        labels.dtype.kind in ["i", "u"]
        and min(labels) == 0
        and max(labels) == len(np.unique(labels)) - 1
    ):
        fastlabels = labels.astype(np.uint, copy=False)  # if possible, dont copy
    else:
        fastlabels = ordinal_encoding(labels)
    dist = _calculate_distances(things, distance, engine, workers)
    if not already_squared:
        dist **= 2
    return PermanovaResults(*permanova(dist, fastlabels))
