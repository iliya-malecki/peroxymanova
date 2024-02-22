from __future__ import annotations
from typing import (
    NamedTuple,
    TypeVar,
    Iterator,
    Iterable,
    Callable,
    Literal,
    Any,
    overload,
)
from . import _oxide
from ._oxide import permanova
from .types import AnySequence
from .distance import (
    get_distance_matrix,
    get_distance_matrix_numba,
    get_distance_matrix_parallel,
)
import numpy as np


T = TypeVar("T")
Tc = TypeVar("Tc", covariant=True)


def ordinal_encoding(
    arr: np.ndarray[Any, np.dtype[_oxide.ordinal_encoding_dtypes]],
) -> np.ndarray[Any, np.dtype[np.uint]]:
    '''
    An implementation of normal ordinal encoding in rust to shed some dependencies.
    Only works with certain types!
    '''
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


@overload
def calculate_distances(
    things: Iterable[T],
    distance: Callable[[T, T], np.float64],
    engine: Literal["python"],
    symmetrification: Literal["roundtrip", "one-sided"],
) -> np.ndarray[Any, np.dtype[np.floating[Any]]]:
    ...


@overload
def calculate_distances(
    things: AnySequence[T],
    distance: Callable[[T, T], np.float64],
    engine: Literal["numba"],
    symmetrification: Literal["roundtrip", "one-sided"],
) -> np.ndarray[Any, np.dtype[np.floating[Any]]]:
    ...


@overload
def calculate_distances(
    things: AnySequence[T],
    distance: Callable[[T, T], np.float64],
    engine: Literal["concurrent.futures",],
    symmetrification: Literal["roundtrip", "one-sided"],
    workers: int,
) -> np.ndarray[Any, np.dtype[np.floating[Any]]]:
    ...


def calculate_distances(
    things: Iterable[T] | AnySequence[T],
    distance: Callable[[T, T], np.float64],
    engine: Literal["python", "numba", "concurrent.futures"],
    symmetrification: Literal["roundtrip", "one-sided"],
    workers: int | None = None,
) -> np.ndarray[Any, np.dtype[np.floating[Any]]]:
    """
    Compute distances between `things`. Primarily used in `peroxymanova.run`
    but exposed here for edge cases when `run` is inconvenient. For parameter
    explanation see `peroxymanova.run` docs.
    """
    return _calculate_distances(
        things=things,
        distance=distance,
        engine=engine,
        symmetrification=symmetrification,
        workers=workers,
    )


def _calculate_distances(
    things: Iterable[T] | AnySequence[T],
    distance: Callable[[T, T], np.float64],
    engine: Literal["python", "numba", "concurrent.futures"],
    symmetrification: Literal["roundtrip", "one-sided"],
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
        return get_distance_matrix(things, distance, symmetrification)

    elif engine == "concurrent.futures":
        if not isinstance(things, AnySequence):
            raise ValueError(
                "`things` must be a `Sequence` for engine == 'concurrent.futures'"
            )
        return get_distance_matrix_parallel(things, distance, symmetrification, workers)

    elif engine == "numba":
        import numba

        return numba.njit(get_distance_matrix_numba)(
            things,
            numba.njit(distance),  # type: ignore
            symmetrification,
        )  # type: ignore

    raise ValueError("engine isnt in the list of allowed ones, consult the type hints")


@overload
def permanova_pipeline(
    things: Iterable[T],
    distance: Callable[[T, T], np.float64],
    labels: np.ndarray[Any, np.dtype[_oxide.ordinal_encoding_dtypes]],
    engine: Literal["python"],
    symmetrification: Literal["roundtrip", "one-sided"],
    already_squared=False,
    permutations=1000,
) -> PermanovaResults:
    ...


@overload
def permanova_pipeline(
    things: AnySequence[T],
    distance: Callable[[T, T], np.float64],
    labels: np.ndarray[Any, np.dtype[_oxide.ordinal_encoding_dtypes]],
    engine: Literal["numba"],
    symmetrification: Literal["roundtrip", "one-sided"],
    already_squared=False,
    permutations=1000,
) -> PermanovaResults:
    ...


@overload
def permanova_pipeline(
    things: AnySequence[T],
    distance: Callable[[T, T], np.float64],
    labels: np.ndarray[Any, np.dtype[_oxide.ordinal_encoding_dtypes]],
    engine: Literal["concurrent.futures"],
    symmetrification: Literal["roundtrip", "one-sided"],
    already_squared=False,
    permutations=1000,
    workers: int | None = None,
) -> PermanovaResults:
    ...


def permanova_pipeline(
    things: Iterable[T] | AnySequence[T],
    distance: Callable[[T, T], np.float64],
    labels: np.ndarray[Any, np.dtype[_oxide.ordinal_encoding_dtypes]],
    engine: Literal["python", "numba", "concurrent.futures"],
    symmetrification: Literal["roundtrip", "one-sided"],
    already_squared=False,
    permutations=1000,
    workers: int | None = None,
) -> PermanovaResults:
    """
    ### Run the full pipeline:
    1. Compute pairwise distances between `things` and resolve inconsistencies*
    2. Perform ordinal encoding of lables for the `peroxymanova.permanova`
    3. Run highly optimized compiled `peroxymanova.permanova`*

    ### Notes
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

    ### Parameters
    for the types please consult the type annotations

    things: a set of (any) things of some type `T` to run permanova for

    distance: a function that accepts two objects of type `T` and returns\
    a float "distance" between them. This can be anything, really,\
    but the closer it is in spirit to the Eucledian (L2) distance, the better\
    (but then again, if an honest Eucledian distance can actually be defined,\
    then `T` is just some kind of number and you should use anova from scipy)

    labels: an array of labels of the same length as `things`\
    (even if `__len__` is not defined for `things`, there must be a correspondence by order)

    engine: the engine that will calculate the distance matrix:
    - python: the most flexible one, only requires `things` to have `__iter__` method.\
    It is implied that iterating over `things` doesnt mutate them.
    - numba: uses numba's just-in-time compilation to calculate the distance matrix faster,\
    but requires `things` to have `__getitem__` method and be numba-friendly,\
    along with the `distance` function. If you arent sure if your objects are numba-friendly,\
    prepare for numba errors.
    - concurrent.futures: uses concurrent.futures to run the distance computation in parallel,\
    requires `things` to have `__getitem__` method. This can be used for relatively fast computation\
    of potentially large objects, as the `things` can be a lazy dataset that loads huge things\
    with its `__getitem__`.

    symmetrification: a strategy for ensuring symmetric distance matrix
    - roundtrip: each `distance(a, b)` is summed with its counterpart `distance(b, a)`
    - one-sided: only compute one `distance(a, b)`\
    for `a` and `b` such that `a` comes before `b` in the `things`

    already_squared: permanova algorithm requires distances to be squared,\
    but for speed one can provide a `distance` function that returns a squared result\
    (it makes sense if `distance` was supposed to take a square root)

    permutations: amount of permutations for calculating p-value in the core algorithm

    workers: amount of workers for concurrent.futures, only makes sense for that engine

    ### Returns
    PermanovaResults(statistic: np.float64, pvalue: np.float64)
    `statistic` is a deterministic number that represents how
    extreme the difference between groups is. It is kind of impossible to interpret
    but its useful to compare different runs on the same data.
    `pvalue` is a permutation-based approximation of the
    probability that the null hypothesis should be accepted. Since the null hypothesis here
    is that there is no difference between groups, a low p-value (typically below 0.05)
    means that groups are likely different.
    """
    if (
        labels.dtype.kind in ["i", "u"]
        and min(labels) == 0
        and max(labels) == len(np.unique(labels)) - 1
    ):
        fastlabels = labels.astype(np.uint, copy=False)  # if possible, dont copy
    else:
        fastlabels = ordinal_encoding(labels)
    dist = _calculate_distances(things, distance, engine, symmetrification, workers)
    if not already_squared:
        dist **= 2
    return PermanovaResults(*permanova(dist, fastlabels, permutations))
