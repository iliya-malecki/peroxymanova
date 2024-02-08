from __future__ import annotations
from typing import TYPE_CHECKING, TypeVar, cast
from ._oxide import permanova, ordinal_encoding
import numpy as np
from typing import NamedTuple

if TYPE_CHECKING:
    from typing import Collection, Callable, Literal, TypeGuard, Any


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


RT = TypeVar("RT")
CT = TypeVar("CT")


# the runtime check will be performed by the package
# am i having too much fun with the type system?
def pinky_promise_guard(
    ct: Callable[[T, T], RT], t: T, check_type: type[CT]
) -> TypeGuard[Callable[[CT, CT], RT]]:
    """
    check if `t:T` is of type `check_type: CT` and if so,
    rely on the fact that the callable accepts [T,T] (same as t!)
    to infer it must also be [CT,CT].
    This narrowing should be automatic but oh well
    """
    return isinstance(t, check_type)


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
    return PermanovaResults(*permanova(dist**2, fastlabels))
