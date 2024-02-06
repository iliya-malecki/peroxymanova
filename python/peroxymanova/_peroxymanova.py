from __future__ import annotations
from typing import TYPE_CHECKING, TypeVar, overload
from ._oxide import permanova, ordinal_encoding
import numpy as np
from scipy.spatial.distance import pdist

if TYPE_CHECKING:
    from typing import Sequence, Callable, Literal, TypeGuard, Any, cast
    from ._scipy_types import _FloatValue, _MetricKind


T = TypeVar("T")


def get_distance_matrix(things: Sequence[T], distance: Callable[[T, T], _FloatValue]):
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
    things: Sequence[T] | np.ndarray[Any, Any],
    distance: Callable[[T, T], _FloatValue] | _MetricKind,
    engine: str,
) -> np.ndarray[np.floating[Any], Any]:
    if len(things) < 2:
        raise ValueError("len(things) < 2")

    if engine == "scipy":
        if isinstance(things, np.ndarray) and len(things.shape) == 2:
            if callable(distance) or isinstance(distance, str):
                return pdist(
                    things,
                    cast(
                        Callable[[np.ndarray, np.ndarray], _FloatValue] | _MetricKind,
                        distance,
                    ),
                )
            else:
                raise ValueError("distance wrong for engine == 'scipy'")
        else:
            raise ValueError("things wrong")

    elif engine == "python":
        if isinstance(distance, str):
            raise ValueError("distance wrong")
        return get_distance_matrix(cast(Sequence[T], things), distance)

    elif engine == "numba":
        if isinstance(distance, str):
            raise ValueError("distance wrong")
        import numba  # type: ignore[reportMissingTypeStubs]

        return numba.jit(get_distance_matrix)(things, distance)  # type: ignore

    raise ValueError("engine wrong")


@overload
def run(
    things: np.ndarray[Any, Any],
    distance: _MetricKind,
    labels: np.ndarray[np.str_ | np.int_, Any],
    engine: Literal["scipy"],
) -> tuple[float, float]:
    ...


@overload
def run(
    things: Sequence[T],
    distance: Callable[[T, T], _FloatValue],
    labels: np.ndarray[np.str_ | np.int_, Any],
    engine: Literal["python", "numba"],
) -> tuple[float, float]:
    ...


def run(
    things: Sequence[T] | np.ndarray[Any, Any],
    distance: Callable[[T, T], _FloatValue] | _MetricKind,
    labels: np.ndarray[np.str_ | np.int_, Any],
    engine: Literal["scipy", "python", "numba"],
) -> tuple[float, float]:
    if isinstance(labels[0], str):
        fastlabels = ordinal_encoding(cast(np.ndarray[np.str_, Any], labels))
    else:
        if not (min(labels) == 0 and max(labels) == len(np.unique(labels)) - 1):
            raise ValueError(
                "in case of integer array it must be an ordinal encoding"
            )  # TODO: do ordinal encoding regardless of type
        fastlabels = cast(np.ndarray[np.uint, Any], labels)
    dist = calculate_distances(things, distance, engine)
    return permanova(dist, fastlabels)
