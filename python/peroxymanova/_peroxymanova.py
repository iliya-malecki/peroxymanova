from typing import TypeVar, Collection, Callable, Literal, overload, TypeGuard, Any
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import pdist, _MetricKind, _FloatValue  # type: ignore


T = TypeVar("T")


def get_distance_matrix(things: Collection[T], distance: Callable[[T, T], _FloatValue]):
    dists = np.empty((len(things), len(things)), dtype=np.float64)
    for i, a in enumerate(things):
        for j, b in enumerate(things):
            if i != j:
                dists[i, j] = np.float64(distance(a, b))
    return dists


@overload
def calculate_distances(
    things: NDArray[Any], distance: _MetricKind, engine: Literal["scipy"]
) -> NDArray[np.floating[Any]]:
    ...


@overload
def calculate_distances(
    things: Collection[T],
    distance: Callable[[T, T], _FloatValue],
    engine: Literal["python"],
) -> NDArray[np.floating[Any]]:
    ...


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
    things: Collection[T] | NDArray[Any],
    distance: Callable[[T, T], _FloatValue] | _MetricKind,
    engine: str,
) -> NDArray[np.floating[Any]]:
    if len(things) < 2:
        raise ValueError("len(things) < 2")

    if engine == "scipy":
        if isinstance(things, np.ndarray) and len(things.shape) == 2:
            if callable(distance) and pinky_promise_guard(
                distance, things[0], NDArray[Any]
            ):
                return pdist(things, distance)
            elif isinstance(distance, str):
                return pdist(things, distance)  # type inference in overload hello?
            else:
                raise ValueError("distance wrong")
        else:
            raise ValueError("things wrong")

    elif engine == "python":
        if isinstance(distance, str):
            raise ValueError("distance wrong")
        return get_distance_matrix(things, distance)

    elif engine == "numba":
        if isinstance(distance, str):
            raise ValueError("distance wrong")
        import numba  # type: ignore[reportMissingTypeStubs]

        return numba.jit(get_distance_matrix)(things, distance)  # type: ignore

    raise ValueError("engine wrong")
