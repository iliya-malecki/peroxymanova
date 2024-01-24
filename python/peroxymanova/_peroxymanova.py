from typing import TypeVar, Collection, Callable, Literal, overload, TypeGuard, Any
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import pdist, _MetricKind, _FloatValue  # type: ignore


T = TypeVar("T")


@overload
def calculate_distances(
    things: NDArray[Any], distance: _MetricKind, engine: Literal["scipy"]
) -> None:
    ...


@overload
def calculate_distances(
    things: Collection[T], distance: Callable[[T, T], _FloatValue], engine: str
) -> None:
    ...


RT = TypeVar("RT")
CT = TypeVar("CT")


# the runtime check will be performed by the package
# am i having too much fun with the type system?
def pinky_promise_guard(
    ct: Callable[[T, T], RT], t: T, check_type: type[CT]
) -> TypeGuard[Callable[[CT, CT], RT]]:
    return isinstance(t, check_type)


def calculate_distances(
    things: Collection[T] | NDArray[Any],
    distance: Callable[[T, T], _FloatValue] | _MetricKind,
    engine: str,
):
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
