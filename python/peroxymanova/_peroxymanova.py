from typing import TypeVar, Collection, Callable, Literal, overload, TypeGuard, Any
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import pdist, _MetricKind, _FloatValue #type: ignore


T = TypeVar('T')
# NUMBER = float | np.float_ | np.int_
# SCIPY_DISTANCE: TypeAlias = Literal['braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon', 'kulczynski1', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']

@overload
def run(
    things: NDArray[Any],
    distance: _MetricKind,
    engine: Literal['scipy']
)->None:...

@overload
def run(
    things: Collection[T],
    distance: Callable[[T,T], _FloatValue],
    engine: str
)->None:...


RT = TypeVar('RT')
CT = TypeVar('CT')
# the runtime check will be performed by the package
# am i having too much fun with the type system?
def pinky_promise_guard(ct: Callable[[T, T], RT], t: T, check_type: type[CT]) -> TypeGuard[Callable[[CT,CT], RT]]:
    return isinstance(t, check_type)

def run(
    things:   Collection[T]                 | NDArray[Any],
    distance: Callable[[T, T], _FloatValue] | _MetricKind,
    engine: str
):
    if len(things) < 2:
        raise ValueError('len(things) < 2')

    if engine == 'scipy':
        if isinstance(things, np.ndarray) and len(things.shape) == 2:
            if callable(distance) and pinky_promise_guard(distance, things[0], NDArray[Any]):
                dist = pdist(things, distance)
            elif isinstance(distance, str):
                dist = pdist(things, distance) # type inference in overload hello?
            else:
                raise ValueError('distance wrong')
        else:
            raise ValueError('things wrong')

def dist(x:str, y:str):
    return 42

run(
    np.ones((10,10)),
    dist,
    'scipy'
)
