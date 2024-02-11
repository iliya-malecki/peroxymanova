from __future__ import annotations
from typing import Any, Generic, TypeVar, Iterator
from reference import square
import peroxymanova
import numpy as np
from scipy.spatial import distance_matrix
from scipy.stats import f_oneway

size = 10
objects = np.random.random((size, 1))
dist: np.ndarray[np.float64, Any] = distance_matrix(objects, objects)
labels = np.random.randint(0, 2, size)

T = TypeVar("T", covariant=True)


def distance_function(a: np.float64, b: np.float64):
    return np.sqrt(np.sum((a - b) ** 2))


class MockDataLoaderIndexable(Generic[T]):
    def __init__(self, arr: np.ndarray) -> None:
        self.arr = arr

    def __getitem__(self, __key: int) -> T:
        return self.arr[__key]

    def __iter__(self) -> Iterator[T]:
        yield from self.arr

    def __len__(self) -> int:
        return len(self.arr)

    def __contains__(self, item):
        return item in self.arr


def test_it():
    python = square.permanova(dist**2, labels.copy())
    our = peroxymanova.permanova(dist**2, labels.copy())
    anova = f_oneway(*(objects[labels == i] for i in np.unique(labels)))
    run_py_results = peroxymanova.run(objects, distance_function, labels, "python")
    run_sci_results = peroxymanova.run(
        objects,
        distance_function,
        labels,
        "concurrent.futures",
    )
    run_sci_results = peroxymanova.run(
        MockDataLoaderIndexable(objects),
        distance_function,
        labels,
        "concurrent.futures",
    )

    assert np.allclose(anova.statistic[0], our[0])
    assert np.allclose(anova.statistic[0], run_py_results.statistic)
    assert np.allclose(anova.statistic[0], run_sci_results.statistic)
