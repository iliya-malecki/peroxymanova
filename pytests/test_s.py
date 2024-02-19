from __future__ import annotations
from typing import Any, Generic, TypeVar
from reference import square
import peroxymanova
import numpy as np
from scipy.spatial import distance_matrix
from scipy.stats import f_oneway
import pytest
import numba

size = 100
objects = np.random.random((size, 1))
dist: np.ndarray[Any, np.dtype[np.float64]] = distance_matrix(objects, objects)
labels = np.random.randint(0, 2, size)
anova = f_oneway(*(objects[labels == i] for i in np.unique(labels)))

T = TypeVar("T", covariant=True)


def distance_function(a: np.float64, b: np.float64):
    return np.sqrt(np.sum((a - b) ** 2))


class MockDataLoaderIndexable(Generic[T]):
    def __init__(self, arr: np.ndarray) -> None:
        self.arr = arr

    def __getitem__(self, __key: int) -> T:
        return self.arr[__key]

    def __len__(self) -> int:
        return len(self.arr)


def test_permanova():
    our = peroxymanova.permanova(dist**2, labels.astype(np.uint))
    assert np.allclose(anova.statistic[0], our[0])


@pytest.mark.parametrize("labeltype", [np.int16, np.int64, np.str_])
@pytest.mark.parametrize("engine", ["python", "numba"])
def test_run_iterable(engine, labeltype):
    if engine == "numba":
        local_distance_function = numba.njit(distance_function)
    else:
        local_distance_function = distance_function
    run_py_results = peroxymanova.permanova_pipeline(
        objects, local_distance_function, labels.astype(labeltype), engine=engine
    )
    assert np.allclose(anova.statistic[0], run_py_results.statistic)


@pytest.mark.parametrize("labeltype", [np.int16, np.int64])
def test_run_ordinal_encoding_on_ints(labeltype):
    run_py_results = peroxymanova.permanova_pipeline(
        objects, distance_function, labels.astype(labeltype) + 42, engine="python"
    )
    assert np.allclose(anova.statistic[0], run_py_results.statistic)


@pytest.mark.parametrize("labeltype", [np.int16, np.int64, np.str_])
def test_run_indexable(labeltype):
    run_py_results = peroxymanova.permanova_pipeline(
        MockDataLoaderIndexable(objects),
        distance_function,
        labels.astype(labeltype),
        engine="concurrent.futures",
    )
    assert np.allclose(anova.statistic[0], run_py_results.statistic)
