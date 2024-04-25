from __future__ import annotations
from typing import Any, Generic, TypeVar
import peroxymanova
import numpy as np
from scipy.spatial import distance_matrix
from scipy.stats import f_oneway
import pytest

size = 100
objects = np.random.random((size, 1))
dist: np.ndarray[Any, np.dtype[np.float64]] = distance_matrix(objects, objects)
labels = np.random.randint(0, 2, size)
anova = f_oneway(*(objects[labels == i] for i in np.unique(labels)))

T = TypeVar("T", covariant=True)


def distance_function(a: np.float64, b: np.float64):
    return np.float64(np.sqrt(np.sum((a - b) ** 2)))


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
@pytest.mark.parametrize("symmetrification", ["roundtrip", "one-sided"])
def test_run_iterable(labeltype, engine, symmetrification):
    run_py_results = peroxymanova.permanova_pipeline(
        objects,
        distance_function,
        labels.astype(labeltype),
        engine=engine,
        symmetrification=symmetrification,
    )
    assert np.allclose(anova.statistic[0], run_py_results.statistic)


@pytest.mark.parametrize("labeltype", [np.int16, np.int64])
def test_run_ordinal_encoding_on_ints(labeltype):
    run_py_results = peroxymanova.permanova_pipeline(
        objects,
        distance_function,
        labels.astype(labeltype) + 42,
        engine="python",
        symmetrification="roundtrip",
    )
    assert np.allclose(anova.statistic[0], run_py_results.statistic)


@pytest.mark.parametrize("labeltype", [np.int16, np.int64, np.str_])
@pytest.mark.parametrize("symmetrification", ["roundtrip", "one-sided"])
def test_run_indexable(labeltype, symmetrification):
    run_py_results = peroxymanova.permanova_pipeline(
        MockDataLoaderIndexable(objects),
        distance_function,
        labels.astype(labeltype),
        engine="concurrent.futures",
        symmetrification=symmetrification,
    )
    assert np.allclose(anova.statistic[0], run_py_results.statistic)


@pytest.mark.parametrize("engine", ["python", "numba"])
def test_clean_stdout(engine, capsys: pytest.CaptureFixture[str]):
    peroxymanova.permanova_pipeline(
        objects,
        distance_function,
        labels,
        engine=engine,
        symmetrification="roundtrip",
    )
    assert not capsys.readouterr().out


@pytest.mark.parametrize("engine", ["python", "numba"])
def test_original_data_intact(engine):
    previousdist = dist.copy()
    previouslables = labels.copy()
    peroxymanova.permanova_pipeline(
        objects,
        distance_function,
        labels,
        engine=engine,
        symmetrification="roundtrip",
    )
    assert np.allclose(previousdist, dist)
    assert np.allclose(previouslables, labels)


@pytest.mark.parametrize("ngroups", range(2, 10))
def test_different_number_of_groups(ngroups):
    labels = np.random.randint(0, ngroups, size)
    our = peroxymanova.permanova(dist**2, labels.astype(np.uint))
    anova = f_oneway(*(objects[labels == i] for i in np.unique(labels)))
    assert np.allclose(our[0], anova.statistic[0])


@pytest.mark.parametrize("npermutations", range(3))
def test_different_number_of_permutations(npermutations):
    labels = np.random.randint(0, 5, size)
    our = peroxymanova.permanova(
        dist**2, labels.astype(np.uint), permutations=npermutations * 10
    )
    anova = f_oneway(*(objects[labels == i] for i in np.unique(labels)))
    assert np.allclose(our[0], anova.statistic[0])


class TestGracefulValueErrors:
    def permanova_dist(self):
        with pytest.raises(ValueError):
            print(
                peroxymanova.permanova(
                    np.random.random((size, size + 1)), labels.astype("uint")
                )
            )

    def permanova_non_onehot_labels(self):
        with pytest.raises(ValueError):
            print(peroxymanova.permanova(dist, (labels + 42).astype("uint")))

    def permanova_empty_labels(self):
        with pytest.raises(ValueError):
            print(peroxymanova.permanova(dist, np.zeros(0).astype("uint")))

    def permanova_empty_both(self):
        with pytest.raises(ValueError):
            print(peroxymanova.permanova(np.zeros((0, 0)), np.zeros(0).astype("uint")))

    def permanova_empty_dist(self):
        with pytest.raises(ValueError):
            print(
                peroxymanova.permanova(np.random.random((0, 0)), labels.astype("uint"))
            )


def test_graceful_warning_nan():
    with pytest.warns():
        print(
            peroxymanova.calculate_distances(
                [np.nan] * 100, lambda a, b: np.float64(a - b), "python", "roundtrip"
            )
        )


type_errors_par = pytest.mark.parametrize(
    "sillytype", [None, object, Generic, lambda: 42]
)


class TestGracefulTypeErrors:
    @type_errors_par
    def test_permanova_dist(self, sillytype):
        with pytest.raises(TypeError):
            print(f"{sillytype = }")
            peroxymanova.permanova(sillytype, labels.astype("uint"))

    @type_errors_par
    def test_permanova_labels(self, sillytype):
        with pytest.raises(TypeError):
            print(f"{sillytype = }")
            peroxymanova.permanova(dist, sillytype)

    @type_errors_par
    def test_ordinal(self, sillytype):
        with pytest.raises(TypeError):
            print(f"{sillytype = }")
            peroxymanova.ordinal_encoding(sillytype)

    @type_errors_par
    def test_calculate_distances_things(self, sillytype):
        with pytest.raises(TypeError):
            print(f"{sillytype = }")
            peroxymanova.calculate_distances(
                sillytype, lambda a, b: np.float64(a - b), "python", "roundtrip"
            )

    @type_errors_par
    def test_calculate_distances_lambda(self, sillytype):
        with pytest.raises(TypeError):
            print(f"{sillytype = }")
            peroxymanova.calculate_distances(
                objects, lambda a, b: sillytype, "python", "roundtrip"
            )
