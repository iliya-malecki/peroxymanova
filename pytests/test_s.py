from __future__ import annotations
from typing import Any
from reference import square
import peroxymanova
import numpy as np
from scipy.spatial import distance_matrix
from scipy.stats import f_oneway

size = 100
objects = np.random.random((size, 1))
dist: np.ndarray[np.float64, Any] = distance_matrix(objects, objects)
labels = np.random.randint(0, 2, size)


def test_it():
    python = square.permanova(dist**2, labels.copy())
    our = peroxymanova.permanova(dist**2, labels.copy())
    anova = f_oneway(*(objects[labels == i] for i in np.unique(labels)))
    run_py_results = peroxymanova.run(objects, lambda a, b: np.sqrt(np.sum((a-b)**2)), labels, 'python')
    run_sci_results = peroxymanova.run(objects, 'euclid', labels, 'scipy')


    assert np.allclose(anova.statistic[0], our[0])
    assert np.allclose(anova.statistic[0], run_py_results.statistic)
    assert np.allclose(anova.statistic[0], run_sci_results.statistic)
