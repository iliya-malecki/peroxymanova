from reference import square
import peroxymanova
import numpy as np
from scipy.spatial import distance_matrix
from scipy.stats import f_oneway
from numpy.typing import NDArray

size = 10
objects = np.random.random((size, 1))
dist: NDArray[np.float64] = distance_matrix(objects, objects)
labels = np.random.randint(0, 2, size)


def test_it():
    python = square.permanova(dist**2, labels.copy())
    our = peroxymanova.permanova(dist**2, labels.copy())
    anova = f_oneway(*(objects[labels == i] for i in np.unique(labels)))

    print(python)
    print(our)
    print(anova)

    assert np.allclose(anova.statistic[0], our[0])
