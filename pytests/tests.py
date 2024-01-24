from skbio.stats.distance import permanova, DistanceMatrix
from reference import square
import peroxymanova
import numpy as np
from scipy.spatial import distance_matrix
from scipy.stats import f_oneway

size = 10
objects = np.random.random((size, 1))
dist = distance_matrix(objects, objects)
labels = np.random.randint(0, 2, size)
standard = permanova(
    DistanceMatrix(dist),
    labels.copy(),
    permutations=1000
)

python = square.permanova(dist**2, labels.copy())

our = peroxymanova.permanova(dist**2, labels.copy())
anova = f_oneway(*(objects[labels == i] for i in np.unique(labels)))

print(standard)
print(python)
print(our)
print(anova)

assert np.allclose(standard['test statistic'], our[0])
