from skbio.stats.distance import permanova, DistanceMatrix
from reference import square
import peroxymanova
import numpy as np

size = 1000
dist = np.random.random((size, size))
dist = dist + dist.T
np.fill_diagonal(dist, 0)
labels = np.random.randint(0, 3, size)

standard = permanova(
    DistanceMatrix(dist),
    labels,
    permutations=1000
)

python = square.permanova(dist, labels)

our = peroxymanova.permanova(dist, labels)

print(standard)
print(python)
print(our)

assert standard['test statistic'] == our[0]
assert standard['p-value'] == our[1]
