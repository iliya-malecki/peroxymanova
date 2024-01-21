from reference.packed import permanova as packed_permanova
from reference.square import permanova as square_permanova
import cProfile
import numpy as np

dist = np.random.random((600,600))
dist = dist + dist.T
np.fill_diagonal(dist, 0)
labels = np.random.randint(0, 3, 600)
cProfile.run('square_permanova(dist, labels)', sort='tottime')
