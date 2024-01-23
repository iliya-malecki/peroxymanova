from reference import square, packed
import peroxymanova
import cProfile
import numpy as np

size = 1000
dist = np.random.random((size, size))
dist = dist + dist.T
np.fill_diagonal(dist, 0)
labels = np.random.randint(0, 3, size)
cProfile.run("square.permanova(dist, labels)", sort="tottime")
cProfile.run("packed.permanova(dist, labels)", sort="tottime")
cProfile.run("peroxymanova.permanova(dist, labels)", sort="tottime")
