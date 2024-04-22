from reference import square, packed
import peroxymanova
import cProfile
import numpy as np
from typing import Callable, Any
import pstats
import inspect
import re


class Color:
    reset = "\u001b[0m"

    def __init__(self, value) -> None:
        self.value = value

    def __add__(self, other: str):
        if not isinstance(other, str):
            return NotImplemented
        return self.value + other + self.reset


YELLOW = Color("\033[1;33m")

size = 1000
dist = np.random.random((size, size))
dist = dist + dist.T
np.fill_diagonal(dist, 0)
labels = np.random.randint(0, 3, size)


def run(callback: Callable[[], Any]):
    code = re.search(r"lambda: (.+)", inspect.getsource(callback))
    if code is None:
        raise ValueError("a lambda should be passed")
    print(YELLOW + code.group(1), "\n")
    with cProfile.Profile() as pr:
        callback()
    pstats.Stats(pr).sort_stats("tottime").print_stats(5)


run(lambda: square.permanova(dist, labels))
run(lambda: packed.permanova(dist, labels))
run(lambda: peroxymanova.permanova(dist, labels.astype("uint")))
