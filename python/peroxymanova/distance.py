from __future__ import annotations
from typing import Callable, TypeVar, Literal, Iterable
from .types import AnySequence
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import itertools

T = TypeVar("T")


def get_filter_for_strategy(symmetrification: Literal["roundtrip", "one-sided"]):
    if symmetrification == "one-sided":
        return lambda i, j: i > j
    else:
        return lambda i, j: i != j


def _access_helper(
    i: int,
    distance: Callable[[T, T], np.float64],
    things: AnySequence[T],
    symmetrification: Literal["roundtrip", "one-sided"],
):
    """
    used for retrieving the `things` elements one by one
    regardless of how the tasks were sent to the worker,
    since the tasks are just ints
    """
    thing = things[i]  # run __getitem__ once on the worker
    filter = get_filter_for_strategy(symmetrification)
    return [
        distance(thing, things[j]) if filter(i, j) else 0.0 for j in range(len(things))
    ]


def get_distance_matrix_parallel(
    things: AnySequence[T],
    distance: Callable[[T, T], np.float64],
    symmetrification: Literal["roundtrip", "one-sided"],
    workers: int | None,
):
    with ProcessPoolExecutor(max_workers=workers) as ppe:
        ret = np.array(
            list(
                ppe.map(
                    partial(
                        _access_helper,
                        distance=distance,
                        things=things,
                        symmetrification=symmetrification,
                    ),
                    range(len(things)),
                )
            ),
            dtype=np.float64,
        )
    ret += ret.T
    return ret


def get_distance_matrix(
    things: Iterable[T],
    distance: Callable[[T, T], np.float64],
    symmetrification: Literal["roundtrip", "one-sided"],
) -> np.ndarray:
    filter = get_filter_for_strategy(symmetrification)
    dists = []
    for i, a in enumerate(things):
        row = []
        for j, b in enumerate(things):
            if filter(i, j):
                row.append(distance(a, b))
            else:
                row.append(0.0)
        dists.append(row)
    ret = np.array(dists)
    ret += ret.T
    print(ret)
    return ret


def get_distance_matrix_numba(
    things: AnySequence[T],
    distance: Callable[[T, T], np.float64],
    symmetrification: Literal["roundtrip", "one-sided"],
):
    dists = np.zeros((len(things), len(things)), dtype=np.float64)
    for i in range(len(things)):
        if symmetrification == "one-sided":
            jmax = i
        else:
            jmax = len(things)
        for j in range(jmax):
            if i == j:
                dists[j, i] = 0.0
            else:
                dists[j, i] = distance(things[i], things[j])
    dists += dists.T
    print(dists)
    return dists
