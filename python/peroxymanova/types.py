from __future__ import annotations
from typing import runtime_checkable, Protocol, TypeVar

Tc = TypeVar("Tc", covariant=True)


@runtime_checkable
class AnySequence(Protocol[Tc]):
    def __getitem__(self, __key: int) -> Tc:
        ...

    def __len__(self) -> int:
        ...
