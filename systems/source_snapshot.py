"""Immutable source snapshots used by SKC page publication.

A trainer may replace a source table, but it must never mutate an array that
has already been pinned for refresh.  ``pin_array`` therefore copies a caller
supplied table and marks the copy read-only.  Refresh code releases the last
Python reference after publication, allowing the snapshot to be reclaimed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SourceSnapshot:
    version: int
    profiles: np.ndarray

    def __post_init__(self) -> None:
        if self.profiles.ndim != 2:
            raise ValueError("SourceSnapshot.profiles must be 2-D")
        if self.profiles.flags.writeable:
            raise ValueError("SourceSnapshot.profiles must be read-only")


def pin_array(profiles: np.ndarray, version: int = 0) -> SourceSnapshot:
    """Copy a table and freeze it as an immutable publication source."""
    values = np.array(profiles, copy=True, order="C")
    values.flags.writeable = False
    return SourceSnapshot(version=int(version), profiles=values)


def as_snapshot(source: SourceSnapshot | np.ndarray, version: int = 0) -> SourceSnapshot:
    if isinstance(source, SourceSnapshot):
        return source
    return pin_array(source, version=version)
