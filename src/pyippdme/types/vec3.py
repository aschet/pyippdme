# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Minimal 3D vector helpers, shared across command handlers and the backend.

``Vec3`` stays a plain ``(x, y, z)`` tuple at the public boundary (it is part
of :class:`~pyippdme.server.backend.MachineBackend`, so implementers should not need
numpy just to yield a point), but the arithmetic itself is delegated to
numpy rather than hand-rolled, since numpy is already a dependency for
:mod:`pyippdme.types.csy`'s rotation matrices.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

Vec3 = tuple[float, float, float]


def to_array(v: Vec3) -> npt.NDArray[np.float64]:
    """Convert to a numpy array, for callers doing more than a single operation."""
    return np.asarray(v, dtype=np.float64)


def from_array(a: npt.NDArray[np.float64]) -> Vec3:
    return (float(a[0]), float(a[1]), float(a[2]))


def add(a: Vec3, b: Vec3) -> Vec3:
    return from_array(to_array(a) + to_array(b))


def sub(a: Vec3, b: Vec3) -> Vec3:
    return from_array(to_array(a) - to_array(b))


def scale(a: Vec3, s: float) -> Vec3:
    return from_array(to_array(a) * s)


def dot(a: Vec3, b: Vec3) -> float:
    return float(np.dot(to_array(a), to_array(b)))


def cross(a: Vec3, b: Vec3) -> Vec3:
    return from_array(np.cross(to_array(a), to_array(b)))


def norm(a: Vec3) -> float:
    return float(np.linalg.norm(to_array(a)))


def normalize(a: Vec3) -> Vec3:
    length = norm(a)
    if length == 0.0:
        raise ValueError("Cannot normalize the zero vector")
    return scale(a, 1.0 / length)
