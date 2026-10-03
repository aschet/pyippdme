# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Small numpy helpers for poses (4x4 homogeneous matrices) and triangle meshes."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Matrix = NDArray[np.float64]


def identity() -> Matrix:
    return np.eye(4)


def translation(x: float, y: float, z: float) -> Matrix:
    m = np.eye(4)
    m[:3, 3] = (x, y, z)
    return m


def rotation(axis: tuple[float, float, float], degrees: float) -> Matrix:
    """Rotation by ``degrees`` about ``axis`` through the origin (Rodrigues)."""
    a = np.asarray(axis, dtype=float)
    n = float(np.linalg.norm(a))
    if n == 0.0:
        return np.eye(4)
    a /= n
    t = math.radians(degrees)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    r = np.eye(3) + math.sin(t) * k + (1 - math.cos(t)) * (k @ k)
    m = np.eye(4)
    m[:3, :3] = r
    return m


def pose(
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    rx: float = 0.0,
    ry: float = 0.0,
    rz: float = 0.0,
) -> Matrix:
    """Build a pose: translation plus extrinsic rotations about X, then Y, then Z (degrees)."""
    return (
        translation(x, y, z)
        @ rotation((0, 0, 1), rz)
        @ rotation((0, 1, 0), ry)
        @ rotation((1, 0, 0), rx)
    )


def decompose(m: Matrix) -> tuple[float, float, float, float, float, float]:
    """Inverse of :func:`pose`: ``(x, y, z, rx, ry, rz)`` with angles in degrees."""
    r = m[:3, :3]
    ry = math.asin(max(-1.0, min(1.0, -float(r[2, 0]))))
    if abs(math.cos(ry)) > 1e-9:
        rx = math.atan2(float(r[2, 1]), float(r[2, 2]))
        rz = math.atan2(float(r[1, 0]), float(r[0, 0]))
    else:  # gimbal lock: fold everything into rz
        rx = 0.0
        rz = math.atan2(-float(r[0, 1]), float(r[1, 1]))
    x, y, z = (float(v) for v in m[:3, 3])
    return (x, y, z, math.degrees(rx), math.degrees(ry), math.degrees(rz))


def rotation_about(
    point: tuple[float, float, float], axis: tuple[float, float, float], degrees: float
) -> Matrix:
    """Rotation by ``degrees`` about the line through ``point`` along ``axis``."""
    p = translation(*point)
    return p @ rotation(axis, degrees) @ np.linalg.inv(p)


def apply(m: Matrix, points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Transform ``(N, 3)`` points by ``m``."""
    return points @ m[:3, :3].T + m[:3, 3]


def apply_dir(m: Matrix, vector: tuple[float, float, float]) -> tuple[float, float, float]:
    v = m[:3, :3] @ np.asarray(vector, dtype=float)
    return (float(v[0]), float(v[1]), float(v[2]))


@dataclass(frozen=True, slots=True)
class Mesh:
    """A triangle mesh: ``vertices`` ``(N, 3)`` and ``faces`` ``(M, 3)`` vertex indices."""

    vertices: NDArray[np.float64]
    faces: NDArray[np.int64]

    @property
    def is_empty(self) -> bool:
        return len(self.faces) == 0

    def bounds(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if len(self.vertices) == 0:
            zero = np.zeros(3)
            return zero, zero
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    def transformed(self, m: Matrix) -> Mesh:
        return Mesh(apply(m, self.vertices), self.faces)


EMPTY_MESH = Mesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))
