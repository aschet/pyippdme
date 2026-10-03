# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""A depth buffer of a triangle mesh seen along one direction: fast ray hits for optical sensors.

Optical sensors fire many parallel rays (a laser line, a grid of pixels). Casting each
one against the CAD kernel is slow; instead the scene's triangles are rasterised once
for a viewing direction into an id buffer, and a ray is answered by looking up the
triangle at its pixel and intersecting that triangle's plane exactly. Pure numpy.

The buffer sees only the surface nearest to the viewer along each ray, so it cannot
represent what lies under an overhang. That is also what a top-down optical sensor sees.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

#: Triangles covering at most this many pixels per side are rasterised in batches.
_BATCH = 7


@dataclass(frozen=True, slots=True)
class Hits:
    """Answers for ``N`` rays: ``valid`` marks the ones that hit a surface ahead of the origin."""

    valid: NDArray[np.bool_]
    #: Distance from each origin along the viewing direction to the surface.
    distance: Array
    point: Array
    #: Unit surface normal, turned to face the sensor.
    normal: Array


class DepthBuffer:
    """Triangles ``faces`` of ``vertices`` rasterised along ``direction`` (towards the scene)."""

    def __init__(
        self, vertices: Array, faces: NDArray[np.int64], direction: Array, pixel: float = 0.2
    ):
        d = np.asarray(direction, dtype=float)
        self.direction = d / np.linalg.norm(d)
        helper = (
            np.array([1.0, 0.0, 0.0]) if abs(self.direction[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        )
        self.u = np.cross(self.direction, helper)
        self.u /= np.linalg.norm(self.u)
        self.v = np.cross(self.direction, self.u)
        self.pixel = pixel
        tri = vertices[faces] if len(faces) else np.zeros((0, 3, 3))
        raw = (
            np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]) if len(tri) else np.zeros((0, 3))
        )
        length = np.linalg.norm(raw, axis=1)
        keep = length > 1e-9  # degenerate triangles (poles of spheres) have no plane to hit
        tri, raw, length = tri[keep], raw[keep], length[keep]
        self._tri = tri
        self._normals = raw / length[:, None] if len(tri) else np.zeros((0, 3))
        self._offset = np.einsum("ij,ij->i", self._normals, tri[:, 0]) if len(tri) else np.zeros(0)
        if len(tri) == 0:
            self._origin = np.zeros(2)
            self.shape = (1, 1)
            self._ids = np.full(self.shape, -1, dtype=np.int64)
            self._height = np.full(self.shape, -np.inf)
            return
        a = tri @ self.u
        b = tri @ self.v
        h = -(tri @ self.direction)  # height towards the sensor
        self._origin = np.array([a.min() - pixel, b.min() - pixel])
        na = int(math.ceil((a.max() - self._origin[0]) / pixel)) + 2
        nb = int(math.ceil((b.max() - self._origin[1]) / pixel)) + 2
        self.shape = (na, nb)
        self._ids = np.full(self.shape, -1, dtype=np.int64)
        self._height = np.full(self.shape, -np.inf)
        self._rasterise(a, b, h)

    def _rasterise(self, a: Array, b: Array, h: Array) -> None:
        """Fill the id buffer: small triangles in vectorised batches, large ones one by one."""
        pixel = self.pixel
        o = self._origin
        i0 = np.maximum(np.floor((a.min(axis=1) - o[0]) / pixel).astype(np.int64), 0)
        i1 = np.minimum(np.ceil((a.max(axis=1) - o[0]) / pixel).astype(np.int64) + 1, self.shape[0])
        j0 = np.maximum(np.floor((b.min(axis=1) - o[1]) / pixel).astype(np.int64), 0)
        j1 = np.minimum(np.ceil((b.max(axis=1) - o[1]) / pixel).astype(np.int64) + 1, self.shape[1])
        span = np.maximum(i1 - i0, j1 - j0)
        small = np.flatnonzero((span <= _BATCH) & (i1 > i0) & (j1 > j0))
        for start in range(0, len(small), 20_000):
            self._batch(small[start : start + 20_000], a, b, h, i0, j0)
        for t in np.flatnonzero((span > _BATCH) & (i1 > i0) & (j1 > j0)):
            self._single(int(t), a[t], b[t], h[t], int(i0[t]), int(i1[t]), int(j0[t]), int(j1[t]))

    @staticmethod
    def _bary(
        ta: tuple[Array, Array, Array], tb: tuple[Array, Array, Array], ga: Array, gb: Array
    ) -> tuple[Array, Array, Array, Array]:
        """Barycentric weights of pixel centres ``(ga, gb)`` in triangles with corners ``ta``/``tb``."""
        det = (tb[1] - tb[2]) * (ta[0] - ta[2]) + (ta[2] - ta[1]) * (tb[0] - tb[2])
        safe = np.where(np.abs(det) > 1e-14, det, np.inf)
        w0 = ((tb[1] - tb[2]) * (ga - ta[2]) + (ta[2] - ta[1]) * (gb - tb[2])) / safe
        w1 = ((tb[2] - tb[0]) * (ga - ta[2]) + (ta[0] - ta[2]) * (gb - tb[2])) / safe
        return w0, w1, 1.0 - w0 - w1, det

    def _batch(
        self, ids: NDArray[np.int64], a: Array, b: Array, h: Array, i0: Array, j0: Array
    ) -> None:
        pixel, o = self.pixel, self._origin
        k = np.arange(_BATCH + 1)
        ii = i0[ids][:, None, None] + k[None, :, None]
        jj = j0[ids][:, None, None] + k[None, None, :]
        ga = o[0] + (ii + 0.5) * pixel
        gb = o[1] + (jj + 0.5) * pixel
        ta = tuple(a[ids, c][:, None, None] for c in range(3))
        tb = tuple(b[ids, c][:, None, None] for c in range(3))
        th = tuple(h[ids, c][:, None, None] for c in range(3))
        w0, w1, w2, _ = self._bary(ta, tb, ga, gb)
        height = w0 * th[0] + w1 * th[1] + w2 * th[2]
        eps = -1e-5
        inside = (
            (w0 >= eps) & (w1 >= eps) & (w2 >= eps) & (ii < self.shape[0]) & (jj < self.shape[1])
        )
        t_index, ki, kj = np.nonzero(inside)
        if len(t_index) == 0:
            return
        hh = height[t_index, ki, kj]
        flat = (i0[ids][t_index] + ki) * self.shape[1] + (j0[ids][t_index] + kj)
        tri = ids[t_index]
        order = np.lexsort((hh, flat))  # by pixel, then by height
        flat, hh, tri = flat[order], hh[order], tri[order]
        last = np.r_[flat[1:] != flat[:-1], True]  # the highest candidate of each pixel
        flat, hh, tri = flat[last], hh[last], tri[last]
        current = self._height.reshape(-1)
        better = hh > current[flat]
        current[flat[better]] = hh[better]
        self._ids.reshape(-1)[flat[better]] = tri[better]

    def _single(
        self, t: int, ta: Array, tb: Array, th: Array, i0: int, i1: int, j0: int, j1: int
    ) -> None:
        pixel, o = self.pixel, self._origin
        pa = o[0] + (np.arange(i0, i1) + 0.5) * pixel
        pb = o[1] + (np.arange(j0, j1) + 0.5) * pixel
        ga, gb = np.meshgrid(pa, pb, indexing="ij")
        w0, w1, w2, det = self._bary((ta[0], ta[1], ta[2]), (tb[0], tb[1], tb[2]), ga, gb)
        if abs(det) < 1e-14:
            return
        eps = -1e-5
        inside = (w0 >= eps) & (w1 >= eps) & (w2 >= eps)
        if not inside.any():
            return
        height = w0 * th[0] + w1 * th[1] + w2 * th[2]
        block_h = self._height[i0:i1, j0:j1]
        better = inside & (height > block_h)
        block_h[better] = height[better]
        self._ids[i0:i1, j0:j1][better] = t

    # -- queries -----------------------------------------------------------------------

    def _index(
        self, points: Array
    ) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.bool_]]:
        a = (points @ self.u - self._origin[0]) / self.pixel
        b = (points @ self.v - self._origin[1]) / self.pixel
        i = np.floor(a).astype(np.int64)
        j = np.floor(b).astype(np.int64)
        ok = (i >= 0) & (i < self.shape[0]) & (j >= 0) & (j < self.shape[1])
        return np.clip(i, 0, self.shape[0] - 1), np.clip(j, 0, self.shape[1] - 1), ok

    def height_at(self, points: Array) -> Array:
        """Height of the nearest surface (towards the sensor) above each point's ``(u, v)``.

        ``-inf`` where nothing is there. Heights are measured against the viewing direction,
        so a point ``p`` has height ``-p . direction``.
        """
        i, j, ok = self._index(points)
        return np.where(ok, self._height[i, j], -np.inf)

    def hit(self, origins: Array) -> Hits:
        """Intersect rays from ``origins`` along the viewing direction with the scene."""
        origins = np.asarray(origins, dtype=float).reshape(-1, 3)
        i, j, ok = self._index(origins)
        ids = self._ids[i, j]
        ok &= ids >= 0
        safe = np.where(ok, ids, 0)
        n = self._normals[safe]
        denom = n @ self.direction
        usable = ok & (np.abs(denom) > 1e-9)
        distance = np.where(
            usable,
            (self._offset[safe] - np.einsum("ij,ij->i", n, origins)) / np.where(usable, denom, 1.0),
            0.0,
        )
        valid = usable & (distance >= 0.0)
        point = origins + np.outer(distance, self.direction)
        facing = np.where((denom > 0)[:, None], -n, n)
        return Hits(valid, distance, point, facing)
