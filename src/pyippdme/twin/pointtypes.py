# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""The types of points an optical scanning procedure delivers (6.15, Figures 40 and 41).

* **RSL**, raw scan lines: the uncalculated raw point cloud, as the sensor saw it.
* **GSL**, gridded scan lines: the same kind of raw data, but on a regular grid of scan lines
  (the figure uses 1 mm).
* **QSP**, qualified surface points: a compressed cloud; every grid node is the average of the raw
  points around it, so the noise of the single points is averaged out.
* **QEP**, a qualified edge point: where two surfaces meet; computed from the two point clouds
  that lie on either side of the edge (Figure 41).

Pure numpy; points are ``(N, 3)`` arrays and ``view`` is the direction the sensor looks along.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Points = NDArray[np.float64]

POINT_TYPES = ("RSL", "GSL", "QSP")


def _frame(view: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    d = view / np.linalg.norm(view)
    helper = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(d, helper)
    u /= np.linalg.norm(u)
    return u, np.cross(d, u)


def _cells(
    points: Points, view: NDArray[np.float64], pitch: float
) -> tuple[Points, NDArray[np.int64]]:
    """Grid cell of each point in the plane perpendicular to ``view``, and the grid node centres."""
    u, v = _frame(np.asarray(view, dtype=float))
    uv = np.column_stack([points @ u, points @ v])
    index = np.floor(uv / pitch + 0.5).astype(np.int64)  # nearest node
    return uv, index


def to_gsl(points: Points, view: tuple[float, float, float], pitch: float = 1.0) -> Points:
    """Resample raw points onto grid scan lines: one point per grid node, the raw one nearest to it.

    The result lies exactly on the grid in the plane perpendicular to ``view``, with the depth
    of the nearest raw point; nodes without a raw point within half a pitch are left out.
    """
    if len(points) == 0:
        return points
    view_v = np.asarray(view, dtype=float)
    u, v = _frame(view_v)
    d = view_v / np.linalg.norm(view_v)
    uv, index = _cells(points, view_v, pitch)
    distance = np.linalg.norm(uv - index * pitch, axis=1)
    order = np.lexsort((distance, index[:, 1], index[:, 0]))
    keys = index[order]
    first = np.r_[True, np.any(keys[1:] != keys[:-1], axis=1)]
    chosen = order[first]
    chosen = chosen[distance[chosen] <= pitch / 2.0]
    node_uv = index[chosen] * pitch
    depth = points[chosen] @ d
    return np.asarray(node_uv[:, :1] * u + node_uv[:, 1:] * v + depth[:, None] * d)


def to_qsp(
    points: Points,
    view: tuple[float, float, float],
    pitch: float = 1.0,
    radius: float | None = None,
    minimum: int = 3,
) -> Points:
    """Qualified surface points: each grid node is the mean of the raw points within ``radius``.

    ``radius`` defaults to 0.4 of the pitch (the circles of Figure 40). Nodes with fewer than
    ``minimum`` points are dropped: they are not qualified.
    """
    if len(points) == 0:
        return points
    radius = 0.4 * pitch if radius is None else radius
    view_v = np.asarray(view, dtype=float)
    uv, index = _cells(points, view_v, pitch)
    near = np.linalg.norm(uv - index * pitch, axis=1) <= radius
    keys = index[near]
    selected = points[near]
    if len(selected) == 0:
        return np.asarray(selected)
    unique, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    sums = np.zeros((len(unique), 3))
    np.add.at(sums, inverse.reshape(-1), selected)
    means = sums / counts[:, None]
    return means[counts >= minimum]


def fit_plane(points: Points) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Least-squares plane: a point on it (the centroid) and its unit normal."""
    centroid = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - centroid, full_matrices=False)
    return centroid, vt[2]


def qualified_edge_point(
    side_a: Points, side_b: Points
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    """The qualified edge point of two point clouds on either side of an edge (Figure 41).

    Fits a plane to each cloud, intersects them, and returns the point of the intersection line
    that lies nearest to the middle of the data, with the direction of the line. ``None`` if a
    cloud has too few points or the planes are parallel.
    """
    if len(side_a) < 3 or len(side_b) < 3:
        return None
    ca, na = fit_plane(side_a)
    cb, nb = fit_plane(side_b)
    direction = np.cross(na, nb)
    length = float(np.linalg.norm(direction))
    if length < 1e-6:
        return None
    direction /= length
    # A point on both planes: solve n_a.p = n_a.c_a, n_b.p = n_b.c_b, direction.p = direction.mid.
    mid = (side_a.mean(axis=0) + side_b.mean(axis=0)) / 2.0
    system = np.vstack([na, nb, direction])
    rhs = np.array([na @ ca, nb @ cb, direction @ mid])
    p = np.linalg.solve(system, rhs)
    return (float(p[0]), float(p[1]), float(p[2])), (
        float(direction[0]),
        float(direction[1]),
        float(direction[2]),
    )
