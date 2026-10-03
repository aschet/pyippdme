# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Nominal features of a test artefact and their evaluation from measured points.

Pure numpy, no CAD: a feature knows its nominal geometry (as it is *in machine
coordinates*, after the artefact is placed), selects the measured points that
belong to it, fits them, and reports size, position and form against the
nominal values and the machine's maximum permissible errors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from pyippdme.twin.spec import Accuracy

Points = NDArray[np.float64]
Vector = NDArray[np.float64]

#: Points this close to a nominal surface belong to its feature (mm).
SELECT_BAND = 1.5


@dataclass(frozen=True, slots=True)
class Result:
    """One evaluated quantity of a feature."""

    feature: str
    quantity: str
    nominal: float
    measured: float
    #: Limit of the error (MPE) in millimetres; ``None`` if the quantity has none.
    limit: float | None
    points: int

    @property
    def error(self) -> float:
        return self.measured - self.nominal

    @property
    def ok(self) -> bool | None:
        return None if self.limit is None else abs(self.error) <= self.limit


def _unit(v: Vector) -> Vector:
    return v / np.linalg.norm(v)


def _frame(axis: Vector) -> tuple[Vector, Vector]:
    helper = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = _unit(np.cross(axis, helper))
    return u, np.cross(axis, u)


class Feature:
    """Base of the nominal features."""

    name: str
    kind: str

    def distance(self, points: Points) -> Points:
        """Unsigned distance of each point to the nominal surface of the feature."""
        raise NotImplementedError

    def inside_extent(self, points: Points) -> NDArray[np.bool_]:
        """Which points lie within the finite extent of the feature."""
        return np.ones(len(points), dtype=bool)

    def select(self, points: Points, band: float = SELECT_BAND) -> Points:
        if len(points) == 0:
            return points
        mask = (self.distance(points) <= band) & self.inside_extent(points)
        return points[mask]

    def evaluate(self, points: Points, accuracy: Accuracy) -> list[Result]:
        raise NotImplementedError

    def transformed(self, m: NDArray[np.float64]) -> Feature:
        raise NotImplementedError


def _apply(m: NDArray[np.float64], p: Vector) -> Vector:
    return m[:3, :3] @ p + m[:3, 3]


@dataclass(frozen=True, slots=True)
class Sphere(Feature):
    """A sphere (or a cap of one: ``cap_axis``/``cap_radius`` limit the part that is measured)."""

    name: str
    center: Vector
    radius: float
    cap_axis: Vector | None = None
    cap_radius: float = 0.0
    kind: str = field(default="sphere", init=False)

    def distance(self, points: Points) -> Points:
        return np.asarray(np.abs(np.linalg.norm(points - self.center, axis=1) - self.radius))

    def inside_extent(self, points: Points) -> NDArray[np.bool_]:
        if self.cap_axis is None:
            return np.ones(len(points), dtype=bool)
        d = points - self.center
        along = d @ self.cap_axis
        radial = np.linalg.norm(d - np.outer(along, self.cap_axis), axis=1)
        return np.asarray((along > 0) & (radial <= self.cap_radius))

    def evaluate(self, points: Points, accuracy: Accuracy) -> list[Result]:
        p = self.select(points)
        n = len(p)
        if n < 4:
            return []
        center, radius = fit_sphere(p)
        residual = np.linalg.norm(p - center, axis=1) - radius
        if self.cap_axis is None:
            return [
                Result(
                    self.name,
                    "diameter",
                    2 * self.radius,
                    2 * radius,
                    accuracy.probing_um * 2e-3 + accuracy.length_error_mm(2 * self.radius),
                    n,
                ),
                Result(
                    self.name,
                    "position",
                    0.0,
                    float(np.linalg.norm(center - self.center)),
                    accuracy.length_error_mm(float(np.linalg.norm(self.center))),
                    n,
                ),
                Result(
                    self.name,
                    "form",
                    0.0,
                    float(residual.max() - residual.min()),
                    accuracy.probing_um * 1e-3,
                    n,
                ),
            ]
        deviation = np.linalg.norm(p - self.center, axis=1) - self.radius
        return [
            Result(
                self.name,
                "max deviation",
                0.0,
                float(np.abs(deviation).max()),
                accuracy.probing_um * 1e-3 * 2,
                n,
            ),
            Result(self.name, "rms deviation", 0.0, float(np.sqrt(np.mean(deviation**2))), None, n),
        ]

    def transformed(self, m: NDArray[np.float64]) -> Sphere:
        axis = None if self.cap_axis is None else m[:3, :3] @ self.cap_axis
        return Sphere(self.name, _apply(m, self.center), self.radius, axis, self.cap_radius)


@dataclass(frozen=True, slots=True)
class Plane(Feature):
    """A finite plane face; ``half_u``/``half_v`` give its size around ``center``."""

    name: str
    center: Vector
    normal: Vector
    half_u: float
    half_v: float
    kind: str = field(default="plane", init=False)

    def _uv(self) -> tuple[Vector, Vector]:
        return _frame(_unit(self.normal))

    def distance(self, points: Points) -> Points:
        return np.abs((points - self.center) @ _unit(self.normal))

    def inside_extent(self, points: Points) -> NDArray[np.bool_]:
        u, v = self._uv()
        d = points - self.center
        return (np.abs(d @ u) <= self.half_u + 0.5) & (np.abs(d @ v) <= self.half_v + 0.5)

    def position(self, points: Points) -> float:
        """Mean position of the selected points along the nominal normal."""
        return float(np.mean(self.select(points) @ _unit(self.normal)))

    def evaluate(self, points: Points, accuracy: Accuracy) -> list[Result]:
        p = self.select(points)
        n = len(p)
        if n < 3:
            return []
        normal = _unit(self.normal)
        offset = float(np.mean(p @ normal))
        nominal = float(self.center @ normal)
        results = [
            Result(
                self.name,
                "position",
                nominal,
                offset,
                accuracy.length_error_mm(float(np.linalg.norm(self.center))),
                n,
            )
        ]
        if n >= 4:
            centred = p - p.mean(axis=0)
            _, _, vt = np.linalg.svd(centred, full_matrices=False)
            residual = centred @ vt[2]
            results.append(
                Result(
                    self.name,
                    "flatness",
                    0.0,
                    float(residual.max() - residual.min()),
                    accuracy.probing_um * 1e-3 * 2,
                    n,
                )
            )
        return results

    def transformed(self, m: NDArray[np.float64]) -> Plane:
        return Plane(
            self.name, _apply(m, self.center), m[:3, :3] @ self.normal, self.half_u, self.half_v
        )


@dataclass(frozen=True, slots=True)
class Cylinder(Feature):
    """A cylinder; ``inside`` for a bore or ring gauge. ``height`` limits it along the axis."""

    name: str
    point: Vector
    axis: Vector
    radius: float
    inside: bool
    z_min: float
    z_max: float
    kind: str = field(default="cylinder", init=False)

    def _radial(self, points: Points) -> tuple[Points, Points]:
        a = _unit(self.axis)
        d = points - self.point
        along = d @ a
        radial = d - np.outer(along, a)
        return along, radial

    def distance(self, points: Points) -> Points:
        _, radial = self._radial(points)
        return np.asarray(np.abs(np.linalg.norm(radial, axis=1) - self.radius))

    def inside_extent(self, points: Points) -> NDArray[np.bool_]:
        along, _ = self._radial(points)
        return (along >= self.z_min - 0.5) & (along <= self.z_max + 0.5)

    def evaluate(self, points: Points, accuracy: Accuracy) -> list[Result]:
        p = self.select(points)
        n = len(p)
        if n < 4:
            return []
        a = _unit(self.axis)
        u, v = _frame(a)
        d = p - self.point
        xy = np.column_stack([d @ u, d @ v])
        center, radius = fit_circle(xy)
        residual = np.linalg.norm(xy - center, axis=1) - radius
        quantity = "diameter (bore)" if self.inside else "diameter"
        return [
            Result(
                self.name,
                quantity,
                2 * self.radius,
                2 * radius,
                accuracy.length_error_mm(2 * self.radius) + accuracy.probing_um * 1e-3,
                n,
            ),
            Result(
                self.name,
                "position",
                0.0,
                float(np.linalg.norm(center)),
                accuracy.length_error_mm(float(np.linalg.norm(self.point))),
                n,
            ),
            Result(
                self.name,
                "roundness",
                0.0,
                float(residual.max() - residual.min()),
                accuracy.probing_um * 1e-3 * 2,
                n,
            ),
        ]

    def transformed(self, m: NDArray[np.float64]) -> Cylinder:
        return Cylinder(
            self.name,
            _apply(m, self.point),
            m[:3, :3] @ self.axis,
            self.radius,
            self.inside,
            self.z_min,
            self.z_max,
        )


@dataclass(frozen=True, slots=True)
class Cone(Feature):
    """A frustum: radius ``r1`` at the base point, ``r2`` at ``height`` along the axis."""

    name: str
    base: Vector
    axis: Vector
    r1: float
    r2: float
    height: float
    kind: str = field(default="cone", init=False)

    def _tr(self, points: Points) -> tuple[Points, Points]:
        a = _unit(self.axis)
        d = points - self.base
        t = d @ a
        r = np.linalg.norm(d - np.outer(t, a), axis=1)
        return t, r

    def distance(self, points: Points) -> Points:
        t, r = self._tr(points)
        a = np.array([0.0, self.r1])
        b = np.array([self.height, self.r2])
        ab = b - a
        q = np.column_stack([t, r]) - a
        s = np.clip(q @ ab / (ab @ ab), 0.0, 1.0)
        return np.asarray(np.linalg.norm(q - np.outer(s, ab), axis=1))

    def evaluate(self, points: Points, accuracy: Accuracy) -> list[Result]:
        p = self.select(points)
        n = len(p)
        if n < 4:
            return []
        dist = self.distance(p)
        return [
            Result(
                self.name,
                "max deviation",
                0.0,
                float(dist.max()),
                accuracy.probing_um * 1e-3 * 2,
                n,
            ),
            Result(self.name, "rms deviation", 0.0, float(np.sqrt(np.mean(dist**2))), None, n),
        ]

    def transformed(self, m: NDArray[np.float64]) -> Cone:
        return Cone(
            self.name, _apply(m, self.base), m[:3, :3] @ self.axis, self.r1, self.r2, self.height
        )


@dataclass(frozen=True, slots=True)
class Length:
    """A length between two parallel plane faces (step gauge, groove, wall thickness)."""

    name: str
    first: str
    second: str
    nominal: float

    def evaluate(
        self, planes: dict[str, Plane], points: Points, accuracy: Accuracy
    ) -> Result | None:
        a, b = planes[self.first], planes[self.second]
        pa, pb = a.select(points), b.select(points)
        if len(pa) < 3 or len(pb) < 3:
            return None
        normal = _unit(a.normal)
        measured = abs(float(np.mean(pb @ normal) - np.mean(pa @ normal)))
        return Result(
            self.name,
            "length",
            self.nominal,
            measured,
            accuracy.length_error_mm(abs(self.nominal)),
            len(pa) + len(pb),
        )


def fit_sphere(points: Points) -> tuple[Vector, float]:
    """Least-squares sphere through ``points`` (algebraic, then refined geometrically)."""
    a = np.column_stack([2 * points, np.ones(len(points))])
    b = (points**2).sum(axis=1)
    x, *_ = np.linalg.lstsq(a, b, rcond=None)
    center = x[:3]
    radius = math.sqrt(max(float(x[3] + center @ center), 0.0))
    for _ in range(5):
        d = points - center
        dist = np.linalg.norm(d, axis=1)
        if np.any(dist < 1e-12):
            break
        jac = -d / dist[:, None]
        jac = np.column_stack([jac, -np.ones(len(points))])
        step, *_ = np.linalg.lstsq(jac, dist - radius, rcond=None)
        center = center - step[:3]
        radius = radius - step[3]
    return center, float(radius)


def fit_circle(xy: NDArray[np.float64]) -> tuple[Vector, float]:
    """Least-squares circle in the plane (algebraic, then refined)."""
    a = np.column_stack([2 * xy, np.ones(len(xy))])
    b = (xy**2).sum(axis=1)
    x, *_ = np.linalg.lstsq(a, b, rcond=None)
    center = x[:2]
    radius = math.sqrt(max(float(x[2] + center @ center), 0.0))
    for _ in range(5):
        d = xy - center
        dist = np.linalg.norm(d, axis=1)
        if np.any(dist < 1e-12):
            break
        jac = np.column_stack([-d / dist[:, None], -np.ones(len(xy))])
        step, *_ = np.linalg.lstsq(jac, dist - radius, rcond=None)
        center = center - step[:2]
        radius = radius - step[2]
    return center, float(radius)


def evaluate_features(
    features: list[Feature], lengths: list[Length], points: Points, accuracy: Accuracy
) -> list[Result]:
    """Evaluate every feature and length that has enough points among ``points``."""
    results: list[Result] = []
    for feature in features:
        results.extend(feature.evaluate(points, accuracy))
    planes = {f.name: f for f in features if isinstance(f, Plane)}
    for length in lengths:
        result = length.evaluate(planes, points, accuracy)
        if result is not None:
            results.append(result)
    return results
