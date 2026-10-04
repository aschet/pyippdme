# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""The deprecated ``FeatureExtraction`` class (Annex J.2): ``ROI`` and ``FeatureExtract``.

``ROI("Name", Type(...), include|exclude)`` defines a region of interest; ``FeatureExtract`` takes
the points of buffered acquisitions, keeps those inside the included regions and outside the
excluded ones, fits the requested primitive and reports it (``GeoElem()``) or the points that
were used (``QEPs(S(step))``). The standard says nothing about the encoding of several things,
so these are guesses: a plane region (circle, polygon) is a prism along its normal, a line
region is a tube of 1 mm radius, ``FeatExtractSettingsName`` is accepted and not used, and the
relative regions (``CircleRel``, ``CylinderRel``, ``SphereRel``) are the feature itself made
wider or taller by the given amounts. The fit is least squares around the nominal axis and
direction that the feature carries.
"""

from __future__ import annotations

import math
from collections.abc import AsyncIterator
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pyippdme.protocol.ast import (
    Argument,
    BasicName,
    Items,
    NamedValue,
    Number,
    NumericData,
    String,
)
from pyippdme.protocol.commands import CommandName
from pyippdme.protocol.errors import ErrorCode, ErrorSeverity, ServerError
from pyippdme.server._util import bad_argument
from pyippdme.server.registry import CommandRegistry, HandlerResult
from pyippdme.simulation.classes.scanning_class import report_values
from pyippdme.simulation.context import Ctx
from pyippdme.twin.features import fit_circle

__all__ = ["Roi", "register"]

#: How many numbers each shape takes (``None``: any multiple of three, at least three points).
_SHAPE_NUMBERS: dict[str, int | None] = {
    "Line": 7,
    "Circle": 10,
    "CircleRel": 1,
    "Polygon2D": None,
    "Cylinder": 11,
    "CylinderRel": 2,
    "Sphere": 10,
    "SphereRel": 1,
    "Polygon3D": None,
}
_FEATURE_NUMBERS = {"Point": 6, "Line": 7, "Circle": 10, "Cylinder": 11}
#: A line region is a tube of this radius in millimetres.
_LINE_TUBE = 1.0

Points = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Roi:
    """A region of interest: the shape, its numbers, and whether it includes or excludes."""

    kind: str
    values: tuple[float, ...]
    include: bool


def _numbers(arg: Argument, cause: str) -> tuple[str, tuple[float, ...]]:
    if not isinstance(arg, NamedValue) or not all(isinstance(a, Number) for a in arg.args):
        raise bad_argument(cause, "Expected a shape such as Circle(x,y,z,i,j,k,l,m,n,r)")
    return arg.name, tuple(a.value for a in arg.args if isinstance(a, Number))


async def _roi(ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
    cause = CommandName.ROI
    if len(args) != 3 or not isinstance(args[0], String) or not isinstance(args[2], BasicName):
        raise bad_argument(cause, 'Expected ROI("Name", Type(...), include|exclude)')
    kind, values = _numbers(args[1], cause)
    if kind not in _SHAPE_NUMBERS:
        raise bad_argument(cause, f"Unknown region type {kind}")
    expected = _SHAPE_NUMBERS[kind]
    if expected is None:
        if len(values) < 9 or len(values) % 3:
            raise bad_argument(cause, f"{kind} needs at least three points of three numbers")
    elif len(values) != expected:
        raise bad_argument(cause, f"{kind} needs {expected} numbers")
    if args[2].value not in ("include", "exclude"):
        raise bad_argument(cause, "The last argument is include or exclude")
    ctx.state.features.rois[args[0].value] = Roi(kind, values, args[2].value == "include")
    return None


def _unit(v: NDArray[np.float64]) -> NDArray[np.float64]:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else np.array([0.0, 0.0, 1.0])


def _inside_circle(
    points: Points, centre: Points, normal: Points, radius: float
) -> NDArray[np.bool_]:
    """Points whose projection along ``normal`` falls into the circle (a prism)."""
    d = points - centre
    d = d - np.outer(d @ normal, normal)
    return np.asarray(np.linalg.norm(d, axis=1) <= radius)


def _inside_polygon(points: Points, vertices: Points) -> NDArray[np.bool_]:
    """Points whose projection on the plane of ``vertices`` is inside the polygon."""
    normal = _unit(np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0]))
    u = _unit(vertices[1] - vertices[0])
    v = np.cross(normal, u)
    poly = np.column_stack([(vertices - vertices[0]) @ u, (vertices - vertices[0]) @ v])
    rel = points - vertices[0]
    xy = np.column_stack([rel @ u, rel @ v])
    inside = np.zeros(len(points), dtype=bool)
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        crosses = (yi > xy[:, 1]) != (yj > xy[:, 1])
        slope = (xj - xi) * (xy[:, 1] - yi) / (yj - yi + 1e-300) + xi
        inside ^= crosses & (xy[:, 0] < slope)
        j = i
    return inside


def _inside(roi: Roi, points: Points, feature: tuple[str, tuple[float, ...]]) -> NDArray[np.bool_]:
    v = np.asarray(roi.values, dtype=float)
    kind = roi.kind
    f_kind, f = feature
    fv = np.asarray(f, dtype=float)
    if kind == "Line":
        position, direction, length = v[:3], _unit(v[3:6]), v[6]
        rel = points - position
        along = rel @ direction
        radial = np.linalg.norm(rel - np.outer(along, direction), axis=1)
        return np.asarray((np.abs(along) <= length / 2.0 + 1e-9) & (radial <= _LINE_TUBE))
    if kind == "Circle":
        return _inside_circle(points, v[:3], _unit(v[3:6]), v[9])
    if kind == "CircleRel":
        if f_kind != "Circle":
            return np.ones(len(points), dtype=bool)
        return _inside_circle(points, fv[:3], _unit(fv[3:6]), fv[9] + v[0])
    if kind in ("Cylinder", "CylinderRel"):
        if kind == "Cylinder":
            position, axis, radius, height = v[:3], _unit(v[3:6]), v[9], v[10]
        elif f_kind == "Cylinder":
            position, axis = fv[:3], _unit(fv[3:6])
            radius, height = fv[9] + v[0], fv[10] + v[1]
        else:
            return np.ones(len(points), dtype=bool)
        rel = points - position
        along = rel @ axis
        radial = np.linalg.norm(rel - np.outer(along, axis), axis=1)
        return np.asarray((np.abs(along) <= height / 2.0) & (radial <= radius))
    if kind in ("Sphere", "SphereRel"):
        centre, radius = v[:3], v[9] if kind == "Sphere" else v[0]
        return np.asarray(np.linalg.norm(points - centre, axis=1) <= radius)
    # Polygon2D / Polygon3D
    return _inside_polygon(points, v.reshape(-1, 3))


def _collect(ctx: Ctx, names: list[str]) -> Points:
    rows: list[tuple[float, float, float]] = []
    for name in names:
        cloud_set = ctx.state.raw_data.acquisitions.get(name)
        if cloud_set is None:
            raise ServerError(
                ErrorSeverity.CRITICAL,
                ErrorCode.RAW_DATA_NOT_AVAILABLE,
                CommandName.FEATURE_EXTRACT,
                f"Acquisition {name} is not available",
            )
        for cloud in cloud_set.point_clouds:
            for point_set in cloud.point_sets:
                rows.extend((p.x, p.y, p.z) for p in point_set.points)
    return np.asarray(rows, dtype=float).reshape(-1, 3)


def _fit(kind: str, nominal: tuple[float, ...], points: Points) -> NamedValue:
    """Fit the primitive to ``points`` and return it in the shape of its own definition."""
    n = np.asarray(nominal, dtype=float)
    centroid = points.mean(axis=0)
    if kind == "Point":
        return NamedValue("Point", tuple(Number.of(float(c)) for c in (*centroid, *n[3:6])))
    if kind == "Line":
        _, _, vt = np.linalg.svd(points - centroid)
        direction = vt[0] if float(vt[0] @ _unit(n[3:6])) >= 0 else -vt[0]
        extent = (points - centroid) @ direction
        length = float(extent.max() - extent.min())
        values = (*centroid, *direction, length)
        return NamedValue("Line", tuple(Number.of(float(c)) for c in values))
    axis = _unit(n[3:6])
    u = _unit(
        np.cross(axis, [1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.cross(axis, [0, 1.0, 0])
    )
    v = np.cross(axis, u)
    rel = points - centroid
    xy = np.column_stack([rel @ u, rel @ v])
    (cx, cy), radius = fit_circle(xy)
    centre = centroid + cx * u + cy * v
    along = rel @ axis
    if kind == "Circle":
        values = (*(centre - axis * 0.0), *axis, *n[6:9], radius)
        return NamedValue("Circle", tuple(Number.of(float(c)) for c in values))
    height = float(along.max() - along.min())
    values = (*centre, *axis, *n[6:9], radius, height)
    return NamedValue("Cylinder", tuple(Number.of(float(c)) for c in values))


async def _feature_extract(ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
    cause = CommandName.FEATURE_EXTRACT
    if len(args) != 5:
        raise bad_argument(
            cause, "Expected Acqs(..), a feature, ROIs(..), a settings name, a report"
        )
    acqs, feature, rois, settings, report = args
    if not (
        isinstance(acqs, NamedValue)
        and acqs.name == "Acqs"
        and all(isinstance(a, String) for a in acqs.args)
        and acqs.args
    ):
        raise bad_argument(cause, "Acqs needs the names of at least one acquisition")
    if not (
        isinstance(rois, NamedValue)
        and rois.name == "ROIs"
        and all(isinstance(a, String) for a in rois.args)
    ):
        raise bad_argument(cause, "ROIs needs region names")
    if not isinstance(settings, String) or not isinstance(report, NamedValue):
        raise bad_argument(cause)
    kind, nominal = _numbers(feature, cause)
    if kind not in _FEATURE_NUMBERS or len(nominal) != _FEATURE_NUMBERS[kind]:
        raise bad_argument(cause, "The feature is a Point, Line, Circle or Cylinder")
    regions = []
    for name in (a.value for a in rois.args if isinstance(a, String)):
        region = ctx.state.features.rois.get(name)
        if region is None:
            raise bad_argument(cause, f"No region of interest {name}")
        regions.append(region)
    names = [a.value for a in acqs.args if isinstance(a, String)]
    points = _collect(ctx, names)
    include = [r for r in regions if r.include]
    keep = np.zeros(len(points), dtype=bool) if include else np.ones(len(points), dtype=bool)
    for region in include:
        keep |= _inside(region, points, (kind, nominal))
    for region in regions:
        if not region.include:
            keep &= ~_inside(region, points, (kind, nominal))
    points = points[keep]
    minimum = {"Point": 1, "Line": 2, "Circle": 3, "Cylinder": 3}[kind]
    if len(points) < minimum:
        raise ServerError(
            ErrorSeverity.ERROR, ErrorCode.SURFACE_NOT_FOUND, cause, "Too few points in the regions"
        )
    if report.name == "GeoElem":
        return Items((_fit(kind, nominal, points),))
    if report.name == "QEPs":
        step = _step(report, cause)
        return _qeps(ctx, points, step, cause)
    raise bad_argument(cause, "The report is GeoElem() or QEPs(S(step))")


def _step(report: NamedValue, cause: str) -> float:
    if len(report.args) == 1 and isinstance(report.args[0], NamedValue):
        inner = report.args[0]
        if inner.name == "S" and len(inner.args) == 1 and isinstance(inner.args[0], Number):
            return float(inner.args[0].value)
    raise bad_argument(cause, "QEPs needs S(step width)")


def _qeps(ctx: Ctx, points: Points, step: float, cause: str) -> AsyncIterator[NumericData]:
    async def stream() -> AsyncIterator[NumericData]:
        last: NDArray[np.float64] | None = None
        for p in points:
            if last is not None and step > 0 and math.dist(p, last) < step:
                continue
            last = p
            point = (float(p[0]), float(p[1]), float(p[2]))
            yield NumericData(report_values(ctx, point, (0.0, 0.0, 1.0), cause))

    return stream()


def register(registry: CommandRegistry) -> None:
    registry.register(CommandName.ROI, _roi, arguments=None)
    registry.register(CommandName.FEATURE_EXTRACT, _feature_extract, arguments=None)
