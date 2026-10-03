# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""The features and regions of interest of the legacy ``FeatureExtraction`` class (Annex J.2)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from pyippdme.protocol.ast import Argument, BasicName, NamedValue, Number, String
from pyippdme.types.vec3 import Vec3


class Shape(Protocol):
    """Something that is sent as one named argument, such as ``Circle(...)``."""

    def to_argument(self) -> NamedValue:
        """Return the shape as the argument it is sent as."""
        ...


def _shape(name: str, *values: float | Vec3) -> NamedValue:
    numbers: list[float] = []
    for value in values:
        if isinstance(value, tuple):
            numbers.extend(value)
        else:
            numbers.append(value)
    return NamedValue(name, tuple(Number.of(v) for v in numbers))


@dataclass(frozen=True, slots=True)
class CylinderShape:
    """``Cylinder(x,y,z,i,j,k,l,m,n,r,h)``: a feature to extract or a region of interest."""

    position: Vec3
    normal: Vec3
    orientation: Vec3
    radius: float
    height: float

    def to_argument(self) -> NamedValue:
        return _shape(
            "Cylinder", self.position, self.normal, self.orientation, self.radius, self.height
        )


@dataclass(frozen=True, slots=True)
class CircleShape:
    """``Circle(x,y,z,i,j,k,l,m,n,r)``: a feature to extract or a region of interest."""

    position: Vec3
    normal: Vec3
    orientation: Vec3
    radius: float

    def to_argument(self) -> NamedValue:
        return _shape("Circle", self.position, self.normal, self.orientation, self.radius)


@dataclass(frozen=True, slots=True)
class LineShape:
    """``Line(x,y,z,i,j,k,l)``: a feature to extract or a region of interest."""

    position: Vec3
    direction: Vec3
    length: float

    def to_argument(self) -> NamedValue:
        return _shape("Line", self.position, self.direction, self.length)


@dataclass(frozen=True, slots=True)
class PointShape:
    """``Point(x,y,z,i,j,k)``: a feature to extract."""

    position: Vec3
    normal: Vec3

    def to_argument(self) -> NamedValue:
        return _shape("Point", self.position, self.normal)


@dataclass(frozen=True, slots=True)
class SphereShape:
    """``Sphere(x,y,z,i,j,k,l,m,n,r)``: a region of interest."""

    position: Vec3
    normal: Vec3
    orientation: Vec3
    radius: float

    def to_argument(self) -> NamedValue:
        return _shape("Sphere", self.position, self.normal, self.orientation, self.radius)


@dataclass(frozen=True, slots=True)
class CircleRel:
    """``CircleRel(dr)``: a region around a circle feature, wider by ``dr``."""

    dr: float

    def to_argument(self) -> NamedValue:
        return _shape("CircleRel", self.dr)


@dataclass(frozen=True, slots=True)
class CylinderRel:
    """``CylinderRel(dr,dh)``: a region around a cylinder feature, larger by ``dr`` and ``dh``."""

    dr: float
    dh: float

    def to_argument(self) -> NamedValue:
        return _shape("CylinderRel", self.dr, self.dh)


@dataclass(frozen=True, slots=True)
class SphereRel:
    """``SphereRel(dr)``: a region around a sphere feature, wider by ``dr``."""

    dr: float

    def to_argument(self) -> NamedValue:
        return _shape("SphereRel", self.dr)


@dataclass(frozen=True, slots=True)
class Polygon2D:
    """``Polygon2D(x1,y1,z1,...,xn,yn,zn)``: a closed polygon without loops."""

    points: Sequence[Vec3]

    def to_argument(self) -> NamedValue:
        return _shape("Polygon2D", *self.points)


@dataclass(frozen=True, slots=True)
class Polygon3D:
    """``Polygon3D(x1,y1,z1,...,xn,yn,zn)``: a closed polygon without loops."""

    points: Sequence[Vec3]

    def to_argument(self) -> NamedValue:
        return _shape("Polygon3D", *self.points)


def roi(name: str, shape: Shape, include: bool) -> tuple[Argument, ...]:
    """Build ``ROI``'s argument list."""
    return (String(name), shape.to_argument(), BasicName("include" if include else "exclude"))


def feature_extract(
    acquisitions: Sequence[str],
    feature: Shape,
    rois: Sequence[str],
    settings_name: str,
    step_width: float | None,
) -> tuple[Argument, ...]:
    """Build ``FeatureExtract``'s argument list.

    Without ``step_width`` the report is the geometric element (``GeoElem()``),
    otherwise the qualified edge points at that step width (``QEPs(S(...))``).
    """
    report = (
        NamedValue("GeoElem", ())
        if step_width is None
        else NamedValue("QEPs", (NamedValue("S", (Number.of(step_width),)),))
    )
    return (
        NamedValue("Acqs", tuple(String(name) for name in acquisitions)),
        feature.to_argument(),
        NamedValue("ROIs", tuple(String(name) for name in rois)),
        String(settings_name),
        report,
    )
