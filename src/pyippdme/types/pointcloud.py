# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Raw point-cloud XML payload (Annex E, referenced by 6.17.1).

A ``PointCloudSet`` holds one or more ``PointCloud``s (e.g. one per scan
pass); each ``PointCloud`` holds one or more ``PointSet``s (Mx/My/Mz is a
surface normal, I/J/K a scan/measurement direction, both shared by every
point in the set); each ``PointSet`` holds one or more ``MeasPoint``s
(X/Y/Z plus a quality byte ``Q``, c.f. the ``Q()`` property).
"""

from __future__ import annotations

from dataclasses import dataclass
from xml.etree import ElementTree as ET  # nosec B405

from defusedxml.ElementTree import fromstring as defused_fromstring

from pyippdme.exceptions import IppDmeProtocolError
from pyippdme.types.vec3 import Vec3


@dataclass(frozen=True, slots=True)
class MeasPoint:
    """A single measured point: ``X()``/``Y()``/``Z()`` plus a quality byte ``Q()``."""

    x: float
    y: float
    z: float
    q: int


@dataclass(frozen=True, slots=True)
class PointSet:
    """A set of measuring points sharing one surface normal and scan direction."""

    normal: Vec3
    direction: Vec3
    points: tuple[MeasPoint, ...]


@dataclass(frozen=True, slots=True)
class PointCloud:
    """A single point cloud: one or more :class:`PointSet`."""

    point_sets: tuple[PointSet, ...]


@dataclass(frozen=True, slots=True)
class PointCloudSet:
    """A set of point clouds, e.g. one per scan pass."""

    point_clouds: tuple[PointCloud, ...]


def to_xml(data: PointCloudSet) -> str:
    root = ET.Element("PointCloudSet")
    for cloud in data.point_clouds:
        cloud_el = ET.SubElement(root, "PointCloud")
        for pset in cloud.point_sets:
            pset_el = ET.SubElement(cloud_el, "PointSet")
            for tag, value in zip(("Mx", "My", "Mz"), pset.normal, strict=True):
                ET.SubElement(pset_el, tag).text = repr(value)
            for tag, value in zip(("I", "J", "K"), pset.direction, strict=True):
                ET.SubElement(pset_el, tag).text = repr(value)
            for point in pset.points:
                mp_el = ET.SubElement(pset_el, "MeasPoint")
                ET.SubElement(mp_el, "X").text = repr(point.x)
                ET.SubElement(mp_el, "Y").text = repr(point.y)
                ET.SubElement(mp_el, "Z").text = repr(point.z)
                ET.SubElement(mp_el, "Q").text = str(point.q)
    return ET.tostring(root, encoding="unicode")


def from_xml(text: str) -> PointCloudSet:
    """Parse ``text``; may carry data from an untrusted remote peer, so uses ``defusedxml``."""
    try:
        root = defused_fromstring(text)
    except (ET.ParseError, ValueError) as exc:
        raise IppDmeProtocolError(f"Invalid point cloud XML: {exc}") from exc
    if root.tag != "PointCloudSet":
        raise IppDmeProtocolError("Expected a <PointCloudSet> root element")
    clouds = []
    for cloud_el in root.findall("PointCloud"):
        point_sets = []
        for pset_el in cloud_el.findall("PointSet"):
            normal = (_f(pset_el, "Mx"), _f(pset_el, "My"), _f(pset_el, "Mz"))
            direction = (_f(pset_el, "I"), _f(pset_el, "J"), _f(pset_el, "K"))
            points = tuple(
                MeasPoint(_f(mp, "X"), _f(mp, "Y"), _f(mp, "Z"), int(mp.findtext("Q") or "0"))
                for mp in pset_el.findall("MeasPoint")
            )
            point_sets.append(PointSet(normal, direction, points))
        clouds.append(PointCloud(tuple(point_sets)))
    return PointCloudSet(tuple(clouds))


def _f(element: ET.Element, tag: str) -> float:
    return float(element.findtext(tag) or "0")
