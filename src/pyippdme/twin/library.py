# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Component libraries: machines, sensors, heads, tips, extensions, rotary tables and racks.

A tool is built from components, the way it is built on a real machine: a head, an optional
extension, a sensor (the probe body) and a stylus with a tip. :func:`build_tool` puts the
named components together into a :class:`~pyippdme.twin.spec.ToolSpec`; the simulator window's
tool creator is a front end for it. The numbers are representative of public data sheet ranges,
not of one vendor's parts, and every entry can be replaced or extended::

    from pyippdme.twin.library import build_tool
    spec = build_tool("Scan6", sensor="scanning probe", head="indexing head (PH10)",
                      tip="ball 6 mm", stylus="stem 50 mm", extension="extension 100 mm")

Pure Python; importable without OpenCASCADE.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pyippdme.twin.spec import PRESETS, MachineSpec, ToolSpec

__all__ = [
    "CATEGORIES",
    "LIBRARY",
    "ROTARY_TABLES",
    "Component",
    "build_tool",
    "components",
    "machine_names",
]


@dataclass(frozen=True, slots=True)
class Component:
    """One part of the library: a name and the ``ToolSpec`` fields it sets."""

    category: str
    name: str
    description: str
    fields: dict[str, Any] = field(default_factory=dict)


def _c(category: str, name: str, description: str, **fields: Any) -> Component:
    return Component(category, name, description, fields)


#: In the order a tool is put together.
CATEGORIES = ("head", "extension", "sensor", "stylus", "tip")

_ALL: tuple[Component, ...] = (
    # -- heads: how the probe is oriented
    _c("head", "fixed mount", "No articulation; the probe always points down.", head="fixed"),
    _c(
        "head",
        "indexing head (PH10)",
        "Two axes in 7.5 degree steps; each position is qualified on its own.",
        head="indexed",
        index_step=7.5,
    ),
    _c(
        "head",
        "continuous head (5-axis)",
        "Two axes continuously, qualified once for all angles.",
        head="continuous",
    ),
    # -- extensions between head and probe
    _c("extension", "no extension", "The probe sits directly on the head.", extension=0.0),
    _c("extension", "extension 20 mm", "Short extension.", extension=20.0),
    _c("extension", "extension 50 mm", "Medium extension.", extension=50.0),
    _c("extension", "extension 100 mm", "Long extension.", extension=100.0),
    _c("extension", "extension 200 mm", "Very long extension; more bending.", extension=200.0),
    # -- sensors: the probe body and how it measures
    _c(
        "sensor",
        "touch trigger (kinematic)",
        "Kinematic touch trigger probe with three-lobe pre-travel.",
        mode="touch",
        probe_radius=10.0,
        probe_length=28.0,
        lobing_um=0.4,
        breakaway=True,
        color=(0.8, 0.1, 0.1),
    ),
    _c(
        "sensor",
        "touch trigger (strain gauge)",
        "Strain-gauge touch probe: low lobing, very repeatable.",
        mode="touch",
        probe_radius=11.0,
        probe_length=30.0,
        lobing_um=0.1,
        repeatability_um=0.1,
        breakaway=True,
        color=(0.3, 0.7, 0.4),
    ),
    _c(
        "sensor",
        "scanning probe",
        "Analog scanning probe, 2000 points per second.",
        mode="scanning",
        probe_radius=14.0,
        probe_length=50.0,
        scan_rate=2000.0,
        repeatability_um=0.1,
        unqualified_um=15.0,
        lobing_um=0.0,
        color=(0.15, 0.55, 0.85),
    ),
    _c(
        "sensor",
        "head-touch (5-axis)",
        "The head flicks the stylus out: fast single points.",
        mode="head_touch",
        probe_radius=14.0,
        probe_length=60.0,
        touch_speed=60.0,
        repeatability_um=0.2,
        unqualified_um=8.0,
        color=(0.1, 0.45, 0.8),
    ),
    _c(
        "sensor",
        "laser line scanner",
        "Laser triangulation line, 320 points per line.",
        mode="laser",
        standoff=60.0,
        depth_range=30.0,
        line_width=30.0,
        points_per_line=320,
        noise_um=3.0,
        point_type="RSL",
        probe_radius=20.0,
        probe_length=70.0,
        color=(0.9, 0.2, 0.2),
    ),
    _c(
        "sensor",
        "laser point sensor",
        "Single-point laser distance sensor.",
        mode="point_laser",
        standoff=30.0,
        depth_range=10.0,
        line_width=0.0,
        noise_um=1.5,
        probe_radius=12.0,
        probe_length=50.0,
        color=(0.9, 0.3, 0.3),
    ),
    _c(
        "sensor",
        "structured-light area scanner",
        "Fringe projection; a whole field per shot.",
        mode="area",
        standoff=150.0,
        depth_range=60.0,
        line_width=60.0,
        field_height=45.0,
        noise_um=8.0,
        probe_radius=40.0,
        probe_length=90.0,
        color=(0.8, 0.5, 0.1),
    ),
    _c(
        "sensor",
        "2D camera",
        "Video camera with telecentric optics.",
        mode="camera",
        standoff=80.0,
        depth_range=4.0,
        line_width=30.0,
        field_height=22.0,
        noise_um=2.0,
        probe_radius=25.0,
        probe_length=70.0,
        color=(0.6, 0.6, 0.2),
    ),
    # -- stylus shafts (tactile sensors only)
    _c("stylus", "stem 20 mm", "Short stem.", shaft_length=20.0, shaft_radius=0.6),
    _c("stylus", "stem 30 mm", "Standard stem.", shaft_length=30.0, shaft_radius=0.75),
    _c("stylus", "stem 50 mm", "Long stem.", shaft_length=50.0, shaft_radius=1.0),
    _c("stylus", "stem 100 mm", "Very long stem.", shaft_length=100.0, shaft_radius=1.5),
    _c(
        "stylus",
        "5-way star",
        "Star with four side arms; every tip is a tool of its own.",
        shaft_length=40.0,
        shaft_radius=0.9,
        star=True,
        arm_length=22.0,
    ),
    # -- tips (ruby balls)
    _c("tip", "ball 0.5 mm", "Micro ball for small bores.", ball_radius=0.25),
    _c("tip", "ball 1 mm", "Small ball.", ball_radius=0.5),
    _c("tip", "ball 2 mm", "Common ball.", ball_radius=1.0),
    _c("tip", "ball 3 mm", "Standard ball.", ball_radius=1.5),
    _c("tip", "ball 4 mm", "Large ball.", ball_radius=2.0),
    _c("tip", "ball 6 mm", "Very large ball.", ball_radius=3.0),
    _c("tip", "ball 8 mm", "Largest ball.", ball_radius=4.0),
)

#: All components by category.
LIBRARY: dict[str, tuple[Component, ...]] = {
    category: tuple(c for c in _ALL if c.category == category) for category in CATEGORIES
}

#: Rotary tables: ``MachineSpec`` fields for the table top.
ROTARY_TABLES: dict[str, dict[str, float]] = {
    "rotary table 200 mm": {"rotary_diameter": 200.0, "rotary_height": 15.0, "rotary_speed": 120.0},
    "rotary table 300 mm": {"rotary_diameter": 300.0, "rotary_height": 20.0, "rotary_speed": 90.0},
    "rotary table 500 mm": {"rotary_diameter": 500.0, "rotary_height": 30.0, "rotary_speed": 60.0},
}

_OPTICAL = ("laser", "point_laser", "area", "camera")


def components(category: str) -> tuple[Component, ...]:
    """Return the components of a category (see :data:`CATEGORIES`)."""
    return LIBRARY[category]


def machine_names() -> list[str]:
    """Return the names of the machine presets, in the order of their size."""
    return sorted(PRESETS, key=lambda key: PRESETS[key].travel[0] * PRESETS[key].travel[1])


def machine(name: str) -> MachineSpec:
    """Return the machine preset ``name``."""
    return PRESETS[name]


def _find(category: str, name: str) -> Component:
    for c in LIBRARY[category]:
        if c.name == name:
            return c
    raise KeyError(f"no {category} named {name!r}")


def build_tool(
    name: str,
    *,
    sensor: str = "touch trigger (kinematic)",
    head: str = "fixed mount",
    extension: str = "no extension",
    stylus: str | None = None,
    tip: str | None = None,
    star_tip: str = "down",
    **overrides: Any,
) -> ToolSpec:
    """Put the named components together into a tool.

    Optical sensors have no stylus or tip. For a tactile sensor ``stylus`` and ``tip`` default to
    the standard stem and a 3 mm ball. ``overrides`` set any ``ToolSpec`` field last.
    """
    chosen = [_find("head", head), _find("extension", extension), _find("sensor", sensor)]
    base = dict(chosen[2].fields)
    optical = base.get("mode") in _OPTICAL
    if not optical:
        chosen.append(_find("stylus", stylus or "stem 30 mm"))
        chosen.append(_find("tip", tip or "ball 3 mm"))
    elif stylus or tip:
        raise ValueError("an optical sensor has no stylus or tip")
    values: dict[str, Any] = {}
    for component in chosen:
        values.update(component.fields)
    if star_tip != "down":
        values["tip"] = star_tip
    values.update(overrides)
    values.setdefault("description", ", ".join(c.name for c in chosen))
    return ToolSpec(name, **values)
