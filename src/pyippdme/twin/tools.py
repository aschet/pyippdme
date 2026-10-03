# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Probe systems of the twin: shapes generated from a :class:`~pyippdme.twin.spec.ToolSpec`.

Frame of a tool: origin at the *pivot* (the point the head turns about, under the
quill), ``-Z`` pointing from the pivot towards the tip. A tool is built from the
head housing (fixed to the quill), the joint (an adapter on a fixed mount, an
A/B knuckle on an articulating head), an optional extension bar, the probe
module and the stylus. The probe, extension and stylus turn with the head;
the housing does not.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from pyippdme.simulation.classes.tool_class import register_tool, unregister_tool
from pyippdme.twin import cad, geometry
from pyippdme.twin.geometry import Matrix, Mesh
from pyippdme.twin.spec import ToolSpec
from pyippdme.twin.toolmath import (
    ARM_DROP,
    HOUSING_HEIGHT,
    OPTICAL_MODES,
    drop,
    join_length,
    tip_offset,
    tool_id,
)
from pyippdme.types.vec3 import Vec3


@dataclass(slots=True)
class ToolPart:
    """One drawn body of a tool, in the pivot frame."""

    name: str
    shape: cad.Shape
    mesh: Mesh
    color: tuple[float, float, float]
    #: Turns with the head (everything but the housing).
    rotates: bool
    #: A stylus ball: the part that touches the work.
    tip: bool = False


@dataclass(slots=True)
class ToolModel:
    """The generated shapes of a :class:`ToolSpec`, built once and placed per query."""

    spec: ToolSpec
    parts: list[ToolPart]
    drop: float
    tip_offset: Vec3
    fixed_shape: cad.Shape
    turning_shape: cad.Shape
    tip_shape: cad.Shape
    fixed_corners: np.ndarray = field(default=None, repr=False)  # type: ignore[arg-type]
    turning_corners: np.ndarray = field(default=None, repr=False)  # type: ignore[arg-type]
    #: Largest distance from the pivot to any point of the turning parts.
    reach: float = 0.0

    def collision_boxes(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Boxes covering the turning parts, as ``(low, high)`` in the pivot frame (Figure 51).

        One box per part, then boxes that overlap are merged as long as the merged box wastes
        little space, so the probe body stays one box, the horizontal arms of a star another,
        and the end balls their own, as in the standard's drawing.
        """
        boxes = [
            (np.asarray(lo), np.asarray(hi))
            for lo, hi in (cad.bounding_box(p.shape) for p in self.parts if p.rotates)
        ]
        merged = True
        while merged and len(boxes) > 1:
            merged = False
            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    if _worth_merging(boxes[i], boxes[j]):
                        lo = np.minimum(boxes[i][0], boxes[j][0])
                        hi = np.maximum(boxes[i][1], boxes[j][1])
                        boxes = [b for k, b in enumerate(boxes) if k not in (i, j)] + [(lo, hi)]
                        merged = True
                        break
                if merged:
                    break
        return boxes

    def aabb(self, pivot: Vec3, rotation: Matrix) -> tuple[np.ndarray, np.ndarray]:
        """Axis-aligned box around the placed tool, from the corners of its parts."""
        fixed = self.fixed_corners + np.asarray(pivot)
        turning = geometry.apply(rotation, self.turning_corners) + np.asarray(pivot)
        both = np.vstack([fixed, turning])
        return both.min(axis=0), both.max(axis=0)

    def placed(
        self, pivot: Vec3, rotation: Matrix, *, with_tip: bool = True
    ) -> tuple[cad.Shape, cad.Shape]:
        """``(fixed, turning)`` shapes in machine coordinates; ``turning`` includes the tip ball."""
        fixed = cad.moved(self.fixed_shape, geometry.translation(*pivot))
        pose = geometry.translation(*pivot) @ rotation
        parts = [self.turning_shape, self.tip_shape] if with_tip else [self.turning_shape]
        turning = cad.make_compound([cad.moved(p, pose) for p in parts])
        return fixed, turning


def _volume(box: tuple[np.ndarray, np.ndarray]) -> float:
    return float(np.prod(np.maximum(box[1] - box[0], 1e-6)))


def _worth_merging(a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]) -> bool:
    """Merge two boxes that touch when the union holds at most 1.6 times their own volumes."""
    if np.any(a[0] > b[1] + 1e-6) or np.any(b[0] > a[1] + 1e-6):
        return False
    union = (np.minimum(a[0], b[0]), np.maximum(a[1], b[1]))
    overlap_lo, overlap_hi = np.maximum(a[0], b[0]), np.minimum(a[1], b[1])
    shared = float(np.prod(np.maximum(overlap_hi - overlap_lo, 0.0)))
    return _volume(union) <= 1.6 * (_volume(a) + _volume(b) - shared)


def _tessellated(shape: cad.Shape) -> Mesh:
    return cad.tessellate(shape, deflection=0.05)


def build_tool(spec: ToolSpec) -> ToolModel:
    """Generate the shapes of ``spec`` (pivot at the origin, tip towards ``-Z``)."""
    dark = (0.12, 0.12, 0.14)
    joint = join_length(spec)
    parts: list[ToolPart] = []

    def add(
        name: str,
        shape: cad.Shape,
        color: tuple[float, float, float],
        rotates: bool,
        tip: bool = False,
    ) -> None:
        parts.append(ToolPart(name, shape, _tessellated(shape), color, rotates, tip))

    housing_r = 20.0 if spec.head == "fixed" else 28.0
    add("housing", cad.make_cylinder(housing_r, HOUSING_HEIGHT), dark, False)
    if spec.head == "fixed":
        add("adapter", cad.make_cylinder(16.0, joint, (0, 0, -joint)), (0.35, 0.36, 0.4), True)
    else:
        add("knuckle", cad.make_sphere(24.0, (0, 0, -6.0)), (0.2, 0.2, 0.24), True)
        add(
            "axis",
            cad.make_cylinder(13.0, 58.0, (-29.0, 0, -6.0), (1, 0, 0)),
            (0.3, 0.3, 0.34),
            True,
        )
    z = -joint
    if spec.extension > 0:
        add(
            "extension",
            cad.make_cylinder(8.0, spec.extension, (0, 0, z - spec.extension)),
            (0.55, 0.56, 0.6),
            True,
        )
        z -= spec.extension
    if spec.mode in OPTICAL_MODES:
        add(
            "sensor",
            cad.rounded_box(
                70.0, 40.0, spec.probe_length, (-35.0, -20.0, z - spec.probe_length), 4
            ),
            spec.color,
            True,
        )
        add(
            "window",
            cad.make_box(36.0, 2.0, 10.0, (-18.0, -22.0, z - spec.probe_length)),
            (0.8, 0.1, 0.1),
            True,
        )
    else:
        add(
            "probe",
            cad.make_cylinder(spec.probe_radius, spec.probe_length, (0, 0, z - spec.probe_length)),
            (0.1, 0.1, 0.12),
            True,
        )
        add(
            "module ring",
            cad.make_cylinder(spec.probe_radius + 1.0, 3.0, (0, 0, z - spec.probe_length)),
            spec.color,
            True,
        )
        z -= spec.probe_length
        if spec.mode != "none" and spec.shaft_length > 0:
            bottom = z - spec.shaft_length
            add(
                "stylus",
                cad.make_cylinder(spec.shaft_radius, spec.shaft_length, (0, 0, bottom)),
                (0.75, 0.75, 0.78),
                True,
            )
            add(
                "ball",
                cad.make_sphere(spec.ball_radius, (0, 0, bottom)),
                (0.85, 0.1, 0.15),
                True,
                tip=True,
            )
            if spec.star:
                arm_z = bottom + ARM_DROP
                add(
                    "hub",
                    cad.make_cylinder(spec.shaft_radius * 1.8, 4.0, (0, 0, arm_z - 2.0)),
                    (0.7, 0.7, 0.74),
                    True,
                )
                for i, (dx, dy) in enumerate(((1, 0), (-1, 0), (0, 1), (0, -1))):
                    add(
                        f"arm{i}",
                        cad.make_cylinder(
                            spec.shaft_radius * 0.8, spec.arm_length, (0, 0, arm_z), (dx, dy, 0)
                        ),
                        (0.75, 0.75, 0.78),
                        True,
                    )
                    add(
                        f"arm{i} ball",
                        cad.make_sphere(
                            spec.ball_radius, (dx * spec.arm_length, dy * spec.arm_length, arm_z)
                        ),
                        (0.85, 0.1, 0.15),
                        True,
                        tip=True,
                    )

    def compound(select: list[ToolPart]) -> cad.Shape:
        return cad.make_compound([p.shape for p in select])

    def corners(select: list[ToolPart]) -> np.ndarray:
        boxes = [cad.bounding_box(p.shape) for p in select] or [((0, 0, 0), (0, 0, 0))]
        return np.array(
            [c for lo, hi in boxes for c in itertools.product(*zip(lo, hi, strict=True))]
        )

    fixed_parts = [p for p in parts if not p.rotates]
    turning_parts = [p for p in parts if p.rotates]
    return ToolModel(
        spec,
        parts,
        drop(spec),
        tip_offset(spec),
        compound(fixed_parts),
        compound([p for p in turning_parts if not p.tip]),
        compound([p for p in turning_parts if p.tip]),
        corners(fixed_parts),
        corners(turning_parts),
        float(
            max(
                (
                    float(np.linalg.norm(p.mesh.vertices, axis=1).max())
                    for p in turning_parts
                    if len(p.mesh.vertices)
                ),
                default=0.0,
            )
        ),
    )


class ToolKit:
    """The tools of a machine: specs, generated models and the registration with the server."""

    def __init__(self, specs: dict[str, ToolSpec]) -> None:
        self.specs = dict(specs)
        self._models: dict[ToolSpec, ToolModel] = {}
        self._registered: set[str] = set()

    def spec(self, name: str) -> ToolSpec:
        return self.specs.get(name) or ToolSpec(name, mode="none")

    def model(self, name: str) -> ToolModel:
        spec = self.spec(name)
        if spec not in self._models:
            self._models[spec] = build_tool(spec)
        return self._models[spec]

    def reference_drop(self) -> float:
        """Pivot height above the tool centre point of the reference tool (quill is built for it)."""
        return drop(self.spec("RefTool"))

    def register(self) -> None:
        """Make the tools known to the protocol (``EnumTools``, ``ChangeTool``, ``Tool.Id()``)."""
        self.unregister()
        from pyippdme.simulation.classes.tool_class import TOOL_CATALOG

        reference = tip_offset(self.spec("RefTool"))
        for name, spec in self.specs.items():
            if name in TOOL_CATALOG and name not in self._registered:
                # A built-in tool keeps its protocol description; only its physics is ours.
                if name in ("RefTool", "RefTool2", "AlignProbe", "NoTool"):
                    continue
            offset = tuple(a - b for a, b in zip(tip_offset(spec), reference, strict=True))
            register_tool(
                name,
                tool_id(spec),
                offset,  # type: ignore[arg-type]
                alignable=spec.head != "fixed",
            )
            self._registered.add(name)

    def unregister(self) -> None:
        for name in list(self._registered):
            unregister_tool(name)
        self._registered.clear()
