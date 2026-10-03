# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Things on the machine table: the sample (workpiece) and fixtures, plus the stylus shape."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pyippdme.twin import cad, geometry
from pyippdme.twin.geometry import Matrix, Mesh
from pyippdme.twin.spec import ToolSpec
from pyippdme.types.vec3 import Vec3

_ids = itertools.count(1)


@dataclass(slots=True)
class SceneObject:
    """A rigid body in the measuring volume: a sample or a fixture.

    ``pose`` places the CAD shape in machine coordinates (at rotary angle 0);
    an object ``on_rotary`` turns with the rotary table.
    """

    name: str
    kind: str  # "sample" or "fixture"
    shape: cad.Shape
    mesh: Mesh
    pose: Matrix = field(default_factory=geometry.identity)
    on_rotary: bool = False
    color: tuple[float, float, float] = (0.35, 0.65, 0.85)
    visible: bool = True
    source: str = ""
    id: int = field(default_factory=lambda: next(_ids))
    _caster: cad.RayCaster | None = field(default=None, repr=False)

    @classmethod
    def from_shape(
        cls, name: str, kind: str, shape: cad.Shape, *, source: str = "", **kw: object
    ) -> SceneObject:
        return cls(name, kind, shape, cad.tessellate(shape), source=source, **kw)  # type: ignore[arg-type]

    @classmethod
    def from_file(
        cls, path: str | Path, kind: str = "sample", name: str | None = None
    ) -> SceneObject:
        """Load a STEP/IGES/STL/BREP file; all its bodies become one object."""
        bodies = cad.load_cad(path)
        shape = (
            bodies[0].shape if len(bodies) == 1 else cad.make_compound([b.shape for b in bodies])
        )
        color = (0.35, 0.65, 0.85) if kind == "sample" else (0.85, 0.7, 0.3)
        return cls.from_shape(name or Path(path).stem, kind, shape, source=str(path), color=color)

    @property
    def caster(self) -> cad.RayCaster:
        if self._caster is None:
            self._caster = cad.RayCaster(self.shape)
        return self._caster

    def world_pose(self, rotary: Matrix) -> Matrix:
        return rotary @ self.pose if self.on_rotary else self.pose

    def local_bounds(self) -> tuple[Vec3, Vec3]:
        return cad.bounding_box(self.shape)

    def world_bounds(self, rotary: Matrix) -> tuple[np.ndarray, np.ndarray]:
        lo, hi = self.local_bounds()
        corners = np.array(list(itertools.product(*zip(lo, hi, strict=True))))
        world = geometry.apply(self.world_pose(rotary), corners)
        return world.min(axis=0), world.max(axis=0)

    def drop_to(self, z: float, x: float | None = None, y: float | None = None) -> None:
        """Rest the object's lowest point on height ``z``, optionally centring it on ``x``/``y``."""
        lo, hi = self.local_bounds()
        corners = np.array(list(itertools.product(*zip(lo, hi, strict=True))))
        rotated = geometry.apply(self.pose, corners)
        lift = z - rotated[:, 2].min()
        shift_x = 0.0 if x is None else x - (rotated[:, 0].min() + rotated[:, 0].max()) / 2
        shift_y = 0.0 if y is None else y - (rotated[:, 1].min() + rotated[:, 1].max()) / 2
        self.pose = geometry.translation(shift_x, shift_y, lift) @ self.pose


def primitive_fixture(kind: str, size: Vec3, name: str | None = None) -> SceneObject:
    """Make a box (``size`` = x, y, z) or cylinder (``size`` = radius, height, ignored) fixture."""
    if kind == "box":
        shape = cad.make_box(*size)
    elif kind == "cylinder":
        shape = cad.make_cylinder(size[0], size[1])
    else:
        raise ValueError(f"unknown fixture kind {kind!r}; use 'box' or 'cylinder'")
    return SceneObject.from_shape(
        name or kind, "fixture", shape, color=(0.85, 0.7, 0.3), source=f"{kind}{size}"
    )


def demo_sample() -> SceneObject:
    """Make a test block, 80 x 60 x 30 mm with a 20 mm bore and a 10 mm boss; handy without CAD."""
    block = cad.make_box(80.0, 60.0, 30.0)
    bore = cad.make_cylinder(10.0, 40.0, (30.0, 30.0, -5.0))
    boss = cad.make_cylinder(5.0, 12.0, (65.0, 15.0, 30.0))
    shape = cad.fuse(cad.cut(block, bore), boss)
    return SceneObject.from_shape("Demo block", "sample", shape, source="builtin")


class ToolShapes:
    """The stylus of a tool as B-rep, built once along +Z from the TCP and placed per query."""

    def __init__(self) -> None:
        self._cache: dict[ToolSpec, tuple[cad.Shape, cad.Shape]] = {}

    def _build(self, spec: ToolSpec) -> tuple[cad.Shape, cad.Shape]:
        stem_and_holder = cad.make_compound(
            [
                *(
                    [cad.make_cylinder(spec.stem_radius, spec.stem_length)]
                    if spec.stem_length > 0 and spec.stem_radius > 0
                    else []
                ),
                cad.make_cylinder(
                    spec.holder_radius, spec.holder_length, (0.0, 0.0, spec.stem_length)
                ),
            ]
        )
        if spec.ball_radius > 0:
            full = cad.make_compound([stem_and_holder, cad.make_sphere(spec.ball_radius)])
        else:
            full = stem_and_holder
        return full, stem_and_holder

    def shape(self, spec: ToolSpec, tcp: Vec3, axis: Vec3, *, with_ball: bool = True) -> cad.Shape:
        full, no_ball = self._cache.setdefault(spec, self._build(spec))
        return cad.moved(full if with_ball else no_ball, tool_pose(tcp, axis))

    def mesh(self, spec: ToolSpec) -> Mesh:
        return cad.tessellate(self._cache.setdefault(spec, self._build(spec))[0])


def tool_pose(tcp: Vec3, axis: Vec3) -> Matrix:
    """Pose that puts a +Z stylus at ``tcp``, its stem running along ``axis`` (towards the head)."""
    a = np.asarray(axis, dtype=float)
    n = float(np.linalg.norm(a))
    a = np.array([0.0, 0.0, 1.0]) if n == 0 else a / n
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, a)
    s = float(np.linalg.norm(v))
    m = np.eye(4)
    if s > 1e-12:
        m = geometry.rotation(tuple(v / s), float(np.degrees(np.arctan2(s, float(z @ a)))))
    elif a[2] < 0:
        m = geometry.rotation((1.0, 0.0, 0.0), 180.0)
    m[:3, 3] = tcp
    return m
