# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Things on the machine table: the sample (workpiece) and fixtures, plus the stylus shape."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from pyippdme.twin import cad, geometry
from pyippdme.twin.geometry import Matrix, Mesh
from pyippdme.types.vec3 import Vec3

if TYPE_CHECKING:
    from pyippdme.twin.artifact import ArtifactData

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
    #: Nominal features and measurement plans, for check artefacts (``twin.artifact``).
    artifact: ArtifactData | None = None
    id: int = field(default_factory=lambda: next(_ids))
    _caster: cad.RayCaster | None = field(default=None, repr=False)
    _bounds: tuple[Vec3, Vec3] | None = field(default=None, repr=False)
    _fine: Mesh | None = field(default=None, repr=False)

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

    @property
    def fine_mesh(self) -> Mesh:
        """A mesh fine enough for optical sensors (some 10 micrometres of sag), built on demand."""
        if self._fine is None:
            self._fine = cad.tessellate(self.shape, deflection=0.02, angular=0.1)
        return self._fine

    def world_pose(self, rotary: Matrix) -> Matrix:
        return rotary @ self.pose if self.on_rotary else self.pose

    def local_bounds(self) -> tuple[Vec3, Vec3]:
        if self._bounds is None:
            self._bounds = cad.bounding_box(self.shape)
        return self._bounds

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
