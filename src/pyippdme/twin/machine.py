# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""The machine of the twin: STEP components that move with the axes.

Three ways to get one:

* :meth:`MachineModel.default` generates a bridge CMM (granite table, bridge,
  carriage, quill, optional rotary table) from the travel ranges in the spec.
* :meth:`MachineModel.from_directory` reads a ``machine.toml`` that names one
  STEP file per component and how it moves (``moves_with = ["y"]`` for the
  bridge, ``["x", "y"]`` for the carriage, ``["x", "y", "z"]`` for the quill,
  ``rotates = true`` for a rotary table top). A component without ``step``
  falls back to the generated body of the same name, so a directory can replace
  just the parts you have CAD for.
* :meth:`MachineModel.from_step` reads one STEP assembly and recognises its
  parts by name (table/granite, bridge/portal, carriage/slide, quill/spindle/ram,
  rotary/turntable). Travel ranges and machine zero that the spec does not give
  are *estimated* from the geometry and listed in ``spec.derived``.

All coordinates are machine coordinates: origin at the home position, X/Y/Z as
travelled, Z up.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from pyippdme.twin import cad, geometry
from pyippdme.twin.geometry import Matrix, Mesh
from pyippdme.twin.spec import (
    DEFAULT_TOOLS,
    PRESETS,
    ComponentSpec,
    MachineManifest,
    MachineSpec,
    ToolSpec,
    load_manifest,
    manifest_to_toml,
)
from pyippdme.twin.toolmath import HOUSING_HEIGHT, rack_keys, rack_slots
from pyippdme.types.vec3 import Vec3

#: Name fragments that give a STEP part its role when only an assembly is available.
_ROLES: tuple[tuple[str, tuple[str, ...], bool, tuple[str, ...]], ...] = (
    ("rotary", ("rotary", "turntable", "rotation", "rundtisch"), True, ()),
    ("quill", ("quill", "spindle", "ram", "pinole", "z-axis", "zaxis"), False, ("x", "y", "z")),
    ("carriage", ("carriage", "slide", "schlitten", "x-axis", "xaxis"), False, ("x", "y")),
    ("bridge", ("bridge", "portal", "beam", "leg", "bruecke", "brücke"), False, ("y",)),
    ("table", ("table", "granite", "base", "plate", "tisch"), False, ()),
)


@dataclass(slots=True)
class MachineBody:
    """A component with its geometry; ``rest_mesh`` is at the home position."""

    spec: ComponentSpec
    shape: cad.Shape
    mesh: Mesh
    color: tuple[float, float, float]
    #: Exact bounding box of ``shape``, computed once.
    bounds: tuple[Vec3, Vec3] = field(default=((0.0,) * 3, (0.0,) * 3), repr=False)

    def __post_init__(self) -> None:
        self.bounds = cad.bounding_box(self.shape)


_DEFAULT_COLORS = {
    "table": (0.13, 0.13, 0.15),
    "stand": (0.2, 0.21, 0.24),
    "bridge": (0.72, 0.74, 0.78),
    "carriage": (0.82, 0.84, 0.88),
    "quill": (0.88, 0.89, 0.92),
    "rack": (0.3, 0.33, 0.38),
    "rotary": (0.3, 0.34, 0.4),
}

#: Distance of the quill bottom above the tool centre point of the reference tool at home.
_QUILL_BOTTOM = HOUSING_HEIGHT + 72.0


def _default_shape(name: str, spec: MachineSpec, slots: dict[str, Vec3]) -> cad.Shape:
    tx, ty, tz = spec.travel
    top = spec.table_top_z
    if name == "table":
        return cad.rounded_box(tx + 400.0, ty + 500.0, 240.0, (-200.0, -250.0, top - 240.0), 6.0)
    if name == "stand":
        legs = [
            cad.rounded_box(110.0, 110.0, 380.0, (x, y, top - 620.0), 6.0)
            for x in (-180.0, tx + 70.0)
            for y in (-230.0, ty + 120.0)
        ]
        frame = [
            cad.make_box(tx + 400.0, 60.0, 40.0, (-200.0, y, top - 360.0))
            for y in (-205.0, ty + 145.0)
        ]
        shape = legs[0]
        for part in (*legs[1:], *frame):
            shape = cad.fuse(shape, part)
        return shape
    if name == "bridge":
        beam_z = tz + 160.0
        feet = [
            cad.rounded_box(150.0, 240.0, 50.0, (x, -120.0, top), 8.0) for x in (-205.0, tx + 55.0)
        ]
        legs = [
            cad.rounded_box(110.0, 170.0, beam_z + 200.0 - top, (x, -85.0, top), 12.0)
            for x in (-185.0, tx + 75.0)
        ]
        beam = cad.rounded_box(tx + 400.0, 190.0, 210.0, (-200.0, -95.0, beam_z), 14.0)
        shape = beam
        for part in (*legs, *feet):
            shape = cad.fuse(shape, part)
        return shape
    if name == "carriage":
        beam_z = tz + 160.0
        return cad.rounded_box(230.0, 250.0, 250.0, (-115.0, -125.0, beam_z - 20.0), 14.0)
    if name == "quill":
        return cad.rounded_box(84.0, 84.0, tz + 190.0, (-42.0, -42.0, _QUILL_BOTTOM), 8.0)
    if name == "rack":
        keys = list(slots)
        xs = [p[0] for p in slots.values()]
        y = next(iter(slots.values()))[1] if slots else ty - 45.0
        x0, x1 = (min(xs) - 40.0, max(xs) + 40.0) if xs else (0.0, 100.0)
        plate = cad.rounded_box(x1 - x0, 70.0, 14.0, (x0, y - 35.0, top), 3.0)
        shape = plate
        for key in keys:
            sx, sy, _ = slots[key]
            shape = cad.fuse(shape, cad.make_cylinder(15.0, 6.0, (sx, sy, top + 14.0)))
        return shape
    if name == "rotary":
        o = spec.rotary_origin or (tx / 2, ty / 2, top)
        return cad.make_cylinder(150.0, 20.0 - top, (o[0], o[1], top))
    raise KeyError(name)


def _default_components(spec: MachineSpec) -> tuple[ComponentSpec, ...]:
    comps = [
        ComponentSpec("table"),
        ComponentSpec("stand", collides=False),
        ComponentSpec("bridge", moves_with=("y",)),
        ComponentSpec("carriage", moves_with=("x", "y"), collides=False),
        ComponentSpec("quill", moves_with=("x", "y", "z")),
        ComponentSpec("rack"),
    ]
    if spec.rotary_origin is not None:
        comps.append(ComponentSpec("rotary", rotates=True))
    return tuple(comps)


class MachineModel:
    """Machine spec, tools and the moving STEP components."""

    def __init__(
        self,
        spec: MachineSpec,
        bodies: list[MachineBody],
        tools: dict[str, ToolSpec] | None = None,
        manifest: MachineManifest | None = None,
        directory: Path | None = None,
    ) -> None:
        self.spec = spec
        self.bodies = bodies
        self.tools: dict[str, ToolSpec] = {**DEFAULT_TOOLS, **(tools or {})}
        self.manifest = manifest
        self.directory = directory

    # -- construction ------------------------------------------------------------------

    @classmethod
    def default(
        cls, spec: MachineSpec | str = "bridge-700", *, rotary: bool = False
    ) -> MachineModel:
        s = PRESETS[spec] if isinstance(spec, str) else spec
        if rotary and s.rotary_origin is None:
            s = replace(s, rotary_origin=(s.travel[0] / 2, s.travel[1] / 2, s.table_top_z))
        return cls._from_components(s, _default_components(s), None, None)

    @classmethod
    def from_directory(cls, path: str | Path) -> MachineModel:
        directory = Path(path)
        manifest = load_manifest(directory / "machine.toml")
        spec = manifest.spec
        components = manifest.components or _default_components(spec)
        if any(c.rotates for c in components) and spec.rotary_origin is None:
            spec = replace(
                spec, rotary_origin=(spec.travel[0] / 2, spec.travel[1] / 2, spec.table_top_z)
            )
        return cls._from_components(spec, components, manifest, directory)

    @classmethod
    def _from_components(
        cls,
        spec: MachineSpec,
        components: tuple[ComponentSpec, ...],
        manifest: MachineManifest | None,
        directory: Path | None,
    ) -> MachineModel:
        bodies: list[MachineBody] = []
        tool_specs = {**DEFAULT_TOOLS, **(manifest.tools if manifest else {})}
        slots = rack_slots(spec, rack_keys(tool_specs))
        origin = manifest.origin if manifest and manifest.origin else None
        shift = geometry.translation(*(-c for c in origin)) if origin else None
        for comp in components:
            if comp.step is not None:
                base = directory or Path()
                parts = cad.load_cad(base / comp.step)
                merged = (
                    parts[0].shape
                    if len(parts) == 1
                    else cad.make_compound([part.shape for part in parts])
                )
                shape = merged if shift is None else cad.moved(merged, shift)
                color = (
                    comp.color or parts[0].color or _DEFAULT_COLORS.get(comp.name, (0.6, 0.6, 0.6))
                )
                bodies.append(MachineBody(comp, shape, cad.tessellate(shape), color))
            else:
                shape = _default_shape(comp.name, spec, slots)
                color = comp.color or _DEFAULT_COLORS.get(comp.name, (0.6, 0.6, 0.6))
                bodies.append(MachineBody(comp, shape, cad.tessellate(shape), color))
        return cls(spec, bodies, manifest.tools if manifest else None, manifest, directory)

    @classmethod
    def from_step(
        cls,
        path: str | Path,
        spec: MachineSpec | None = None,
        *,
        origin: Vec3 | None = None,
    ) -> MachineModel:
        """Build the machine from one STEP assembly, recognising parts by name."""
        parts = cad.load_cad(path)
        classified: list[tuple[str, cad.CadBody, tuple[str, ...], bool]] = []
        for part in parts:
            role, rotates = "table", False
            axes: tuple[str, ...] = ()
            lowered = part.name.lower()
            for name, keywords, rot, moves in _ROLES:
                if any(k in lowered for k in keywords):
                    role, axes, rotates = name, moves, rot
                    break
            classified.append((role, part, axes, rotates))

        derived: list[str] = []
        s = spec or PRESETS["bridge-700"]
        tables = [p for r, p, _, _ in classified if r == "table"]
        if tables:
            lo = np.min([cad.bounding_box(t.shape)[0] for t in tables], axis=0)
            hi = np.max([cad.bounding_box(t.shape)[1] for t in tables], axis=0)
            if origin is None:
                # Machine zero: a tenth of the table in from its corner, on the table surface.
                size = hi - lo
                origin = (
                    float(lo[0] + 0.1 * size[0]),
                    float(lo[1] + 0.1 * size[1]),
                    float(hi[2]) - s.table_top_z,
                )
                derived.append("origin")
            if spec is None:
                top_z = max(cad.bounding_box(p.shape)[1][2] for _, p, _, _ in classified)
                size = hi - lo
                tz = max((top_z - origin[2]) * 0.6, 100.0)
                s = replace(
                    s,
                    name=Path(path).stem,
                    travel=(
                        round(float(size[0]) * 0.8, 1),
                        round(float(size[1]) * 0.8, 1),
                        round(float(tz), 1),
                    ),
                )
                derived.append("travel")
        if origin is None:
            origin = (0.0, 0.0, 0.0)
        shift = geometry.translation(*(-c for c in origin))
        bodies: list[MachineBody] = []
        for role, part, axes, rotates in classified:
            shape = cad.moved(part.shape, shift)
            comp = ComponentSpec(part.name or role, None, axes, rotates)
            color = part.color or _DEFAULT_COLORS.get(role, (0.6, 0.6, 0.6))
            bodies.append(MachineBody(comp, shape, cad.tessellate(shape), color))
        rotaries = [b for b in bodies if b.spec.rotates]
        if rotaries and s.rotary_origin is None:
            lo, hi = cad.bounding_box(rotaries[0].shape)
            s = replace(s, rotary_origin=((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, hi[2]))
            derived.append("rotary_origin")
        return cls(replace(s, derived=tuple(derived)), bodies)

    # -- kinematics --------------------------------------------------------------------

    def body_pose(
        self,
        body: MachineBody,
        position: Vec3,
        rotary_deg: float,
        shift: Vec3 = (0.0, 0.0, 0.0),
    ) -> Matrix:
        """Pose of a component; ``shift`` is how far the head pivot is from where the quill
        was built for (a different tool, an articulated head).
        """
        offset = [0.0, 0.0, 0.0]
        for axis in body.spec.moves_with:
            i = "xyz".index(axis)
            offset[i] = position[i] + shift[i]
        m = geometry.translation(*offset)
        if body.spec.rotates and self.spec.rotary_origin is not None:
            m = m @ geometry.rotation_about(
                self.spec.rotary_origin, self.spec.rotary_axis, rotary_deg
            )
        return m

    def rotary_pose(self, rotary_deg: float) -> Matrix:
        """Pose of everything mounted on the rotary table at ``rotary_deg``."""
        if self.spec.rotary_origin is None:
            return geometry.identity()
        return geometry.rotation_about(self.spec.rotary_origin, self.spec.rotary_axis, rotary_deg)

    def rack_slots(self) -> dict[str, Vec3]:
        return rack_slots(self.spec, rack_keys(self.tools))

    def tool(self, name: str) -> ToolSpec:
        return self.tools.get(name) or ToolSpec(name)

    # -- export ------------------------------------------------------------------------

    def export(self, directory: str | Path) -> Path:
        """Write the machine as ``machine.toml`` plus one STEP file per component."""
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        components = []
        for body in self.bodies:
            filename = f"{body.spec.name}.step"
            cad.write_step([cad.CadBody(body.spec.name, body.shape)], out / filename)
            components.append(replace(body.spec, step=filename))
        manifest = MachineManifest(self.spec, tuple(components), {})
        path = out / "machine.toml"
        path.write_text(manifest_to_toml(manifest), encoding="utf-8")
        return path
