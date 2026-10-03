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

from dataclasses import dataclass, replace
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


_DEFAULT_COLORS = {
    "table": (0.25, 0.25, 0.28),
    "bridge": (0.55, 0.58, 0.62),
    "carriage": (0.65, 0.67, 0.7),
    "quill": (0.75, 0.77, 0.8),
    "rotary": (0.35, 0.4, 0.45),
}


def _default_shape(name: str, spec: MachineSpec) -> cad.Shape:
    tx, ty, tz = spec.travel
    top = spec.table_top_z
    if name == "table":
        return cad.make_box(tx + 300.0, ty + 400.0, 200.0, (-150.0, -200.0, top - 200.0))
    if name == "bridge":
        legs = cad.fuse(
            cad.make_box(70.0, 90.0, tz + 260.0, (-120.0, -45.0, top)),
            cad.make_box(70.0, 90.0, tz + 260.0, (tx + 50.0, -45.0, top)),
        )
        beam = cad.make_box(tx + 240.0, 90.0, 110.0, (-120.0, -45.0, tz + 160.0))
        return cad.fuse(legs, beam)
    if name == "carriage":
        return cad.make_box(140.0, 120.0, 130.0, (-70.0, -60.0, tz + 130.0))
    if name == "quill":
        # Hangs from the carriage; its lower end is the probe mount, 75 mm above the TCP.
        return cad.make_box(50.0, 50.0, tz + 145.0, (-25.0, -25.0, 75.0))
    if name == "rotary":
        o = spec.rotary_origin or (tx / 2, ty / 2, top)
        return cad.make_cylinder(150.0, 20.0 - top, (o[0], o[1], top))
    raise KeyError(name)


def _default_components(spec: MachineSpec) -> tuple[ComponentSpec, ...]:
    comps = [
        ComponentSpec("table"),
        ComponentSpec("bridge", moves_with=("y",)),
        ComponentSpec("carriage", moves_with=("x", "y")),
        ComponentSpec("quill", moves_with=("x", "y", "z"), collides=True),
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
        origin = manifest.origin if manifest and manifest.origin else None
        shift = geometry.translation(*(-c for c in origin)) if origin else None
        for comp in components:
            if comp.step is not None:
                base = directory or Path()
                parts = cad.load_cad(base / comp.step)
                for part in parts:
                    shape = part.shape if shift is None else cad.moved(part.shape, shift)
                    color = (
                        comp.color or part.color or _DEFAULT_COLORS.get(comp.name, (0.6, 0.6, 0.6))
                    )
                    bodies.append(MachineBody(comp, shape, cad.tessellate(shape), color))
            else:
                shape = _default_shape(comp.name, spec)
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

    def body_pose(self, body: MachineBody, position: Vec3, rotary_deg: float) -> Matrix:
        offset = [0.0, 0.0, 0.0]
        for axis in body.spec.moves_with:
            i = "xyz".index(axis)
            offset[i] = position[i]
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
