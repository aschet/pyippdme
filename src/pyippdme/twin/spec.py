# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Machine and tool specifications of the digital twin, and the ``machine.toml`` that carries them.

A machine is a directory with a ``machine.toml`` next to the STEP files of its
components (see :mod:`pyippdme.twin.machine`). Everything in the spec is
optional; what is missing is derived from the STEP geometry where possible
(travel ranges from the table and bridge, component roles from their names) or
taken from the defaults below. The presets are *representative* of the public
data sheet ranges of bridge-type CMMs (measuring range, MPE_E = A + L/K, axis
speed and acceleration), not a model of one particular machine: replace them
with the values of the machine you simulate.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path

from pyippdme.types.vec3 import Vec3

AXES = ("x", "y", "z")


@dataclass(frozen=True, slots=True)
class Accuracy:
    """Length measuring error in the style of ISO 10360-2: MPE_E = ``a`` + L/``k`` micrometres."""

    a_um: float = 1.9
    k: float = 300.0
    #: Probing error MPE_P (ISO 10360-5) in micrometres.
    probing_um: float = 1.9

    def length_error_mm(self, length_mm: float) -> float:
        return (self.a_um + length_mm / self.k) * 1e-3

    def sigma_mm(self, length_mm: float) -> float:
        """One standard deviation of a probed point, a third of the MPE (95 %+ inside it)."""
        e = self.length_error_mm(length_mm)
        p = self.probing_um * 1e-3
        return float(((e / 3.0) ** 2 + (p / 3.0) ** 2) ** 0.5)


@dataclass(frozen=True, slots=True)
class MachineSpec:
    """Kinematics and accuracy of a bridge-type Cartesian CMM."""

    name: str = "Bridge CMM 700/700/600"
    #: Travel of X, Y and Z in millimetres, measured from the machine zero (the home position).
    travel: tuple[float, float, float] = (700.0, 700.0, 600.0)
    #: Maximum vector speed in mm/s and acceleration in mm/s^2.
    max_speed: float = 520.0
    acceleration: float = 1200.0
    #: Speed of the final probing approach in mm/s (``PtMeas``).
    probing_speed: float = 8.0
    #: Speed while scanning along a surface in mm/s.
    scanning_speed: float = 30.0
    accuracy: Accuracy = field(default_factory=Accuracy)
    #: Whether the machine refuses to move before ``Home()``.
    require_home: bool = True
    #: Rotary table: speed in degrees/s; ``None`` axis origin means no rotary table.
    rotary_speed: float = 90.0
    rotary_origin: Vec3 | None = None
    rotary_axis: Vec3 = (0.0, 0.0, 1.0)
    #: Height of the table surface below the machine zero.
    table_top_z: float = -10.0
    #: Fields that were derived from CAD geometry and not given (informational).
    derived: tuple[str, ...] = ()

    def with_(self, **changes: object) -> MachineSpec:
        return replace(self, **changes)  # type: ignore[arg-type]


#: Representative bridge CMM classes; see the module docstring for what these are.
PRESETS: dict[str, MachineSpec] = {
    "bridge-700": MachineSpec(
        "Bridge CMM 700/700/600",
        (700.0, 700.0, 600.0),
        520.0,
        1200.0,
        accuracy=Accuracy(1.9, 300.0, 1.9),
    ),
    "bridge-900": MachineSpec(
        "Bridge CMM 900/1200/700",
        (900.0, 1200.0, 700.0),
        520.0,
        1200.0,
        accuracy=Accuracy(1.6, 350.0, 1.6),
    ),
    "bridge-1000-large": MachineSpec(
        "Bridge CMM 1000/2100/600",
        (1000.0, 2100.0, 600.0),
        520.0,
        1000.0,
        accuracy=Accuracy(1.9, 300.0, 1.9),
    ),
    "bridge-high-precision": MachineSpec(
        "High-precision bridge CMM 1000/1200/700",
        (1000.0, 1200.0, 700.0),
        400.0,
        800.0,
        accuracy=Accuracy(0.5, 600.0, 0.5),
    ),
}


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """The physical stylus of a tool: a ball on a stem, optionally on an indexable head.

    The tool's name is the one the protocol uses (``Tool.Name()``, ``ChangeTool``).
    """

    name: str
    ball_radius: float = 1.5
    stem_radius: float = 0.75
    #: Distance from the ball centre (the TCP) up to the probe body.
    stem_length: float = 30.0
    #: Probe body (the holder above the stem).
    holder_radius: float = 12.0
    holder_length: float = 40.0
    color: tuple[float, float, float] = (0.8, 0.1, 0.1)
    description: str = ""


#: Physical styli for the tools of :data:`pyippdme.simulation.classes.tool_class.TOOL_CATALOG`.
DEFAULT_TOOLS: dict[str, ToolSpec] = {
    "RefTool": ToolSpec("RefTool", 1.5, 0.75, 30.0, description="3 mm ruby ball, 30 mm stem"),
    "RefTool2": ToolSpec(
        "RefTool2", 3.0, 1.5, 50.0, color=(0.9, 0.5, 0.1), description="6 mm ruby ball, 50 mm stem"
    ),
    "AlignProbe": ToolSpec(
        "AlignProbe",
        2.0,
        1.0,
        40.0,
        holder_radius=18.0,
        holder_length=60.0,
        color=(0.2, 0.5, 0.9),
        description="4 mm ball on an indexing head",
    ),
    "NoTool": ToolSpec(
        "NoTool", 0.0, 0.0, 0.0, holder_radius=8.0, holder_length=20.0, color=(0.5, 0.5, 0.5)
    ),
}


@dataclass(frozen=True, slots=True)
class ComponentSpec:
    """One STEP body of the machine and how it moves."""

    name: str
    #: STEP file relative to the machine directory; ``None`` takes the generated default body.
    step: str | None = None
    #: Which axes carry the component: any of ``x``, ``y``, ``z``.
    moves_with: tuple[str, ...] = ()
    #: Turns with the rotary table.
    rotates: bool = False
    #: Takes part in collision checks of the stylus.
    collides: bool = True
    color: tuple[float, float, float] | None = None


@dataclass(frozen=True, slots=True)
class MachineManifest:
    """The parsed ``machine.toml``."""

    spec: MachineSpec
    components: tuple[ComponentSpec, ...] = ()
    tools: dict[str, ToolSpec] = field(default_factory=dict)
    #: Machine zero in the coordinate system of the STEP files; ``None`` derives it.
    origin: Vec3 | None = None


def _vec(value: object, default: Vec3 | None = None) -> Vec3 | None:
    if value is None:
        return default
    x, y, z = (float(v) for v in value)  # type: ignore[attr-defined]
    return (x, y, z)


def parse_manifest(data: dict[str, object], *, base: MachineSpec | None = None) -> MachineManifest:
    """Build a manifest from the decoded TOML ``data``; unknown keys are an error."""
    spec = base or PRESETS["bridge-700"]
    m: dict[str, object] = dict(data.get("machine", {}))  # type: ignore[call-overload]
    preset = m.pop("preset", None)
    if preset is not None:
        spec = PRESETS[str(preset)]
    travel = _vec(m.pop("travel", None))
    acc_data: dict[str, float] = dict(m.pop("accuracy", {}))  # type: ignore[call-overload]
    accuracy = replace(
        spec.accuracy,
        a_um=float(acc_data.get("a_um", spec.accuracy.a_um)),
        k=float(acc_data.get("k", spec.accuracy.k)),
        probing_um=float(acc_data.get("probing_um", spec.accuracy.probing_um)),
    )
    rotary = m.pop("rotary", None)
    changes: dict[str, object] = {"accuracy": accuracy}
    if travel is not None:
        changes["travel"] = travel
    if rotary is not None:
        r: dict[str, object] = dict(rotary)  # type: ignore[call-overload]
        changes["rotary_origin"] = _vec(r.get("origin"))
        changes["rotary_axis"] = _vec(r.get("axis"), (0.0, 0.0, 1.0))
        if "speed" in r:
            changes["rotary_speed"] = float(r["speed"])  # type: ignore[arg-type]
    for key in ("name",):
        if key in m:
            changes[key] = str(m.pop(key))
    for key in (
        "max_speed",
        "acceleration",
        "probing_speed",
        "scanning_speed",
        "table_top_z",
    ):
        if key in m:
            changes[key] = float(m.pop(key))  # type: ignore[arg-type]
    if "require_home" in m:
        changes["require_home"] = bool(m.pop("require_home"))
    if m:
        raise ValueError(f"unknown keys in [machine]: {', '.join(sorted(m))}")
    spec = replace(spec, **changes)  # type: ignore[arg-type]

    components = tuple(
        ComponentSpec(
            name=str(c["name"]),
            step=c.get("step"),
            moves_with=tuple(str(a).lower() for a in c.get("moves_with", ())),
            rotates=bool(c.get("rotates", False)),
            collides=bool(c.get("collides", True)),
            color=_vec(c.get("color")),
        )
        for c in data.get("component", [])  # type: ignore[attr-defined]
    )
    for c in components:
        bad = set(c.moves_with) - set(AXES)
        if bad:
            raise ValueError(f"component {c.name!r}: unknown axes {sorted(bad)}")
    tools = {
        str(t["name"]): ToolSpec(
            name=str(t["name"]),
            ball_radius=float(t.get("ball_radius", 1.5)),
            stem_radius=float(t.get("stem_radius", 0.75)),
            stem_length=float(t.get("stem_length", 30.0)),
            holder_radius=float(t.get("holder_radius", 12.0)),
            holder_length=float(t.get("holder_length", 40.0)),
            color=_vec(t.get("color"), (0.8, 0.1, 0.1)) or (0.8, 0.1, 0.1),
            description=str(t.get("description", "")),
        )
        for t in data.get("tool", [])  # type: ignore[attr-defined]
    }
    return MachineManifest(spec, components, tools, _vec(data.get("origin")))


def load_manifest(path: str | Path) -> MachineManifest:
    with Path(path).open("rb") as handle:
        return parse_manifest(tomllib.load(handle))


def manifest_to_toml(manifest: MachineManifest) -> str:
    """Write ``manifest`` as ``machine.toml`` text (the format :func:`load_manifest` reads)."""
    s = manifest.spec
    lines = [
        "[machine]",
        f'name = "{s.name}"',
        f"travel = [{s.travel[0]}, {s.travel[1]}, {s.travel[2]}]",
        f"max_speed = {s.max_speed}",
        f"acceleration = {s.acceleration}",
        f"probing_speed = {s.probing_speed}",
        f"scanning_speed = {s.scanning_speed}",
        f"table_top_z = {s.table_top_z}",
        f"require_home = {str(s.require_home).lower()}",
        "",
        "[machine.accuracy]",
        f"a_um = {s.accuracy.a_um}",
        f"k = {s.accuracy.k}",
        f"probing_um = {s.accuracy.probing_um}",
    ]
    if s.rotary_origin is not None:
        o, a = s.rotary_origin, s.rotary_axis
        lines += [
            "",
            "[machine.rotary]",
            f"origin = [{o[0]}, {o[1]}, {o[2]}]",
            f"axis = [{a[0]}, {a[1]}, {a[2]}]",
            f"speed = {s.rotary_speed}",
        ]
    if manifest.origin is not None:
        o = manifest.origin
        lines.insert(0, f"origin = [{o[0]}, {o[1]}, {o[2]}]")
    for c in manifest.components:
        lines += ["", "[[component]]", f'name = "{c.name}"']
        if c.step:
            lines.append(f'step = "{c.step}"')
        axes = ", ".join(f'"{a}"' for a in c.moves_with)
        lines.append(f"moves_with = [{axes}]")
        if c.rotates:
            lines.append("rotates = true")
        if not c.collides:
            lines.append("collides = false")
    for t in manifest.tools.values():
        lines += [
            "",
            "[[tool]]",
            f'name = "{t.name}"',
            f"ball_radius = {t.ball_radius}",
            f"stem_radius = {t.stem_radius}",
            f"stem_length = {t.stem_length}",
        ]
    return "\n".join(lines) + "\n"
