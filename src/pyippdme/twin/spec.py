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
from dataclasses import dataclass, field, fields, replace
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
        """One standard deviation of the random error of a probed point.

        Sized so that 25 points on a sphere spread by well under MPE_P (the peak-to-valley of
        ``n`` normal samples is about ``3.9 sigma``) and the length error stays inside
        MPE_E(L) once the systematic parts of the probe are added.
        """
        e = self.length_error_mm(length_mm)
        p = self.probing_um * 1e-3
        return float(((0.12 * p) ** 2 + (e / 12.0) ** 2) ** 0.5)


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


#: Measurement modes of a tool and what they allow (``basicfunction`` of Annex G).
MODES = ("touch", "head_touch", "scanning", "laser", "point_laser", "area", "camera", "none")
#: Heads that carry the probe: fixed mount, indexing (PH10 style) or continuous 5-axis (PH20/REVO style).
HEADS = ("fixed", "indexed", "continuous")
TIPS = ("down", "+x", "-x", "+y", "-y")


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """A probe system defined by numbers, not by CAD: head, probe, stylus and how it measures.

    The tool's ``name`` is the one the protocol uses (``Tool.Name()``,
    ``ChangeTool``). Everything else can be edited in the simulator window or in
    ``machine.toml`` (``[[tool]]``). The shapes drawn and checked for collisions
    are generated from these numbers.
    """

    name: str
    #: ``touch`` (kinematic or strain-gauge touch trigger), ``head_touch`` (the head flicks the
    #: stylus out for fast points, PH20/REVO style), ``scanning``, an optical sensor (``laser``
    #: line scanner, ``point_laser``, ``area`` scanner, ``camera``) or ``none``.
    mode: str = "touch"
    head: str = "fixed"
    #: Step of an indexing head in degrees (PH10: 7.5).
    index_step: float = 7.5
    # -- probe body and extension (mm)
    probe_radius: float = 10.0
    probe_length: float = 28.0
    extension: float = 0.0
    # -- stylus (mm); a star adds four arms of ``arm_length`` with a ball on each
    ball_radius: float = 1.5
    shaft_radius: float = 0.75
    shaft_length: float = 30.0
    star: bool = False
    arm_length: float = 22.0
    #: Which ball is the tool centre point: the bottom one or an arm of a star.
    tip: str = "down"
    # -- behaviour
    #: Amplitude of the three-lobe pre-travel variation of a kinematic probe (micrometres).
    lobing_um: float = 0.0
    #: Extra random error of the probe itself (one standard deviation, micrometres).
    repeatability_um: float = 0.0
    #: Systematic error of a tool that is not qualified (one standard deviation, micrometres).
    unqualified_um: float = 12.0
    #: The magnetic stylus module breaks away in a crash instead of bending.
    breakaway: bool = False
    #: Speed of the final approach in mm/s; ``None`` uses the ``PtMeasPar`` speed of the protocol.
    touch_speed: float | None = None
    #: Scanning points per second (``scanning`` mode).
    scan_rate: float = 0.0
    # -- optical sensors: ``laser`` (line), ``point_laser``, ``area`` (structured light), ``camera``
    standoff: float = 60.0
    depth_range: float = 30.0
    #: Width of the line, or of the field of an area sensor or camera (mm).
    line_width: float = 30.0
    field_height: float = 22.0
    points_per_line: int = 320
    line_pitch: float = 1.0
    #: Noise at the stand-off looking straight at the surface (micrometres, one sigma).
    noise_um: float = 3.0
    dropout: float = 0.005
    max_angle_deg: float = 75.0
    triangulation_deg: float = 30.0
    #: What the sensor delivers: ``RSL`` raw scan lines, ``GSL`` gridded, ``QSP`` averaged on the grid.
    point_type: str = "RSL"
    grid_pitch: float = 1.0
    #: Where the tool waits in the rack, shared by the tips of one star; ``None`` is the name.
    rack: str | None = None
    color: tuple[float, float, float] = (0.8, 0.1, 0.1)
    description: str = ""

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"tool {self.name!r}: mode must be one of {', '.join(MODES)}")
        if self.head not in HEADS:
            raise ValueError(f"tool {self.name!r}: head must be one of {', '.join(HEADS)}")
        if self.tip not in TIPS:
            raise ValueError(f"tool {self.name!r}: tip must be one of {', '.join(TIPS)}")
        if self.tip != "down" and not self.star:
            raise ValueError(f"tool {self.name!r}: only a star stylus has side tips")

    @property
    def rack_key(self) -> str:
        return self.rack or self.name


def _star(name: str, tip: str) -> ToolSpec:
    return ToolSpec(
        name,
        "touch",
        "fixed",
        ball_radius=1.0,
        shaft_radius=0.9,
        shaft_length=40.0,
        star=True,
        arm_length=22.0,
        tip=tip,
        lobing_um=0.4,
        breakaway=True,
        rack="Star",
        color=(0.85, 0.25, 0.2),
        description=f"5-way star stylus, tip {tip}",
    )


#: Probe systems for the tools of :data:`pyippdme.simulation.classes.tool_class.TOOL_CATALOG`
#: and the ones the twin adds; the first four are the built-in tools of the simulation.
DEFAULT_TOOLS: dict[str, ToolSpec] = {
    t.name: t
    for t in (
        ToolSpec(
            "RefTool",
            ball_radius=1.5,
            shaft_radius=0.75,
            shaft_length=30.0,
            lobing_um=0.4,
            breakaway=True,
            description="TP20-style kinematic probe on a fixed mount, 3 mm ball, 30 mm stem",
        ),
        ToolSpec(
            "RefTool2",
            ball_radius=3.0,
            shaft_radius=1.5,
            shaft_length=50.0,
            lobing_um=0.5,
            breakaway=True,
            color=(0.9, 0.5, 0.1),
            description="TP20-style kinematic probe, 6 mm ball, 50 mm stem",
        ),
        ToolSpec(
            "AlignProbe",
            head="continuous",
            extension=50.0,
            ball_radius=2.0,
            shaft_radius=1.0,
            shaft_length=40.0,
            lobing_um=0.4,
            breakaway=True,
            color=(0.2, 0.5, 0.9),
            description="TP20-style probe on a continuous 2-axis head, 4 mm ball, 50 mm extension",
        ),
        ToolSpec(
            "NoTool",
            mode="none",
            ball_radius=0.0,
            shaft_radius=0.0,
            shaft_length=0.0,
            probe_radius=9.0,
            probe_length=15.0,
            color=(0.5, 0.5, 0.5),
            description="no probe mounted",
        ),
        _star("StarDown", "down"),
        _star("StarXP", "+x"),
        _star("StarXN", "-x"),
        _star("StarYP", "+y"),
        _star("StarYN", "-y"),
        ToolSpec(
            "IndexedTP200",
            head="indexed",
            extension=100.0,
            probe_radius=11.0,
            ball_radius=2.0,
            shaft_radius=1.0,
            shaft_length=40.0,
            repeatability_um=0.1,
            breakaway=True,
            color=(0.3, 0.7, 0.4),
            description="strain-gauge probe on a PH10-style head (7.5 degree steps), 100 mm extension",
        ),
        ToolSpec(
            "ScanSP25",
            mode="scanning",
            head="indexed",
            extension=50.0,
            probe_radius=14.0,
            probe_length=50.0,
            ball_radius=1.5,
            shaft_radius=0.8,
            shaft_length=40.0,
            scan_rate=2000.0,
            repeatability_um=0.1,
            unqualified_um=15.0,
            color=(0.15, 0.55, 0.85),
            description="SP25-style scanning probe on an indexing head, 3 mm ball",
        ),
        ToolSpec(
            "RevoScan",
            mode="scanning",
            head="continuous",
            extension=0.0,
            probe_radius=14.0,
            probe_length=60.0,
            ball_radius=1.5,
            shaft_radius=0.8,
            shaft_length=50.0,
            scan_rate=6000.0,
            repeatability_um=0.1,
            unqualified_um=8.0,
            color=(0.1, 0.45, 0.8),
            description="5-axis scanning head, qualified once for all angles",
        ),
        ToolSpec(
            "RevoHeadTouch",
            mode="head_touch",
            head="continuous",
            probe_radius=14.0,
            probe_length=60.0,
            ball_radius=1.0,
            shaft_radius=0.7,
            shaft_length=40.0,
            touch_speed=60.0,
            repeatability_um=0.2,
            unqualified_um=8.0,
            color=(0.1, 0.45, 0.8),
            description="5-axis head touches: the head flicks the stylus out for fast points",
        ),
        ToolSpec(
            "LaserLine",
            mode="laser",
            head="indexed",
            probe_radius=0.0,
            probe_length=95.0,
            ball_radius=0.0,
            shaft_radius=0.0,
            shaft_length=0.0,
            standoff=60.0,
            depth_range=30.0,
            line_width=30.0,
            points_per_line=320,
            color=(0.15, 0.15, 0.2),
            description="laser line scanner on an indexing head",
        ),
        ToolSpec(
            "LaserPoint",
            mode="point_laser",
            head="indexed",
            probe_length=70.0,
            ball_radius=0.0,
            shaft_radius=0.0,
            shaft_length=0.0,
            standoff=30.0,
            depth_range=10.0,
            noise_um=1.0,
            triangulation_deg=0.0,
            color=(0.2, 0.2, 0.3),
            description="confocal/laser point sensor: one distance per control point",
        ),
        ToolSpec(
            "AreaScanner",
            mode="area",
            head="indexed",
            probe_length=110.0,
            ball_radius=0.0,
            shaft_radius=0.0,
            shaft_length=0.0,
            standoff=150.0,
            depth_range=60.0,
            line_width=60.0,
            field_height=45.0,
            points_per_line=240,
            noise_um=8.0,
            color=(0.25, 0.2, 0.2),
            description="structured-light scanner: a grid of points per shot",
        ),
        ToolSpec(
            "Camera2D",
            mode="camera",
            head="indexed",
            probe_length=90.0,
            ball_radius=0.0,
            shaft_radius=0.0,
            shaft_length=0.0,
            standoff=80.0,
            depth_range=100.0,
            line_width=30.0,
            field_height=22.5,
            points_per_line=400,
            color=(0.2, 0.25, 0.2),
            description="telecentric camera: the edges it sees, as a video measuring system",
        ),
    )
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
    tools = {str(t["name"]): _parse_tool(t) for t in data.get("tool", [])}  # type: ignore[attr-defined]
    return MachineManifest(spec, components, tools, _vec(data.get("origin")))


_TOOL_FIELDS = {f.name for f in fields(ToolSpec)}


def _parse_tool(t: dict[str, object]) -> ToolSpec:
    """Read one ``[[tool]]``; anything not given keeps the default of ``ToolSpec``."""
    unknown = set(t) - _TOOL_FIELDS
    if unknown:
        raise ValueError(f"tool {t.get('name')!r}: unknown keys {', '.join(sorted(unknown))}")
    values: dict[str, object] = dict(t)
    if "color" in values:
        values["color"] = _vec(values["color"])
    return ToolSpec(**values)  # type: ignore[arg-type]


def tool_to_toml(tool: ToolSpec) -> list[str]:
    """Write the fields of ``tool`` that differ from the defaults as a ``[[tool]]`` table."""
    default = ToolSpec(tool.name)
    lines = ["[[tool]]", f'name = "{tool.name}"']
    for f in fields(ToolSpec):
        value = getattr(tool, f.name)
        if f.name == "name" or value == getattr(default, f.name):
            continue
        if isinstance(value, bool):
            text = str(value).lower()
        elif isinstance(value, str):
            text = f'"{value}"'
        elif isinstance(value, tuple):
            text = "[" + ", ".join(str(v) for v in value) + "]"
        else:
            text = str(value)
        lines.append(f"{f.name} = {text}")
    return lines


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
        lines += ["", *tool_to_toml(t)]
    return "\n".join(lines) + "\n"
