# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Tool kinematics and descriptions, without CAD: where a tool's tip is and how its head turns.

Frame of a tool: origin at the *pivot* (the point the head turns about, under the quill),
``-Z`` pointing from the pivot towards the tip. :mod:`pyippdme.twin.tools` builds
the shapes for the same numbers.
"""

from __future__ import annotations

import math

import numpy as np

from pyippdme.twin import geometry
from pyippdme.twin.geometry import Matrix
from pyippdme.twin.spec import MachineSpec, ToolSpec
from pyippdme.types.tool_id import (
    ContinuousAlignMode,
    DataAcquisitionFunction,
    DataAcquisitionFunctions,
    FixedAlignMode,
    IndexedAlignMode,
    ToolId,
    ToolIdOptical1D,
    ToolIdOptical2DRs,
    ToolIdOptical3D,
    ToolIdTactileMeasuring,
    ToolIdTactileTouchTrigger,
)
from pyippdme.types.vec3 import Vec3, normalize

#: Height of the head housing above the pivot, up to the bottom of the quill (mm).
HOUSING_HEIGHT = 35.0
#: How far above the bottom ball the arms of a star stylus sit (mm).
ARM_DROP = 6.0

_TOUCH_FUNCTIONS = ("GoTo", "PtMeas")
_SCAN_FUNCTIONS = (
    "GoTo",
    "PtMeas",
    "ScanOnLine",
    "ScanOnCircle",
    "ScanOnHelix",
    "ScanOnCurve",
    "PTMeasSelfCenter",
    "PTMeasSelfCenterLocked",
)
_OPTICAL_FUNCTIONS = (
    "GoTo",
    "DataAcquire",
    "DeleteAcquisition",
    "DeleteAllAcquisition",
    "GetRawDataBin",
    "GetRawDataShaMem",
    "ReleaseShaMem",
)
#: Modes that acquire raw data optically instead of touching.
OPTICAL_MODES = ("laser", "point_laser", "area", "camera")


def join_length(spec: ToolSpec) -> float:
    """Distance from the pivot to where the extension (or probe) starts."""
    return 14.0 if spec.head == "fixed" else 30.0


def drop(spec: ToolSpec) -> float:
    """Distance from the pivot down to the bottom ball (or the window of an optical sensor)."""
    length = join_length(spec) + spec.extension + spec.probe_length
    if spec.mode in OPTICAL_MODES or spec.mode == "none":
        return length
    return length + spec.shaft_length


def tip_offset(spec: ToolSpec) -> Vec3:
    """Tool centre point relative to the pivot, with the head pointing straight down."""
    z = -drop(spec)
    if spec.tip == "down":
        return (0.0, 0.0, z)
    sign = 1.0 if spec.tip[0] == "+" else -1.0
    arm = sign * spec.arm_length
    return (arm, 0.0, z + ARM_DROP) if spec.tip[1] == "x" else (0.0, arm, z + ARM_DROP)


def functions(spec: ToolSpec) -> tuple[str, ...]:
    """The ``basicfunction`` list (Annex G) of a tool: what the protocol may ask of it."""
    if spec.mode in OPTICAL_MODES:
        return _OPTICAL_FUNCTIONS
    return {
        "touch": _TOUCH_FUNCTIONS,
        "head_touch": _TOUCH_FUNCTIONS,
        "scanning": _SCAN_FUNCTIONS,
        "none": ("GoTo",),
    }[spec.mode]


def tool_id(spec: ToolSpec) -> ToolId:
    """The ``Tool.Id()`` description of a tool: type, functions, axes and alignment."""
    articulated = spec.head != "fixed"
    axes = ("X", "Y", "Z", "A", "B") if articulated else ("X", "Y", "Z")
    align = (
        FixedAlignMode()
        if spec.head == "fixed"
        else IndexedAlignMode(aligncaa=True, step=spec.index_step)
        if spec.head == "indexed"
        else ContinuousAlignMode(aligncaa=True)
    )
    common = {
        "id": spec.name,
        "basic_functions": functions(spec),
        "cnc_axes": axes,
        "supports_optimization_mode": False,
        "align_mode": align,
        "move_on_acquisition": False,
    }
    if spec.mode in OPTICAL_MODES:
        size = DataAcquisitionFunction(spec.line_width, None, spec.depth_range)
        daq = DataAcquisitionFunctions(size, size, size)
        optical = {"point_laser": ToolIdOptical1D, "camera": ToolIdOptical2DRs}.get(
            spec.mode, ToolIdOptical3D
        )
        return optical(**common, daq_functions=daq)  # type: ignore[arg-type]
    if spec.mode == "scanning":
        return ToolIdTactileMeasuring(**common)  # type: ignore[arg-type]
    return ToolIdTactileTouchTrigger(**common)  # type: ignore[arg-type]


def head_rotation(spec: ToolSpec, axis: Vec3) -> tuple[Matrix, Vec3, tuple[float, float] | None]:
    """Rotation of the head for a tool axis (towards the head): ``R = Rz(B) Ry(A)``.

    A fixed mount ignores the axis; an indexing head snaps A (0..105 degrees) and B
    (-180..180) to its step; a continuous head follows the axis exactly. Returns the
    4x4 rotation, the axis the head really ends up with, and for an indexing head the
    ``(A, B)`` position it sits at (a tool is qualified per position).
    """
    if spec.head == "fixed":
        return np.eye(4), (0.0, 0.0, 1.0), None
    d = normalize(axis)
    a = math.degrees(math.acos(max(-1.0, min(1.0, d[2]))))
    b = math.degrees(math.atan2(d[1], d[0])) if abs(d[0]) + abs(d[1]) > 1e-9 else 0.0
    if spec.head == "indexed":
        step = spec.index_step
        a = min(round(a / step) * step, 105.0)
        b = round(b / step) * step
    r = geometry.rotation((0, 0, 1), b) @ geometry.rotation((0, 1, 0), a)
    position = (a, b) if spec.head == "indexed" else None
    return r, (float(r[0, 2]), float(r[1, 2]), float(r[2, 2])), position


def pivot_for(spec: ToolSpec, tcp: Vec3, axis: Vec3) -> Vec3:
    """Where the head pivot is when the tool centre point is at ``tcp`` and the tool points ``axis``."""
    rotation, _, _ = head_rotation(spec, axis)
    offset = rotation[:3, :3] @ np.asarray(tip_offset(spec))
    return (tcp[0] - float(offset[0]), tcp[1] - float(offset[1]), tcp[2] - float(offset[2]))


# -- tool rack ---------------------------------------------------------------------------


def rack_keys(tools: dict[str, ToolSpec]) -> list[str]:
    """One port per stylus/probe module: the tips of a star share one."""
    keys: list[str] = []
    for tool in tools.values():
        if tool.mode != "none" and tool.rack_key not in keys:
            keys.append(tool.rack_key)
    return keys


def rack_slots(spec: MachineSpec, keys: list[str]) -> dict[str, Vec3]:
    """Positions of the tool rack's ports (top of each port), along the back of the table."""
    count = max(len(keys), 1)
    pitch = min(80.0, 0.8 * spec.travel[0] / count)
    x0 = 0.08 * spec.travel[0] + 20.0
    y = spec.travel[1] - 45.0
    z = spec.table_top_z + 14.0 + 6.0
    return {key: (x0 + i * pitch, y, z) for i, key in enumerate(keys)}
