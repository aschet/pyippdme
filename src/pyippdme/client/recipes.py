# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Ready-made command lines for the usual measuring tasks, for any front end.

Each function returns the text of one command (or the few commands a task needs, in order) that
:func:`~pyippdme.protocol.parser.parse_method` accepts and a server answers::

    goto_line(x=10, y=20)                 # 'GoTo(X(10),Y(20))'
    pt_meas_line((10, 20, 0), (0, 0, 1))  # 'PtMeas(X(10),Y(20),Z(0),IJK(0,0,1))'
    scan_line_lines((0, 0, 0), (50, 0, 0), (0, 0, 1), 2.0)  # ['OnScanReport(X,Y,Z)', ...]
"""

from __future__ import annotations

from pyippdme.client import builders
from pyippdme.protocol.ast import BasicName, Method, String
from pyippdme.types.csy import CoordinateTransform
from pyippdme.types.vec3 import Vec3

__all__ = [
    "AXIS_DIRECTIONS",
    "SPEED_PARAMETERS",
    "change_tool_line",
    "get_csy_transformation_line",
    "goto_line",
    "named_csy_line",
    "pt_meas_line",
    "scan_circle_lines",
    "scan_line_lines",
    "set_coord_system_line",
    "set_csy_transformation_line",
    "set_tool_line",
    "speed_line",
    "speed_lines",
    "status_lines",
]

#: Probing directions along the machine axes, for buttons.
AXIS_DIRECTIONS: dict[str, Vec3] = {
    "+X": (1.0, 0.0, 0.0),
    "-X": (-1.0, 0.0, 0.0),
    "+Y": (0.0, 1.0, 0.0),
    "-Y": (0.0, -1.0, 0.0),
    "+Z": (0.0, 0.0, 1.0),
    "-Z": (0.0, 0.0, -1.0),
}

#: The parameter blocks of the active tool and what they hold (6.10.4, Table 77, 6.13.2).
SPEED_PARAMETERS = {
    "GoToPar": "Move (GoTo)",
    "PtMeasPar": "Point measurement",
    "ScanPar": "Scan",
}

_SCAN_FIELDS = ("X", "Y", "Z")


def _line(name: str, args: tuple[object, ...]) -> str:
    return Method(name, args).to_wire()  # type: ignore[arg-type]


def goto_line(
    x: float | None = None,
    y: float | None = None,
    z: float | None = None,
    *,
    r: float | None = None,
    sync: bool | None = None,
    relative: bool = False,
) -> str:
    """``GoTo`` (absolute) or ``Step`` (relative); axes left as ``None`` stay where they are."""
    if relative:
        return _line("Step", builders.step(x, y, z, r=r, sync=sync))
    return _line("GoTo", builders.go_to(x, y, z, r=r, sync=sync))


def pt_meas_line(position: Vec3, direction: Vec3 | None = None, *, r: float | None = None) -> str:
    """``PtMeas`` at a nominal position, probing along ``direction`` (6.12.1)."""
    return _line(
        "PtMeas", builders.pt_meas(position[0], position[1], position[2], ijk=direction, r=r)
    )


def _scan_report() -> str:
    return _line("OnScanReport", builders.on_scan_report(*_SCAN_FIELDS))


def scan_line_lines(
    start: Vec3, end: Vec3, direction: Vec3, step_width: float, *, rotary: bool | None = None
) -> list[str]:
    """Report X,Y,Z per scan point, then scan a straight line (6.13.2.1)."""
    scan = _line("ScanOnLine", builders.positional(start, end, direction, step_width, rotary))
    return [_scan_report(), scan]


def scan_circle_lines(
    center: Vec3,
    start: Vec3,
    normal: Vec3,
    delta: float,
    surface_angle: float,
    step_width: float,
    *,
    pitch: float = 0.0,
    rotary: bool | None = None,
) -> list[str]:
    """Scan a circular arc, or a helix when ``pitch`` is not zero (6.13.2.1)."""
    if pitch:
        arguments = builders.scan_on_circle(
            center, start, normal, delta, surface_angle, step_width, rotary, pitch=pitch
        )
        return [_scan_report(), _line("ScanOnHelix", arguments)]
    arguments = builders.scan_on_circle(
        center, start, normal, delta, surface_angle, step_width, rotary
    )
    return [_scan_report(), _line("ScanOnCircle", arguments)]


def change_tool_line(name: str) -> str:
    return _line("ChangeTool", (String(name),))


def set_tool_line(name: str) -> str:
    return _line("SetTool", (String(name),))


def set_coord_system_line(csy: str) -> str:
    """Choose the coordinate system the client works in (6.5.2)."""
    return _line("SetCoordSystem", (BasicName(csy),))


def set_csy_transformation_line(
    csy: str, offset: Vec3, theta: float, psi: float, phi: float
) -> str:
    """Place ``csy`` relative to its parent: offset and the Euler angles of 6.5.1."""
    transform = CoordinateTransform(*offset, theta, psi, phi)
    return _line(
        "SetCsyTransformation", (BasicName(csy), *builders.set_csy_transformation(transform))
    )


def get_csy_transformation_line(csy: str) -> str:
    return _line("GetCsyTransformation", (BasicName(csy),))


def named_csy_line(command: str, name: str) -> str:
    """``SaveActiveCoordSystem``, ``LoadCoordSystem`` or ``DeleteCoordSystem`` for ``name``."""
    return _line(command, (String(name),))


def speed_lines(block: str, speed: float | None, accel: float | None) -> list[str]:
    """Set the speed and acceleration of a parameter block (see :data:`SPEED_PARAMETERS`)."""
    lines = []
    if speed is not None:
        lines.append(_line("SetProp", builders.set_prop(f"Tool.{block}.Speed", speed)))
    if accel is not None:
        lines.append(_line("SetProp", builders.set_prop(f"Tool.{block}.Accel", accel)))
    return lines


def speed_line(block: str, parameter: str, value: float) -> str:
    """Set one parameter of a block, e.g. ``speed_line("PtMeasPar", "Retract", -1)``."""
    return _line("SetProp", builders.set_prop(f"Tool.{block}.{parameter}", value))


def status_lines() -> list[str]:
    """List what a front end polls for its status bar: the position and the error state."""
    return [
        _line("Get", builders.get("X", "Y", "Z")),
        "GetXtdErrStatus()",
        "GetCoordSystem()",
        "IsUserEnabled()",
        _line("GetProp", builders.get_prop("Tool.Name")),
    ]
