# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""A measuring program for the check artefact, written against the public client API.

It is what a metrology program does against a real machine: pick a tool, qualify it on the
reference sphere, probe the features, and move on to the next measurement mode. Nothing in
it knows about the simulation, so the same program can run against any I++ DME server that
has the artefact on its table. Run it with a client connected to the twin, then evaluate the
points the twin recorded with :meth:`pyippdme.twin.twin.DigitalTwin.evaluate`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

from pyippdme.client.builders import AcquisitionPoint
from pyippdme.client.model import IppDmeMachine
from pyippdme.twin.artifact import ArtifactData
from pyippdme.types.vec3 import Vec3

#: Measurement modes the program covers, in the order it runs them.
MODES = ("touch", "head_touch", "scanning", "laser", "point_laser", "area", "camera")


@dataclass(slots=True)
class CheckOptions:
    """Which tools to use for which mode and how the program moves."""

    modes: tuple[str, ...] = ("touch", "scanning", "laser")
    tools: dict[str, str] = field(
        default_factory=lambda: {
            "touch": "RefTool",
            "head_touch": "RevoHeadTouch",
            "scanning": "ScanSP25",
            "laser": "LaserLine",
            "point_laser": "LaserPoint",
            "area": "AreaScanner",
            "camera": "Camera2D",
        }
    )
    #: Clearance from the surface when approaching a touch point (mm).
    clearance: float = 10.0
    #: Qualify each tactile tool on the reference sphere before measuring with it.
    qualify: bool = True
    #: Sensor stand-offs for the optical modes (mm), as defined by the tools.
    standoffs: dict[str, float] = field(
        default_factory=lambda: {
            "laser": 60.0,
            "point_laser": 30.0,
            "area": 150.0,
            "camera": 80.0,
        }
    )
    #: Widths of the optical sensors' fields, to space the sweep rows (mm).
    widths: dict[str, float] = field(
        default_factory=lambda: {"laser": 30.0, "point_laser": 4.0, "area": 60.0, "camera": 30.0}
    )


Progress = Callable[[str], None]


def _above(point: Vec3, normal: Vec3, distance: float) -> Vec3:
    return (
        point[0] + normal[0] * distance,
        point[1] + normal[1] * distance,
        point[2] + normal[2] * distance,
    )


async def _tactile(
    machine: IppDmeMachine,
    data: ArtifactData,
    options: CheckOptions,
    mode: str,
    progress: Progress,
) -> None:
    await machine.tool_changer.change_tool(options.tools[mode])
    if options.qualify:
        progress(f"qualifying {options.tools[mode]}")
        await machine.tool.re_qualify()
    safe_z = data.extent[1][2] + 60.0
    progress(f"{mode}: {len(data.touch)} touch points")
    for plan in data.touch:
        approach = _above(
            plan.point,
            plan.normal,
            options.clearance if plan.clearance is None else plan.clearance,
        )
        # Retract to a safe height between points, as a program does that cannot see the part.
        await machine.cart_cmm.go_to(z=safe_z)
        await machine.cart_cmm.go_to(x=approach[0], y=approach[1])
        await machine.cart_cmm.go_to(x=approach[0], y=approach[1], z=approach[2])
        await machine.cart_cmm.pt_meas(
            x=plan.point[0], y=plan.point[1], z=plan.point[2], ijk=plan.normal
        )
    await machine.cart_cmm.go_to(z=safe_z)


async def _scanning(
    machine: IppDmeMachine, data: ArtifactData, options: CheckOptions, progress: Progress
) -> None:
    await machine.tool_changer.change_tool(options.tools["scanning"])
    if options.qualify:
        progress(f"qualifying {options.tools['scanning']}")
        await machine.tool.re_qualify()
    safe_z = data.extent[1][2] + 60.0
    progress(f"scanning: {len(data.lines)} lines, {len(data.circles)} circles")
    for line in data.lines:
        await machine.cart_cmm.go_to(z=safe_z)
        await machine.cart_cmm.go_to(x=line.start[0], y=line.start[1])
        async for _ in machine.scanning.scan_on_line(line.start, line.end, line.normal, line.step):
            pass
    for circle in data.circles:
        await machine.cart_cmm.go_to(z=safe_z)
        await machine.cart_cmm.go_to(x=circle.start[0], y=circle.start[1])
        async for _ in machine.scanning.scan_on_circle(
            circle.center, circle.start, circle.normal, circle.delta, 0.0, circle.step
        ):
            pass
    await machine.cart_cmm.go_to(z=safe_z)


def sweeps_for(data: ArtifactData, standoff: float, width: float) -> list[tuple[Vec3, Vec3]]:
    """Rows of sensor positions that cover the artefact's footprint, as ``(start, end)`` pairs.

    The sensor window sits ``standoff`` above the base plate, rows are ``0.9 * width`` apart.
    """
    (x0, y0, _), (x1, y1, _) = data.extent
    z = data.top + standoff
    count = max(1, math.ceil((y1 - y0) / (0.9 * width)))
    rows = []
    for i in range(count):
        y = (
            y0 + width / 2 + i * (y1 - y0 - width) / max(count - 1, 1)
            if count > 1
            else (y0 + y1) / 2
        )
        rows.append(((x0 + 5.0, y, z), (x1 - 5.0, y, z)))
    return rows


async def _optical(
    machine: IppDmeMachine,
    data: ArtifactData,
    options: CheckOptions,
    mode: str,
    progress: Progress,
) -> None:
    await machine.tool_changer.change_tool(options.tools[mode])
    rows = sweeps_for(data, options.standoffs[mode], options.widths[mode])
    progress(f"{mode}: {len(rows)} sweeps")
    safe_z = data.extent[1][2] + 60.0 + options.standoffs[mode]
    await machine.cart_cmm.go_to(z=min(safe_z, data.top + options.standoffs[mode] + 100.0))
    for i, (start, end) in enumerate(rows):
        points = [
            AcquisitionPoint(start, (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
            AcquisitionPoint(end, (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
        ]
        await machine.raw_data.data_acquire(f"{mode}{i}", "Sweep", "default", points)
        await machine.raw_data.delete_acquisition(f"{mode}{i}")


async def run_check_program(
    machine: IppDmeMachine,
    data: ArtifactData,
    options: CheckOptions | None = None,
    progress: Progress = lambda _: None,
) -> None:
    """Measure the check artefact in the requested modes.

    ``data`` is the artefact's description in machine coordinates (``placed`` for its pose).
    The session must be started; the program homes the machine if it is not homed.
    """
    options = options or CheckOptions()
    if not await machine.dme.is_homed():
        progress("homing")
        await machine.dme.home()
    for mode in options.modes:
        if mode in ("touch", "head_touch"):
            await _tactile(machine, data, options, mode, progress)
        elif mode == "scanning":
            await _scanning(machine, data, options, progress)
        elif mode in ("laser", "point_laser", "area", "camera"):
            await _optical(machine, data, options, mode, progress)
        else:
            raise ValueError(f"unknown mode {mode!r}; use one of {', '.join(MODES)}")


async def run_check_against_server(
    host: str,
    port: int,
    data: ArtifactData,
    options: CheckOptions | None = None,
    progress: Progress = lambda _: None,
) -> None:
    """Connect to a server (the twin's own, for instance), run the check program, disconnect.

    A server accepts one client at a time (5.8), so this fails if another client is connected.
    """
    machine = await IppDmeMachine.connect(host, port)
    try:
        await machine.start_session()
        await run_check_program(machine, data, options, progress)
        await machine.end_session()
    finally:
        await machine.close()
