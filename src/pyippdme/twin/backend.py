# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""The twin's :class:`~pyippdme.server.backend.MachineBackend`: timed scans that touch the part."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING

from pyippdme.protocol.errors import ErrorCode, ErrorSeverity
from pyippdme.server.backend import CancellationToken
from pyippdme.server.contour import ContourConstraint, ContourScan, ContourStop, trace_contour
from pyippdme.server.motion import MotionError
from pyippdme.simulation.backend import SimulatedBackend
from pyippdme.types.vec3 import Vec3, add, norm, normalize, scale, sub

if TYPE_CHECKING:
    from pyippdme.twin.twin import DigitalTwin
    from pyippdme.types.csy import CsyContext

#: How far from the nominal path a probe still finds the surface to follow (mm).
_FOLLOW_RANGE = 6.0
_APPROACH = 3.0


class TwinBackend:
    """Streams the scan points at the scan speed and the scanning rate of the tool.

    The nominal path comes from :class:`~pyippdme.simulation.backend.SimulatedBackend`.
    Each point is then paced by the ``ScanPar`` speed (limited by the machine) and the
    tool's points per second, shown moving in the twin, and projected onto the CAD surface
    when the surface is within a few millimetres of the path: along the scan's ``normal``
    for line scans, and radially (outwards or inwards, whichever finds the surface first)
    for circles and helices. The machine's and the tool's measuring errors are added.
    Points over empty space stay on the nominal path.
    """

    def __init__(self, twin: DigitalTwin) -> None:
        self._twin = twin
        self._path = SimulatedBackend(step_delay=0.0)

    async def _paced(
        self,
        points: AsyncIterator[Vec3],
        cancel: CancellationToken,
        directions: Callable[[Vec3], tuple[Vec3, ...] | None],
    ) -> AsyncIterator[Vec3]:
        twin = self._twin
        twin.check_ready("Scan")
        context = twin.csy_context()
        params = twin.protocol_parameters()
        speed = max(params["scan_speed"] * twin.speed_override, 1e-3)
        tool = twin.toolkit.spec(twin.tool_name())
        interval = 1.0 / tool.scan_rate if tool.scan_rate > 0 else 0.0
        previous: Vec3 | None = None
        last = time.monotonic()
        batch: list[Vec3] = []
        try:
            async for nominal in points:
                if cancel.is_set():
                    return
                twin.check_ready("Scan")
                nominal = twin.to_machine(nominal, context)
                measured, unit = self._follow(nominal, directions(nominal))
                self._check_collision(measured, unit, context)
                if previous is not None and twin.time_scale > 0.0:
                    due = max(norm(sub(measured, previous)) / speed, interval) / twin.time_scale
                    wait = due - (time.monotonic() - last)
                    if wait > 0.0:
                        await asyncio.sleep(wait)
                last = time.monotonic()
                previous = measured
                twin.publish_motion(measured, twin.snapshot().rotary)
                batch.append(measured)
                yield twin.to_client(measured, context)
        finally:
            if batch:
                twin.record_scan_points(batch, "scanning")

    def _check_collision(self, point: Vec3, unit: Vec3 | None, context: CsyContext) -> None:
        """Stop the scan if the stylus or the probe body touches anything but the scanned point."""
        twin = self._twin
        if unit is None:
            return
        offset = scale(unit, twin.toolkit.spec(twin.tool_name()).ball_radius + 0.05)
        with twin.lock:
            hits, _ = twin._collisions(point, twin.snapshot().rotary, twin.tool_name(), offset)
        if hits:
            what = sorted(hits)[0]
            twin.report_collision(what, point)
            raise twin.client_error(
                MotionError(
                    ErrorSeverity.CRITICAL,
                    ErrorCode.COLLISION,
                    "Scan",
                    f"Collision with {what}",
                    point,
                ),
                context,
            )

    def _follow(
        self, nominal: Vec3, directions: tuple[Vec3, ...] | None
    ) -> tuple[Vec3, Vec3 | None]:
        twin = self._twin
        if not directions:
            return nominal, None
        best: tuple[Vec3, float, Vec3] | None = None
        for direction in directions:
            if norm(direction) < 1e-9:
                continue
            unit = normalize(direction)
            hit = twin.cast(add(nominal, scale(unit, _APPROACH)), scale(unit, -1.0))
            if (
                hit is not None
                and hit[1] <= _APPROACH + _FOLLOW_RANGE
                and (best is None or hit[1] < best[1])
            ):
                best = (hit[0], hit[1], unit)
        if best is None:
            return nominal, None
        point, _, unit = best
        if twin.noise_enabled:
            spec = twin.toolkit.spec(twin.tool_name())
            placement = twin.placement(twin._pos)
            point = twin._apply_probe_errors(point, unit, spec, placement)
        return point, unit

    def scan_line(
        self, start: Vec3, end: Vec3, normal: Vec3, step_w: float, cancel: CancellationToken
    ) -> AsyncIterator[Vec3]:
        twin = self._twin
        machine_normal = twin.to_machine_direction(normal)
        return self._paced(
            self._path.scan_line(start, end, normal, step_w, cancel),
            cancel,
            lambda _: (machine_normal,),
        )

    def _to_machine_scan(self, scan: ContourScan) -> ContourScan:
        twin = self._twin
        context = twin.csy_context()
        c = scan.constraint
        constraint = ContourConstraint(
            c.kind,
            None if c.normal is None else twin.to_machine_direction(c.normal, context),
            None if c.axis_point is None else twin.to_machine(c.axis_point, context),
            None if c.axis is None else twin.to_machine_direction(c.axis, context),
        )
        s = scan.stop
        stop = ContourStop(
            s.kind,
            twin.to_machine(s.point, context),
            None if s.direction is None else twin.to_machine_direction(s.direction, context),
            s.size,
        )
        start = twin.to_machine(scan.start, context)
        return ContourScan(
            constraint,
            start,
            twin.to_machine_direction(scan.probe, context),
            twin.to_machine(scan.direction_point, context),
            scan.step,
            stop,
            scan.count,
            twin.to_machine_direction(scan.end_direction, context)
            if norm(scan.end_direction) > 0.0
            else scan.end_direction,
        )

    async def scan_contour(
        self, scan: ContourScan, cancel: CancellationToken
    ) -> AsyncIterator[Vec3]:
        """Follow the unknown contour of the CAD part until the stop element (Figures 35-39)."""
        twin = self._twin
        twin.check_ready("Scan")
        context = twin.csy_context()
        machine_scan = self._to_machine_scan(scan)
        params = twin.protocol_parameters()
        speed = max(params["scan_speed"] * twin.speed_override, 1e-3)
        tool = twin.toolkit.spec(twin.tool_name())
        interval = 1.0 / tool.scan_rate if tool.scan_rate > 0 else 0.0
        spec = tool
        placement = twin.placement(twin._pos)

        def probe_surface(origin: Vec3, direction: Vec3) -> tuple[Vec3, Vec3] | None:
            hit = twin.cast(origin, direction)
            if hit is None:
                return None
            normal = twin.surface_normal(hit[0], scale(direction, -1.0))
            return hit[0], normal

        previous: Vec3 | None = None
        last = time.monotonic()
        batch: list[Vec3] = []
        try:
            for nominal in trace_contour(machine_scan, probe_surface):
                if cancel.is_set():
                    return
                twin.check_ready("Scan")
                point = nominal
                if twin.noise_enabled:
                    point = twin._apply_probe_errors(
                        nominal, normalize(machine_scan.probe), spec, placement
                    )
                if previous is not None and twin.time_scale > 0.0:
                    due = max(norm(sub(point, previous)) / speed, interval) / twin.time_scale
                    wait = due - (time.monotonic() - last)
                    if wait > 0.0:
                        await asyncio.sleep(wait)
                else:
                    await asyncio.sleep(0)
                last = time.monotonic()
                previous = point
                twin.publish_motion(point, twin.snapshot().rotary)
                batch.append(point)
                yield twin.to_client(point, context)
        finally:
            if batch:
                twin.record_scan_points(batch, "scanning")

    @staticmethod
    def _radial(center: Vec3, axis: Vec3) -> Callable[[Vec3], tuple[Vec3, ...] | None]:
        """Probing directions for a circle or helix: radially out and in, whichever finds it."""
        unit = normalize(axis)

        def directions(nominal: Vec3) -> tuple[Vec3, ...] | None:
            offset = sub(nominal, center)
            along = sum(a * b for a, b in zip(offset, unit, strict=True))
            radial = sub(offset, scale(unit, along))
            return (radial, scale(radial, -1.0)) if norm(radial) > 1e-9 else None

        return directions

    def scan_circle(
        self,
        center: Vec3,
        start: Vec3,
        normal: Vec3,
        delta_deg: float,
        step_w_deg: float,
        cancel: CancellationToken,
    ) -> AsyncIterator[Vec3]:
        return self._paced(
            self._path.scan_circle(center, start, normal, delta_deg, step_w_deg, cancel),
            cancel,
            self._radial(self._twin.to_machine(center), self._twin.to_machine_direction(normal)),
        )

    def scan_helix(
        self,
        center: Vec3,
        start: Vec3,
        normal: Vec3,
        delta_deg: float,
        step_w_deg: float,
        pitch: float,
        cancel: CancellationToken,
    ) -> AsyncIterator[Vec3]:
        return self._paced(
            self._path.scan_helix(center, start, normal, delta_deg, step_w_deg, pitch, cancel),
            cancel,
            self._radial(self._twin.to_machine(center), self._twin.to_machine_direction(normal)),
        )
