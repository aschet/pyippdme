# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""The twin's :class:`~pyippdme.server.backend.MachineBackend`: timed scans that touch the part."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from pyippdme.server.backend import CancellationToken
from pyippdme.simulation.backend import SimulatedBackend
from pyippdme.types.vec3 import Vec3, add, norm, normalize, scale, sub

if TYPE_CHECKING:
    from pyippdme.twin.twin import DigitalTwin

#: How far from the nominal path a probe still finds the surface to follow (mm).
_FOLLOW_RANGE = 6.0
_APPROACH = 3.0


class TwinBackend:
    """Streams the scan points at the machine's scanning speed.

    The nominal path comes from :class:`~pyippdme.simulation.backend.SimulatedBackend`.
    Each point is then paced by the scanning speed, shown moving in the twin, and, for
    line scans, projected onto the CAD surface along the probing direction (the scan's
    ``normal``) when the surface is within a few millimetres, with the machine's
    measuring noise added. Points over empty space stay on the nominal path.
    """

    def __init__(self, twin: DigitalTwin) -> None:
        self._twin = twin
        self._path = SimulatedBackend(step_delay=0.0)

    async def _paced(
        self, points: AsyncIterator[Vec3], cancel: CancellationToken, probe_direction: Vec3 | None
    ) -> AsyncIterator[Vec3]:
        twin = self._twin
        speed = max(twin.machine.spec.scanning_speed * twin.speed_override, 1e-3)
        previous: Vec3 | None = None
        last = time.monotonic()
        async for nominal in points:
            if cancel.is_set():
                return
            measured = self._follow(nominal, probe_direction)
            if previous is not None and twin.time_scale > 0.0:
                due = norm(sub(measured, previous)) / speed / twin.time_scale
                wait = due - (time.monotonic() - last)
                if wait > 0.0:
                    await asyncio.sleep(wait)
            last = time.monotonic()
            previous = measured
            twin.publish_motion(measured, twin.snapshot().rotary)
            yield measured

    def _follow(self, nominal: Vec3, probe_direction: Vec3 | None) -> Vec3:
        twin = self._twin
        if probe_direction is None or norm(probe_direction) < 1e-9:
            return nominal
        unit = normalize(probe_direction)
        origin = add(nominal, scale(unit, _APPROACH))
        hit = twin.cast(origin, scale(unit, -1.0))
        if hit is None or hit[1] > _APPROACH + _FOLLOW_RANGE:
            return nominal
        point = hit[0]
        if twin.noise_enabled:
            sigma = twin.machine.spec.accuracy.sigma_mm(norm(point))
            point = add(point, scale(unit, twin.rng.gauss(0.0, sigma)))
        return point

    def scan_line(
        self, start: Vec3, end: Vec3, normal: Vec3, step_w: float, cancel: CancellationToken
    ) -> AsyncIterator[Vec3]:
        return self._paced(self._path.scan_line(start, end, normal, step_w, cancel), cancel, normal)

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
            None,
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
            None,
        )
