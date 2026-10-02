# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""The default, simulated :class:`~pyippdme.server.backend.MachineBackend`.

No real hardware, only geometry - see :mod:`pyippdme.server.backend` for the
Protocol this implements and why a real integrator would supply their own
instead.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import AsyncIterator

from pyippdme.server.backend import CancellationToken
from pyippdme.types.vec3 import Vec3, add, cross, norm, normalize, scale, sub


class SimulatedBackend:
    """The default backend: no real hardware, only geometry.

    Points are yielded one at a time with a small delay between them, both
    to resemble real (non-instantaneous) motion and so that cancellation via
    ``AbortE()`` has something observable to interrupt in tests and demos.
    """

    def __init__(self, *, step_delay: float = 0.01) -> None:
        self._step_delay = step_delay

    async def scan_line(
        self, start: Vec3, end: Vec3, normal: Vec3, step_w: float, cancel: CancellationToken
    ) -> AsyncIterator[Vec3]:
        del normal  # only used for validation, already done by the caller
        length = norm(sub(end, start))
        direction = scale(sub(end, start), 1.0 / length)
        steps = max(1, round(length / step_w))
        for s in range(steps + 1):
            if cancel.is_set():
                return
            yield add(start, scale(direction, length * s / steps))
            if s < steps:
                await asyncio.sleep(self._step_delay)

    async def scan_circle(
        self,
        center: Vec3,
        start: Vec3,
        normal: Vec3,
        delta_deg: float,
        step_w_deg: float,
        cancel: CancellationToken,
    ) -> AsyncIterator[Vec3]:
        radial = normalize(sub(start, center))
        plane_normal = normalize(normal)
        tangential = cross(plane_normal, radial)
        radius = norm(sub(start, center))
        steps = max(1, round(abs(delta_deg) / step_w_deg))
        delta_rad = math.radians(delta_deg)
        for s in range(steps + 1):
            if cancel.is_set():
                return
            theta = delta_rad * s / steps
            yield add(
                center,
                add(
                    scale(radial, radius * math.cos(theta)),
                    scale(tangential, radius * math.sin(theta)),
                ),
            )
            if s < steps:
                await asyncio.sleep(self._step_delay)

    async def scan_helix(
        self,
        center: Vec3,
        start: Vec3,
        normal: Vec3,
        delta_deg: float,
        step_w_deg: float,
        pitch: float,
        cancel: CancellationToken,
    ) -> AsyncIterator[Vec3]:
        """Follow :meth:`scan_circle`'s path, adding a ``pitch``-per-turn rise along ``normal``."""
        radial = normalize(sub(start, center))
        plane_normal = normalize(normal)
        tangential = cross(plane_normal, radial)
        radius = norm(sub(start, center))
        steps = max(1, round(abs(delta_deg) / step_w_deg))
        delta_rad = math.radians(delta_deg)
        for s in range(steps + 1):
            if cancel.is_set():
                return
            theta_deg = delta_deg * s / steps
            theta_rad = delta_rad * s / steps
            rise = scale(plane_normal, pitch * theta_deg / 360.0)
            yield add(
                center,
                add(
                    add(
                        scale(radial, radius * math.cos(theta_rad)),
                        scale(tangential, radius * math.sin(theta_rad)),
                    ),
                    rise,
                ),
            )
            if s < steps:
                await asyncio.sleep(self._step_delay)
