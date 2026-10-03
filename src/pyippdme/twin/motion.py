# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Timed motion with machine limits and collisions for the twin."""

from __future__ import annotations

import asyncio
import math
from typing import TYPE_CHECKING

from pyippdme.protocol.errors import ErrorCode, ErrorSeverity, ServerError
from pyippdme.server.motion import MotionError, MotionRequest
from pyippdme.types.vec3 import Vec3, add, norm, scale, sub

if TYPE_CHECKING:
    from pyippdme.twin.twin import DigitalTwin

#: Hold-off so a stopped machine is not left touching what it hit (mm).
_BACKOFF = 0.5
_FRAME = 1.0 / 60.0
_EPS = 1e-6


def travel_time(distance: float, vmax: float, accel: float) -> float:
    """Duration of a trapezoidal (or triangular) velocity profile over ``distance``."""
    if distance <= 0.0:
        return 0.0
    ramp = vmax * vmax / accel
    if distance >= ramp:
        return distance / vmax + vmax / accel
    return 2.0 * math.sqrt(distance / accel)


def travelled(t: float, distance: float, vmax: float, accel: float) -> float:
    """Distance covered ``t`` seconds into the profile of :func:`travel_time`."""
    total = travel_time(distance, vmax, accel)
    if t >= total:
        return distance
    t_acc = min(vmax / accel, total / 2.0)
    v_peak = accel * t_acc
    if t < t_acc:
        return 0.5 * accel * t * t
    if t > total - t_acc:
        rest = total - t
        return distance - 0.5 * accel * rest * rest
    return 0.5 * accel * t_acc * t_acc + v_peak * (t - t_acc)


class TwinMotion:
    """A :class:`~pyippdme.server.motion.MotionModel` driven by the twin's machine and scene."""

    def __init__(self, twin: DigitalTwin) -> None:
        self._twin = twin

    async def travel(self, request: MotionRequest) -> Vec3:
        twin = self._twin
        spec = twin.machine.spec
        cause = request.cause
        if spec.require_home and not request.homed:
            raise ServerError(
                ErrorSeverity.CRITICAL,
                ErrorCode.UNABLE_TO_MOVE,
                cause,
                "The machine is not homed; send Home() first",
            )
        for value, limit, axis in zip(request.end, spec.travel, "XYZ", strict=True):
            if value < -_EPS or value > limit + _EPS:
                raise ServerError(
                    ErrorSeverity.ERROR,
                    ErrorCode.TARGET_OUT_OF_MACHINE_VOLUME,
                    cause,
                    f"{axis}={value:g} is outside the machine volume 0..{limit:g}",
                )
        probing = cause in ("PtMeas", "PtMeasSelfCenter", "PtMeasSelfCenterLocked")
        stop_at, hit = twin.first_collision(request, probing=probing)
        end = stop_at if hit is not None else request.end
        rotary_end = request.rotary_end if hit is None else request.rotary_start

        distance = norm(sub(end, request.start))
        rotary_distance = abs(rotary_end - request.rotary_start)
        vmax = spec.max_speed * twin.speed_override
        duration = max(
            travel_time(distance, vmax, spec.acceleration),
            travel_time(rotary_distance, spec.rotary_speed, spec.rotary_speed * 4.0),
        )
        position = request.start
        if twin.time_scale > 0.0 and duration > 0.0:
            loop = asyncio.get_running_loop()
            started = loop.time()
            while not request.cancel.is_set():
                t = (loop.time() - started) * twin.time_scale
                if t >= duration:
                    break
                fraction = t / duration
                along = (
                    travelled(t, distance, vmax, spec.acceleration) / distance if distance else 0.0
                )
                position = add(request.start, scale(sub(end, request.start), min(along, 1.0)))
                rotary = request.rotary_start + (rotary_end - request.rotary_start) * fraction
                twin.publish_motion(position, rotary)
                await asyncio.sleep(_FRAME)
            if request.cancel.is_set():
                twin.publish_motion(position, request.rotary_start)
                return position
        twin.publish_motion(end, rotary_end)
        if hit is not None:
            twin.report_collision(hit, end)
            raise MotionError(
                ErrorSeverity.CRITICAL, ErrorCode.COLLISION, cause, f"Collision with {hit}", end
            )
        return end
