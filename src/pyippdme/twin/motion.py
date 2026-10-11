# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Timed motion with machine limits, safety and collisions for the twin."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import TYPE_CHECKING

from pyippdme.protocol.errors import ErrorCode, ErrorSeverity, ServerError
from pyippdme.server.motion import MotionError, MotionRequest, ProbeRequest, ProbeResult
from pyippdme.twin.planning import travel_time, travelled  # noqa: F401 (re-exported)
from pyippdme.types.csy import CsyContext
from pyippdme.types.vec3 import Vec3

if TYPE_CHECKING:
    from pyippdme.twin.twin import DigitalTwin

_EPS = 1e-6
_PROBING = ("PtMeas", "PtMeasSelfCenter", "PtMeasSelfCenterLocked")


class TwinMotion:
    """A :class:`~pyippdme.server.motion.MotionModel` driven by the twin's machine and scene."""

    def __init__(self, twin: DigitalTwin) -> None:
        self._twin = twin

    def csy_context(self) -> CsyContext:
        return self._twin.csy_context()

    def part_temperature(self) -> float:
        """Return the temperature of the part as its sensor would read it."""
        return self._twin.temperature

    def ambient_temperature(self) -> float:
        """Return the temperature of the room."""
        return self._twin.ambient_temperature

    def scale_temperature(self, axis: str) -> float:
        """Return the temperature of the scale of ``axis``; the scales follow the room."""
        del axis
        return self._twin.ambient_temperature

    async def home(self, cancel: asyncio.Event) -> Vec3:
        await self._twin.run_home(cancel)
        return self._twin.to_client((0.0, 0.0, 0.0))

    async def travel(self, request: MotionRequest) -> Vec3:
        """Carry out a move given in the client's coordinate system; return where it ended."""
        twin = self._twin
        context = twin.csy_context()
        machine = replace(
            request,
            start=twin.to_machine(request.start, context),
            end=twin.to_machine(request.end, context),
        )
        try:
            return twin.to_client(await self._travel(machine), context)
        except MotionError as error:
            raise twin.client_error(error, context) from None

    async def probe(self, request: ProbeRequest) -> ProbeResult:
        """Run the probing cycle for a request in the client's coordinate system."""
        twin = self._twin
        context = twin.csy_context()
        state = twin.state
        start = twin.to_machine(state.cart_cmm.position, context) if state else twin.position
        machine = replace(
            request,
            nominal=twin.to_machine(request.nominal, context),
            direction=twin.to_machine_direction(request.direction, context),
        )
        try:
            result = await twin.run_probe(machine, start)
        except MotionError as error:
            raise twin.client_error(error, context) from None
        return ProbeResult(
            twin.to_client(result.contact, context), twin.to_client(result.rest, context)
        )

    async def _travel(self, request: MotionRequest) -> Vec3:
        twin = self._twin
        spec = twin.machine.spec
        cause = request.cause
        twin.check_ready(cause)
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
        stop_at, hit = twin.first_collision(request, probing=False)
        end = stop_at if hit is not None else request.end
        rotary_end = request.rotary_end if hit is None else request.rotary_start

        params = twin.protocol_parameters()
        speed = max(params["go_speed"] * twin.speed_override, 1e-3)
        exit_speed = 0.0
        if cause in _PROBING and hit is None:
            # Figure 29: the machine slows down to the probing speed at the approach point and
            # carries on, it does not stop there.
            tool = twin.toolkit.spec(request.tool_name)
            exit_speed = min(tool.touch_speed or params["probe_speed"], speed)
        position = await twin.run_leg(
            request.start,
            end,
            speed,
            params["go_accel"],
            request.cancel,
            (request.rotary_start, rotary_end),
            cause,
            v_end=exit_speed,
        )
        if request.cancel.is_set():
            return position
        if hit is not None:
            if twin.last_hit_was_touch:
                code, text = twin.report_touch(hit, end)
                raise MotionError(ErrorSeverity.ERROR, ErrorCode(code), cause, text, end)
            twin.report_collision(hit, end)
            raise MotionError(
                ErrorSeverity.CRITICAL, ErrorCode.COLLISION, cause, f"Collision with {hit}", end
            )
        return position
