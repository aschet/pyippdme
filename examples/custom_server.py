#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Build your own simulated CMM from the server seams, using the planning and CSY helpers.

``VirtualCMM`` takes four optional parts; each is a small protocol you can implement:

* ``MotionModel``: what a move costs (here: the time from ``pyippdme.twin.planning``).
* ``SampleSurface``: what ``PtMeas`` hits (here: a sphere).
* ``ToolHandler``: what a tool change does (here: nothing but the answers it must give).
* ``MachineBackend``: the scans (here: the default straight paths).

Run ``python examples/custom_server.py`` and connect with ``ippdme client-gui 127.0.0.1``.
"""

from __future__ import annotations

import asyncio

from pyippdme import (
    IppDmeMachine,
    MemoryNetwork,
    MotionRequest,
    ProbeRequest,
    ProbeResult,
    SphereSurface,
    VirtualCMM,
)
from pyippdme.protocol.errors import ErrorCode, ErrorSeverity, ServerError
from pyippdme.twin.planning import travel_time
from pyippdme.types.csy import CsyContext
from pyippdme.types.vec3 import Vec3, add, norm, scale, sub

#: A 40 mm sphere to measure.
SURFACE = SphereSurface((200.0, 200.0, 50.0), 20.0)


class SlowMotion:
    """A move takes as long as the machine's speed and acceleration say."""

    def __init__(self, speed: float = 300.0, accel: float = 800.0, time_scale: float = 1.0) -> None:
        self.speed, self.accel, self.time_scale = speed, accel, time_scale
        self.travelled = 0.0

    async def travel(self, request: MotionRequest) -> Vec3:
        distance = norm(sub(request.end, request.start))
        self.travelled += distance
        seconds = travel_time(distance, self.speed, self.accel) * self.time_scale
        try:
            await asyncio.wait_for(request.cancel.wait(), timeout=seconds)
        except TimeoutError:
            return request.end
        return request.start  # aborted: stay where the move began

    async def probe(self, request: ProbeRequest) -> ProbeResult:
        """Search from the approach point towards the surface; 1006 if it is not there."""
        start = add(request.nominal, scale(request.direction, request.approach))
        hit = SURFACE.intersect(start, scale(request.direction, -1.0))
        if hit is None or norm(sub(hit, start)) > request.approach + request.search:
            raise ServerError(
                ErrorSeverity.ERROR, ErrorCode.SURFACE_NOT_FOUND, request.cause, "No surface found"
            )
        return ProbeResult(hit, add(hit, scale(request.direction, max(request.retract, 0.0))))

    def csy_context(self) -> CsyContext | None:
        return None  # the commands' coordinate systems are all there is

    async def home(self, cancel: asyncio.Event) -> Vec3 | None:
        return None


def build(network: MemoryNetwork | None = None, time_scale: float = 0.0) -> VirtualCMM:
    """Return a VirtualCMM with a timed motion model and a 40 mm sphere to measure."""
    kwargs = {"network": network} if network is not None else {}
    return VirtualCMM(sample_surface=SURFACE, motion=SlowMotion(time_scale=time_scale), **kwargs)


async def main() -> None:
    network = MemoryNetwork()
    server = build(network)
    port = await server.start("127.0.0.1", 0)
    machine = await IppDmeMachine.connect("127.0.0.1", port, network=network)
    await machine.start_session()
    await machine.dme.home()
    await machine.mover.enable_user()
    hit = await machine.cart_cmm.pt_meas(x=200, y=200, z=71, ijk=(0, 0, 1))
    print("sphere top at z =", round(hit.number("Z"), 3))
    await machine.close()
    await server.close()


if __name__ == "__main__":
    asyncio.run(main())
