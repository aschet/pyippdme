# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Offline programming: turn points on a CAD part into a collision-free measuring program.

The program is a list of I++ DME command lines (what ``ippdme client --file`` runs): moves between
approach points along paths the digital twin found collision free, and ``PtMeas`` at each point.
Run it with a client, or simulate it with :func:`run_program`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from pyippdme.client import IppDmeClient, recipes
from pyippdme.client.interaction import ConnectionLost, Failed, ParseFailed, run_command_line
from pyippdme.protocol.network import TCP_NETWORK, Network
from pyippdme.twin.twin import DigitalTwin
from pyippdme.types.vec3 import Vec3, add, scale

__all__ = ["run_program", "touch_program"]


def touch_program(
    twin: DigitalTwin,
    points: Sequence[tuple[Vec3, Vec3]],
    *,
    clearance: float = 10.0,
    start: Vec3 | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> list[str]:
    """Generate the program that probes ``points``: ``(position, surface normal)``, machine mm.

    For each point the machine goes to the approach point ``clearance`` above it along the normal
    by a collision-free path, then probes. ``PtMeasPar.Retract`` is set to -1 so that the machine
    returns to the approach point after each touch. A point that cannot be reached collision free
    raises ``ValueError`` naming it.
    """
    lines = ["Home()", "EnableUser()", recipes.speed_line("PtMeasPar", "Retract", -1.0)]
    here = start or (0.0, 0.0, twin.machine.spec.travel[2])
    for index, (position, normal) in enumerate(points, 1):
        approach = add(position, scale(normal, clearance))
        path = twin.plan_collision_free(approach, here)
        if path is None:
            raise ValueError(f"point {index} at {position} cannot be reached collision free")
        for waypoint in path[1:]:
            lines.append(recipes.goto_line(*waypoint))
        lines.append(recipes.pt_meas_line(position, normal))
        here = approach
        if progress is not None:
            progress(index, len(points))
    return lines


async def run_program(
    host: str,
    port: int,
    lines: Sequence[str],
    *,
    on_line: Callable[[str, bool], None] | None = None,
    network: Network = TCP_NETWORK,
) -> bool:
    """Run command lines through a new client connected to a server; True if all succeeded.

    ``on_line(text, ok)`` is called after each line. The run stops at the first failure.
    ``StartSession()`` and ``EndSession()`` are sent around the program.
    """
    client = await IppDmeClient.connect(host, port, network=network)
    try:
        await client.start_session()
        for text in lines:
            ok = True
            async for event in run_command_line(client, text):
                if isinstance(event, ParseFailed | Failed | ConnectionLost):
                    ok = False
            if on_line is not None:
                on_line(text, ok)
            if not ok:
                return False
        await client.end_session()
        return True
    finally:
        await client.close()
