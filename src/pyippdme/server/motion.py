# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""The pluggable interface between a move command and the machine that carries it out.

Without a motion model a move (``GoTo``, ``Step``, ``PtMeas``, ...) simply
sets the new position, instantly and without any limits - what the minimal
:class:`~pyippdme.simulation.virtual_cmm.VirtualCMM` does. A
:class:`MotionModel` makes a move take time, keep within the machine volume,
stop at a collision, or wait for ``Home()``: the move handlers ``await``
:meth:`MotionModel.travel` before they update the position, and a
:class:`~pyippdme.protocol.errors.ServerError` raised by it is the command's
error response. Like :class:`~pyippdme.server.surface.SampleSurface` this
module holds only the interface; the concrete models live in the packages
that depend on it (see :mod:`pyippdme.twin`).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from pyippdme.protocol.errors import ServerError
from pyippdme.types.csy import CsyContext
from pyippdme.types.obb import Obb
from pyippdme.types.vec3 import Vec3


@dataclass(frozen=True, slots=True)
class MotionRequest:
    """One straight move of the tool centre point (and optionally the rotary table)."""

    #: The command that moves, e.g. ``GoTo`` or ``PtMeas``.
    cause: str
    start: Vec3
    end: Vec3
    rotary_start: float
    rotary_end: float
    homed: bool
    #: Name of the active tool, a key of the tool catalog.
    tool_name: str
    #: Set when the move must stop as soon as possible (``AbortE()``).
    cancel: asyncio.Event


@dataclass(frozen=True, slots=True)
class ProbeRequest:
    """The probing cycle of ``PtMeas`` (6.12.1) from the approach position.

    The machine stands at ``nominal + direction * approach``; ``direction`` points from the
    surface towards the probe. ``approach`` is ``Part.Approach + Tool.Approach + AvrRadius``
    (6.12.1): for a tool with a ball the tool centre point starts one radius further out. The
    search goes on until ``search`` beyond the nominal point, measured at the tip (so a ball
    centre travels ``approach - radius + search``), and the machine retracts by ``retract``
    (negative: back to the approach position) after the trigger.
    """

    cause: str
    nominal: Vec3
    direction: Vec3
    approach: float
    search: float
    retract: float
    #: Speed and acceleration of the search (``PtMeasPar``).
    speed: float
    accel: float
    tool_name: str
    cancel: asyncio.Event


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """What the cycle found: the measured surface point and where the machine rests."""

    #: The measured point on the surface (compensated for the tip radius).
    contact: Vec3
    #: Tool centre point after the retract.
    rest: Vec3


@dataclass(frozen=True, slots=True)
class MotionError(ServerError):
    """A move that ended early: the machine stands at :attr:`stopped_at`, not at the target."""

    stopped_at: Vec3


@runtime_checkable
class ToolHandler(Protocol):
    """The physical side of changing and qualifying tools.

    Without one, ``ChangeTool`` and ``ReQualify`` take no time and always
    succeed. A handler can animate the trip to the tool rack, refuse while the
    machine is not ready, or find the reference sphere for ``ReQualify``;
    raise a :class:`~pyippdme.protocol.errors.ServerError` to fail the command.
    """

    async def change_tool(
        self, current: str, target: str, position: Vec3, cancel: asyncio.Event
    ) -> None: ...

    async def requalify(self, tool_name: str, cancel: asyncio.Event) -> None: ...

    def alignment_volume(self, tool_name: str) -> tuple[Vec3, float] | None:
        """Return the sphere the tool occupies while aligned (``Tool.AlignmentVolume``, Fig. 52/53).

        ``(centre, radius)``: the vector from the tool's reference point to the sphere centre,
        in machine coordinates, and the radius. ``None`` if the tool has no such volume.
        """
        ...

    def collision_volume(self, tool_name: str) -> list[Obb] | None:
        """Oriented bounding boxes covering the tool (``Tool.CollisionVolume``, Figures 49-51).

        Centres relative to the tool's reference point and axes, in machine coordinates.
        ``None`` if the tool has no such volume.
        """
        ...


@runtime_checkable
class CalibrationAware(Protocol):
    """Optional part of a :class:`ToolHandler`: which tools are calibrated.

    ``False`` makes ``Tool.Alignment`` and ``Tool.AvrOffsets`` fail with ``2000``
    "Tool not calibrated" (Tables 114 and 116). Without it every tool counts as calibrated.
    """

    def is_calibrated(self, tool_name: str) -> bool: ...


@runtime_checkable
class OffsetProvider(Protocol):
    """Optional part of a :class:`ToolHandler`: the average tool offsets (Table 116).

    Returns them in machine coordinates, or ``None`` to answer with zeros.
    """

    def avr_offsets(self, tool_name: str) -> Vec3 | None: ...


@runtime_checkable
class RadiusProvider(Protocol):
    """Optional part of a :class:`ToolHandler`: the average tip radius (``AvrRadius``, Table 106).

    The approach position of ``PtMeas`` lies ``Part.Approach + Tool.Approach + AvrRadius`` from the
    nominal point (6.12.1), so a tool with a ball reports its radius here. Without it the radius
    is zero, as for optical tools.
    """

    def avr_radius(self, tool_name: str) -> float: ...


@runtime_checkable
class AlignmentSolver(Protocol):
    """Optional part of a :class:`ToolHandler`: the orientation a tool really reaches.

    ``AlignTool`` and the ``Tool.A``/``Tool.B``/``Tool.Alignment`` arguments of ``GoTo`` ask it.
    It returns the primary (and secondary) direction the head ends up in, which is not exactly
    the requested one for a head that indexes in steps, and raises ``2505`` for an angle the head
    cannot reach. Without it every orientation is reached exactly.
    """

    def reach(
        self, tool_name: str, primary: Vec3, secondary: Vec3 | None
    ) -> tuple[Vec3, Vec3 | None]: ...


@runtime_checkable
class TemperatureProvider(Protocol):
    """Optional part of a :class:`MotionModel`: the temperatures the machine really has.

    The temperature sensors of 6.5.2 (``ReadTemperatureSensor``, ``ReadAllTemperatures``) and
    ``UpdateScaleTemperatures`` read them. Without it the sensors answer from what the client set.
    """

    def part_temperature(self) -> float: ...

    def ambient_temperature(self) -> float: ...

    def scale_temperature(self, axis: str) -> float: ...


@runtime_checkable
class MotionModel(Protocol):
    """Carries out a move; returns the position the tool centre point ends at."""

    async def travel(self, request: MotionRequest) -> Vec3: ...

    async def probe(self, request: ProbeRequest) -> ProbeResult:
        """Search for the surface from the approach position; raise ``1006`` if none is found."""
        ...

    def csy_context(self) -> CsyContext | None:
        """Return the coordinate system chain as this machine has it, or ``None`` for the default.

        A model that knows more than the commands set (the angle of its rotary table for
        ``RotaryTableVarCsy``) returns the complete context; the position a client reads is
        kept consistent through it when the client changes coordinate system.
        """
        ...

    async def home(self, cancel: asyncio.Event) -> Vec3 | None:
        """Drive to the reference position for ``Home()``; raise a ``ServerError`` if it cannot.

        Returns where the machine stands afterwards in the coordinates a client sees (the
        active coordinate system), or ``None`` for the machine's own home position.
        """
        ...
