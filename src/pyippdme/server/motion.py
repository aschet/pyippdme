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
class MotionError(ServerError):
    """A move that ended early: the machine stands at :attr:`stopped_at`, not at the target."""

    stopped_at: Vec3


@runtime_checkable
class MotionModel(Protocol):
    """Carries out a move; returns the position the tool centre point ends at."""

    async def travel(self, request: MotionRequest) -> Vec3: ...
