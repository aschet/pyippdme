# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""The pluggable interface between command handlers and real hardware.

Handlers for commands that just report or mutate protocol-level state (the
active coordinate system, a property store, session/error bookkeeping) work
directly against :class:`~pyippdme.server.registry.MachineState` and need no
backend. Handlers for commands that move or measure a *real* machine
delegate that part to a :class:`MachineBackend`, so a server integrator can
swap in real hardware control without touching argument parsing, validation,
or response encoding: pass your own object satisfying this Protocol (no
inheritance required - it's structural) to ``IppDmeServer(backend=...)``.
This module holds only the interface itself - the concrete, simulated
implementation (:class:`~pyippdme.simulation.backend.SimulatedBackend`) lives
in :mod:`pyippdme.simulation`, which depends on this module, not the other
way around.

Currently only the ``Scanning`` class subset
(:mod:`pyippdme.simulation.classes.scanning_class`) goes through a backend; other
classes still hold their (simpler, instantaneous) state directly on
``MachineState``. The same pattern extends to them the same way, by adding
a method here and delegating to it from the handler. Argument validation
(degenerate geometry, orthogonality, ...) stays in the handler, which runs
before the backend is invoked at all: a backend only ever sees geometry
already known to be well-formed.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Protocol, TypeAlias

from pyippdme.types.vec3 import Vec3

#: Signalled (via ``.set()``) when a running operation must stop as soon as
#: possible, e.g. because the client sent ``AbortE()``. A backend method
#: should check it cooperatively between physical steps (and, if it wraps a
#: blocking SDK in an executor thread, pass it - or an equivalent signal -
#: into that thread) rather than relying solely on ``asyncio.Task`` cancel:
#: cancelling the *task* only stops the coroutine from being awaited, it does
#: not by itself stop a blocking call already running in another thread.
CancellationToken: TypeAlias = asyncio.Event


class MachineBackend(Protocol):
    """Physical operations behind the ``Scanning`` command subset.

    Every method is an async generator: implementations should ``yield``
    each point as soon as it is actually acquired (matching 6.13.2's
    "already measured data may be transmitted... while execution...
    is still in progress"), and return early - after any hardware-specific
    cleanup, e.g. stopping motion - once ``cancel`` is set.
    """

    def scan_line(
        self, start: Vec3, end: Vec3, normal: Vec3, step_w: float, cancel: CancellationToken
    ) -> AsyncIterator[Vec3]: ...

    def scan_circle(
        self,
        center: Vec3,
        start: Vec3,
        normal: Vec3,
        delta_deg: float,
        step_w_deg: float,
        cancel: CancellationToken,
    ) -> AsyncIterator[Vec3]: ...

    def scan_helix(
        self,
        center: Vec3,
        start: Vec3,
        normal: Vec3,
        delta_deg: float,
        step_w_deg: float,
        pitch: float,
        cancel: CancellationToken,
    ) -> AsyncIterator[Vec3]: ...

    # Optional: a backend that can probe the real surface of a part also implements
    #
    #     def scan_contour(self, scan: ContourScan, cancel: CancellationToken) -> AsyncIterator[Vec3]: ...
    #
    # for the five unknown-contour scans (6.13.2.2, Figures 35-39); see
    # :mod:`pyippdme.server.contour` for the request and :func:`~pyippdme.server.contour.trace_contour`
    # for an algorithm that follows a surface given a function that probes it. Without it
    # those scans fall back to a straight line towards their stop element.
