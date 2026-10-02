# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for the MachineBackend seam and the prioritized-execution/AbortE model.

Uses a small controllable fake backend (rather than real delays) so the
"AbortE interrupts an in-progress scan without waiting for it" behaviour can
be tested deterministically and fast, via explicit asyncio.Events instead of
timing.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from pyippdme import IppDmeClient, VirtualCMM
from pyippdme.exceptions import IppDmeServerError
from pyippdme.protocol.ast import EventTag, Number, NumericData
from pyippdme.protocol.commands import CommandName
from pyippdme.server.backend import CancellationToken
from pyippdme.types.vec3 import Vec3


class _HangingBackend:
    """Yields one point, then blocks until released or cancelled."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.released = asyncio.Event()
        self.was_cancelled = False

    async def scan_line(
        self, start: Vec3, end: Vec3, normal: Vec3, step_w: float, cancel: CancellationToken
    ) -> AsyncIterator[Vec3]:
        del end, normal, step_w
        self.started.set()
        yield start
        try:
            await self.released.wait()
        except asyncio.CancelledError:
            self.was_cancelled = True
            raise
        if cancel.is_set():
            return
        yield start

    async def scan_circle(
        self,
        center: Vec3,
        start: Vec3,
        normal: Vec3,
        delta_deg: float,
        step_w_deg: float,
        cancel: CancellationToken,
    ) -> AsyncIterator[Vec3]:
        del center, normal, delta_deg, step_w_deg, cancel
        yield start


class _RecordingBackend:
    """A trivial backend proving the pattern: swap geometry without touching command code."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def scan_line(
        self, start: Vec3, end: Vec3, normal: Vec3, step_w: float, cancel: CancellationToken
    ) -> AsyncIterator[Vec3]:
        del normal, step_w, cancel
        self.calls.append("scan_line")
        yield start
        yield end

    async def scan_circle(
        self,
        center: Vec3,
        start: Vec3,
        normal: Vec3,
        delta_deg: float,
        step_w_deg: float,
        cancel: CancellationToken,
    ) -> AsyncIterator[Vec3]:
        del center, normal, delta_deg, step_w_deg, cancel
        self.calls.append("scan_circle")
        yield start


async def _connected(backend: object) -> tuple[VirtualCMM, IppDmeClient]:
    server = VirtualCMM(backend=backend)  # type: ignore[arg-type]
    port = await server.start("127.0.0.1", 0)
    client = await IppDmeClient.connect("127.0.0.1", port)
    await client.start_session()
    return server, client


async def _teardown(client: IppDmeClient, server: VirtualCMM) -> None:
    await client.close()
    await server.close()


async def test_custom_backend_is_used_without_redefining_commands() -> None:
    backend = _RecordingBackend()
    server, client = await _connected(backend)
    try:
        data = await client.call(
            CommandName.SCAN_ON_LINE, *(Number.of(v) for v in (0, 0, 0, 10, 0, 0, 0, 0, 1, 5))
        )
        assert backend.calls == ["scan_line"]
        assert (
            len(data) == 2
        )  # exactly what _RecordingBackend.scan_line yields, unmodified by the handler shape
    finally:
        await _teardown(client, server)


async def test_abort_e_interrupts_a_running_scan_without_waiting() -> None:
    backend = _HangingBackend()
    server, client = await _connected(backend)
    try:
        scan_txn = client.send(
            CommandName.SCAN_ON_LINE, *(Number.of(v) for v in (0, 0, 0, 10, 0, 0, 0, 0, 1, 1))
        )
        await scan_txn.wait_ack()
        await asyncio.wait_for(backend.started.wait(), timeout=1.0)

        abort_txn = client.send(CommandName.ABORT_E, tag=EventTag.of(1))
        await asyncio.wait_for(abort_txn.wait_ack(), timeout=1.0)
        await asyncio.wait_for(abort_txn.wait_complete(), timeout=1.0)

        # AbortE must not have waited for `released` - prove the backend saw cancellation.
        assert backend.was_cancelled is True

        with pytest.raises(IppDmeServerError) as excinfo:
            await scan_txn.wait_complete()
        assert excinfo.value.error.number == "0006"
    finally:
        backend.released.set()  # unblock in case anything is still awaiting it
        await _teardown(client, server)


async def test_abort_e_rejects_queued_commands_too() -> None:
    backend = _HangingBackend()
    server, client = await _connected(backend)
    try:
        scan_txn = client.send(
            CommandName.SCAN_ON_LINE, *(Number.of(v) for v in (0, 0, 0, 10, 0, 0, 0, 0, 1, 1))
        )
        await scan_txn.wait_ack()
        await asyncio.wait_for(backend.started.wait(), timeout=1.0)

        # A normal command sent while the scan is running goes on the standard
        # queue behind it and must not start (VDMA 8722 6.3.1 Table 12).
        queued_txn = client.send(CommandName.GET_DME_VERSION)
        await asyncio.wait_for(queued_txn.wait_ack(), timeout=1.0)

        abort_txn = client.send(CommandName.ABORT_E, tag=EventTag.of(1))
        await asyncio.wait_for(abort_txn.wait_complete(), timeout=1.0)

        with pytest.raises(IppDmeServerError) as excinfo:
            await asyncio.wait_for(queued_txn.wait_complete(), timeout=1.0)
        assert excinfo.value.error.number == "0006"
    finally:
        backend.released.set()
        await _teardown(client, server)


async def test_error_state_after_abort_e_requires_clear_all_errors() -> None:
    backend = _HangingBackend()
    server, client = await _connected(backend)
    try:
        scan_txn = client.send(
            CommandName.SCAN_ON_LINE, *(Number.of(v) for v in (0, 0, 0, 10, 0, 0, 0, 0, 1, 1))
        )
        await scan_txn.wait_ack()
        await asyncio.wait_for(backend.started.wait(), timeout=1.0)
        abort_txn = client.send(CommandName.ABORT_E, tag=EventTag.of(1))
        await asyncio.wait_for(abort_txn.wait_complete(), timeout=1.0)
        with pytest.raises(IppDmeServerError):
            await scan_txn.wait_complete()

        with pytest.raises(IppDmeServerError) as excinfo:
            await client.call(CommandName.GET_DME_VERSION)
        assert excinfo.value.error.number == "0514"

        await client.clear_all_errors()
        await client.call(CommandName.GET_DME_VERSION)  # no longer raises
    finally:
        backend.released.set()
        await _teardown(client, server)


async def test_prioritized_commands_run_concurrently_with_a_stuck_queued_command() -> None:
    backend = _HangingBackend()
    server, client = await _connected(backend)
    try:
        scan_txn = client.send(
            CommandName.SCAN_ON_LINE, *(Number.of(v) for v in (0, 0, 0, 10, 0, 0, 0, 0, 1, 1))
        )
        await scan_txn.wait_ack()
        await asyncio.wait_for(backend.started.wait(), timeout=1.0)

        # GetErrStatusE ends in "E": it must complete promptly even though the
        # (non-"E") scan ahead of it in the standard queue is still stuck.
        (data,) = await asyncio.wait_for(client.call(CommandName.GET_ERR_STATUS_E), timeout=1.0)
        assert isinstance(data, NumericData)
    finally:
        backend.released.set()
        await _teardown(client, server)
