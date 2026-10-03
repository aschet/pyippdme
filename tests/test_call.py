# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for the Call/StreamCall handles the object model's methods return."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from pyippdme import IppDmeMachine, VirtualCMM
from pyippdme.exceptions import IppDmeServerError
from pyippdme.protocol.network import MemoryNetwork
from pyippdme.server.backend import CancellationToken
from pyippdme.types.csy import InMemoryCsyStore
from pyippdme.types.vec3 import Vec3


class _GatedBackend:
    """Yields the start point, waits for ``release``, then yields the end point."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def scan_line(
        self, start: Vec3, end: Vec3, normal: Vec3, step_w: float, cancel: CancellationToken
    ) -> AsyncIterator[Vec3]:
        del normal, step_w, cancel
        self.started.set()
        yield start
        await self.release.wait()
        yield end

    async def scan_circle(self, *args: object, **kwargs: object) -> AsyncIterator[Vec3]:
        del args, kwargs
        raise NotImplementedError
        yield  # pragma: no cover

    async def scan_helix(self, *args: object, **kwargs: object) -> AsyncIterator[Vec3]:
        del args, kwargs
        raise NotImplementedError
        yield  # pragma: no cover


@pytest.fixture
async def gated_backend() -> _GatedBackend:
    return _GatedBackend()


@pytest.fixture
async def machine(
    network: MemoryNetwork, gated_backend: _GatedBackend
) -> AsyncIterator[IppDmeMachine]:
    server = VirtualCMM(
        network=network,
        csy_store=InMemoryCsyStore(),
        backend=gated_backend,
    )
    port = await server.start()
    m = await IppDmeMachine.connect("x", port, network=network)
    try:
        yield m
    finally:
        await m.close()
        await server.close()


@pytest.fixture
async def started_machine(machine: IppDmeMachine) -> IppDmeMachine:
    await machine.start_session()
    return machine


async def test_awaiting_a_call_returns_the_typed_result(started_machine: IppDmeMachine) -> None:
    await started_machine.cart_cmm.go_to(x=1, y=2, z=3)
    position = await started_machine.cart_cmm.get_position()
    assert position == (1.0, 2.0, 3.0)


async def test_a_call_can_be_acknowledged_before_it_is_awaited(
    started_machine: IppDmeMachine,
) -> None:
    call = started_machine.cart_cmm.pt_meas(x=1, y=2, z=3)
    await call.acknowledged()
    measured = await call
    assert measured == {"X": 1.0, "Y": 2.0, "Z": 3.0}


async def test_a_call_is_sent_when_it_is_created_not_when_it_is_awaited(
    started_machine: IppDmeMachine,
) -> None:
    move = started_machine.cart_cmm.go_to(x=7)
    await move.acknowledged()
    assert await started_machine.cart_cmm.get_position() == (7.0, 0.0, 0.0)
    await move


async def test_a_failing_command_is_acknowledged_and_then_raises_when_awaited(
    machine: IppDmeMachine,
) -> None:
    call = machine.cart_cmm.go_to(x=1)  # no session started yet
    await call.acknowledged()
    with pytest.raises(IppDmeServerError):
        await call


async def test_acknowledged_can_be_awaited_again_and_after_completion(
    started_machine: IppDmeMachine,
) -> None:
    call = started_machine.dme.get_machine_class()
    await call.acknowledged()
    await call
    await call.acknowledged()
    assert "_VirtualCMM" in await call


async def test_calls_work_with_gather_and_wait_for(started_machine: IppDmeMachine) -> None:
    machine_class, position = await asyncio.gather(
        started_machine.dme.get_machine_class(), started_machine.cart_cmm.get_position()
    )
    assert "_VirtualCMM" in machine_class
    assert position == (0.0, 0.0, 0.0)
    assert await asyncio.wait_for(started_machine.dme.is_homed(), 1.0) is False


async def test_a_stream_is_acknowledged_while_it_is_still_running(
    started_machine: IppDmeMachine, gated_backend: _GatedBackend
) -> None:
    scan = started_machine.scanning.scan_on_line((0, 0, 0), (10, 0, 0), (0, 0, 1), 2.0)
    await scan.acknowledged()  # the scan itself, not the OnScanReport setup before it
    await asyncio.wait_for(gated_backend.started.wait(), 1.0)
    assert not gated_backend.release.is_set()

    points = []
    async for point in scan:
        points.append(point)
        if len(points) == 1:
            gated_backend.release.set()
    assert points == [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)]


async def test_stopping_a_stream_early_cancels_it(
    started_machine: IppDmeMachine, gated_backend: _GatedBackend
) -> None:
    scan = started_machine.scanning.scan_on_line((0, 0, 0), (10, 0, 0), (0, 0, 1), 2.0)
    async for _point in scan:
        break
    await asyncio.sleep(0.05)
    gated_backend.release.set()
    assert await started_machine.dme.get_machine_class()


async def test_the_raw_responses_are_available_through_the_transaction(
    started_machine: IppDmeMachine,
) -> None:
    call = started_machine.cart_cmm.pt_meas(x=1, y=2, z=3)
    transaction = await call.transaction()
    raw = await transaction.wait_complete()
    assert [item.to_wire() for item in raw] == ["X(1),Y(2),Z(3)"]
    assert await call == {"X": 1.0, "Y": 2.0, "Z": 3.0}


async def test_the_transaction_of_a_stream_gives_the_raw_responses(
    started_machine: IppDmeMachine, gated_backend: _GatedBackend
) -> None:
    gated_backend.release.set()
    scan = started_machine.scanning.scan_on_line((0, 0, 0), (10, 0, 0), (0, 0, 1), 2.0)
    transaction = await scan.transaction()
    raw = await transaction.wait_complete()
    assert [item.to_wire() for item in raw] == ["0,0,0", "10,0,0"]
