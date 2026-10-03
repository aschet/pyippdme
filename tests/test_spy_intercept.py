# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for Spy's intercept function: forwarding, altering, dropping and expanding lines."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence

import pytest

from pyippdme import CommandName, IppDmeClient, VirtualCMM
from pyippdme.exceptions import IppDmeConnectionError
from pyippdme.protocol.network import MemoryNetwork, StreamReaderLike, StreamWriterLike
from pyippdme.spy import Interceptor, Spy, SpyDirection, SpyMessage, SpyOutcome


async def _echo_line(reader: StreamReaderLike, writer: StreamWriterLike) -> None:
    while True:
        try:
            line = await reader.readuntil(b"\r\n")
        except (asyncio.IncompleteReadError, asyncio.LimitOverrunError):
            writer.close()
            return
        writer.write(line)
        await writer.drain()


class _Rig:
    def __init__(self, network: MemoryNetwork, spy_port: int, messages: list[SpyMessage]) -> None:
        self.network = network
        self.spy_port = spy_port
        self.messages = messages

    async def connect(self) -> tuple[StreamReaderLike, StreamWriterLike]:
        return await self.network.open_connection("x", self.spy_port)


async def _start_rig(intercept: Interceptor | None) -> tuple[_Rig, Spy, object]:
    network = MemoryNetwork()
    echo = await network.start_server(_echo_line, "x", 0)
    messages: list[SpyMessage] = []
    spy = Spy("x", echo.port, network=network, intercept=intercept, on_message=messages.append)
    spy_port = await spy.start("x", 0)
    return _Rig(network, spy_port, messages), spy, echo


@pytest.fixture
async def rig_factory() -> AsyncIterator[object]:
    started: list[tuple[Spy, object]] = []

    async def make(intercept: Interceptor | None) -> _Rig:
        rig, spy, echo = await _start_rig(intercept)
        started.append((spy, echo))
        return rig

    yield make
    for spy, echo in started:
        await spy.close()
        await echo.close()  # type: ignore[attr-defined]


async def _roundtrip(rig: _Rig, *lines: bytes, expect: int) -> list[bytes]:
    reader, writer = await rig.connect()
    for line in lines:
        writer.write(line)
    await writer.drain()
    received = [await asyncio.wait_for(reader.readuntil(b"\r\n"), 1.0) for _ in range(expect)]
    writer.close()
    return received


async def test_without_an_interceptor_lines_are_forwarded_unchanged(rig_factory: object) -> None:
    rig = await rig_factory(None)  # type: ignore[operator]
    assert await _roundtrip(rig, b"abc\r\n", expect=1) == [b"abc\r\n"]
    await asyncio.sleep(0.05)
    assert all(m.outcome is SpyOutcome.FORWARDED and m.forwarded is None for m in rig.messages)


async def test_an_interceptor_returning_the_same_line_forwards_it(rig_factory: object) -> None:
    async def keep(message: SpyMessage) -> bytes:
        return message.line

    rig = await rig_factory(keep)  # type: ignore[operator]
    assert await _roundtrip(rig, b"abc\r\n", expect=1) == [b"abc\r\n"]
    await asyncio.sleep(0.05)
    assert all(m.outcome is SpyOutcome.FORWARDED for m in rig.messages)


async def test_an_interceptor_can_alter_a_line(rig_factory: object) -> None:
    async def shout(message: SpyMessage) -> bytes | None:
        if message.direction is SpyDirection.TO_SERVER:
            return message.line.upper()
        return message.line

    rig = await rig_factory(shout)  # type: ignore[operator]
    assert await _roundtrip(rig, b"abc\r\n", expect=1) == [b"ABC\r\n"]
    await asyncio.sleep(0.05)
    altered = [m for m in rig.messages if m.outcome is SpyOutcome.ALTERED]
    assert [(m.line, m.forwarded) for m in altered] == [(b"abc\r\n", (b"ABC\r\n",))]


@pytest.mark.parametrize("dropped", [None, b"", ()])
async def test_an_interceptor_can_drop_a_line(rig_factory: object, dropped: object) -> None:
    async def drop_second(message: SpyMessage) -> bytes | Sequence[bytes] | None:
        if message.direction is SpyDirection.TO_SERVER and message.line == b"two\r\n":
            return dropped  # type: ignore[return-value]
        return message.line

    rig = await rig_factory(drop_second)  # type: ignore[operator]
    received = await _roundtrip(rig, b"one\r\n", b"two\r\n", b"three\r\n", expect=2)
    assert received == [b"one\r\n", b"three\r\n"]
    await asyncio.sleep(0.05)
    assert [m.line for m in rig.messages if m.outcome is SpyOutcome.DROPPED] == [b"two\r\n"]


async def test_an_interceptor_can_replace_a_line_with_several(rig_factory: object) -> None:
    async def twice(message: SpyMessage) -> Sequence[bytes]:
        if message.direction is SpyDirection.TO_SERVER:
            return [message.line, b"extra\r\n"]
        return [message.line]

    rig = await rig_factory(twice)  # type: ignore[operator]
    assert await _roundtrip(rig, b"abc\r\n", expect=2) == [b"abc\r\n", b"extra\r\n"]


async def test_an_interceptor_may_delay_and_order_is_preserved(rig_factory: object) -> None:
    async def slow_first(message: SpyMessage) -> bytes:
        if message.direction is SpyDirection.TO_SERVER and message.line == b"first\r\n":
            await asyncio.sleep(0.1)
        return message.line

    rig = await rig_factory(slow_first)  # type: ignore[operator]
    assert await _roundtrip(rig, b"first\r\n", b"second\r\n", expect=2) == [
        b"first\r\n",
        b"second\r\n",
    ]


async def test_an_interceptor_that_raises_closes_the_connection(rig_factory: object) -> None:
    async def broken(message: SpyMessage) -> bytes:
        raise RuntimeError(message.line)

    rig = await rig_factory(broken)  # type: ignore[operator]
    reader, writer = await rig.connect()
    writer.write(b"abc\r\n")
    await writer.drain()
    with pytest.raises(asyncio.IncompleteReadError):
        await asyncio.wait_for(reader.readuntil(b"\r\n"), 1.0)
    writer.close()


async def test_a_command_can_be_rewritten_on_its_way_to_a_real_server() -> None:
    network = MemoryNetwork()
    server = VirtualCMM(network=network)
    server_port = await server.start()

    async def rewrite(message: SpyMessage) -> bytes:
        if message.direction is SpyDirection.TO_SERVER:
            return message.line.replace(b"GetDMEVersion", b"GetMachineClass")
        return message.line

    spy = Spy("x", server_port, network=network, intercept=rewrite)
    spy_port = await spy.start("x", 0)
    try:
        client = await IppDmeClient.connect("x", spy_port, network=network)
        try:
            await client.start_session()
            (data,) = await client.call(CommandName.GET_DME_VERSION)
            assert "_VirtualCMM" in data.to_wire()
        finally:
            await client.close()
    finally:
        await spy.close()
        await server.close()


async def test_a_dropped_command_never_reaches_the_server() -> None:
    network = MemoryNetwork()
    server = VirtualCMM(network=network)
    server_port = await server.start()

    async def drop_version(message: SpyMessage) -> bytes | None:
        if b"GetDMEVersion" in message.line:
            return None
        return message.line

    spy = Spy("x", server_port, network=network, intercept=drop_version)
    spy_port = await spy.start("x", 0)
    try:
        client = await IppDmeClient.connect("x", spy_port, network=network)
        try:
            await client.start_session()
            transaction = client.send(CommandName.GET_DME_VERSION)
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(transaction.wait_ack(), 0.2)
        finally:
            await client.close()
    except IppDmeConnectionError:  # pragma: no cover - only if close races the pending call
        pass
    finally:
        await spy.close()
        await server.close()
