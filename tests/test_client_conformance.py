# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Client behaviour the standard prescribes: waiting for the Ack (5.4.3), one-shot events (5.5)."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from pyippdme import CommandName, IppDmeClient
from pyippdme.protocol.network import MemoryNetwork, StreamReaderLike, StreamWriterLike

Handler = Callable[[StreamReaderLike, StreamWriterLike], Awaitable[None]]


async def _client_for(network: MemoryNetwork, handler: Handler) -> IppDmeClient:
    listener = await network.start_server(handler, "x", 0)
    return await IppDmeClient.connect("x", listener.port, network=network)


async def test_the_client_waits_for_the_ack_before_it_sends_the_next_command(
    network: MemoryNetwork,
) -> None:
    """5.4.3: a client may not start a transaction before the previous one was acknowledged."""
    second_arrived_early = False
    release_ack = asyncio.Event()

    async def server(reader: StreamReaderLike, writer: StreamWriterLike) -> None:
        nonlocal second_arrived_early
        first = (await reader.readuntil(b"\r\n")).decode()
        try:
            await asyncio.wait_for(reader.readuntil(b"\r\n"), 0.1)
            second_arrived_early = True
        except TimeoutError:
            pass
        tag = first[:5]
        writer.write(f"{tag} &\r\n{tag} %\r\n".encode())
        release_ack.set()
        second = (await reader.readuntil(b"\r\n")).decode()
        writer.write(f"{second[:5]} &\r\n{second[:5]} %\r\n".encode())

    client = await _client_for(network, server)
    try:
        await asyncio.gather(
            client.call(CommandName.IS_HOMED),
            client.call(CommandName.GET_COORD_SYSTEM),
        )
    finally:
        await client.close()
    assert not second_arrived_early


async def test_prioritized_commands_do_not_wait_for_the_ack(network: MemoryNetwork) -> None:
    """5.7: commands ending in E go past the queue, so an AbortE needs no earlier Ack."""
    lines: list[str] = []
    both_seen = asyncio.Event()

    async def server(reader: StreamReaderLike, writer: StreamWriterLike) -> None:
        del writer
        for _ in range(2):
            lines.append((await reader.readuntil(b"\r\n")).decode().strip())
        both_seen.set()

    client = await _client_for(network, server)
    try:
        slow = asyncio.ensure_future(client.call(CommandName.IS_HOMED))  # never acknowledged
        await asyncio.sleep(0.05)
        urgent = asyncio.ensure_future(client.call(CommandName.ABORT_E))
        await asyncio.wait_for(both_seen.wait(), 1.0)
        slow.cancel()
        urgent.cancel()
    finally:
        await client.close()
    assert [line.split(" ", 1)[1] for line in lines] == ["IsHomed()", "AbortE()"]


async def test_a_one_shot_event_ends_with_the_transaction(network: MemoryNetwork) -> None:
    """5.5.1/5.5.2: an event sent before the transaction completes is a one-shot event."""

    async def server(reader: StreamReaderLike, writer: StreamWriterLike) -> None:
        line = (await reader.readuntil(b"\r\n")).decode()
        tag = line[:5]
        writer.write(f"{tag} &\r\n{tag} # X(1)\r\n{tag} %\r\n".encode())

    client = await _client_for(network, server)
    try:
        txn = await client.start_daemon(CommandName.GET_PROP_E)
        events = [item async for item in txn.events()]
        assert events == []  # the one data response came before Done
        assert client._pending == {}
    finally:
        await client.close()


async def test_a_multiple_shot_event_keeps_delivering_after_the_transaction(
    network: MemoryNetwork,
) -> None:
    async def server(reader: StreamReaderLike, writer: StreamWriterLike) -> None:
        line = (await reader.readuntil(b"\r\n")).decode()
        tag = line[:5]
        writer.write(f"{tag} &\r\n{tag} %\r\n{tag} # X(1)\r\n{tag} # X(2)\r\n".encode())

    client = await _client_for(network, server)
    try:
        txn = await client.start_daemon(CommandName.ON_MOVE_REPORT_E)
        seen = []
        async for item in txn.events():
            seen.append(item.to_wire())
            if len(seen) == 2:
                break
        assert seen == ["X(1)", "X(2)"]
    finally:
        await client.close()
