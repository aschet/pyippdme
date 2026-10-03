# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for pyippdme.protocol.network: the TCP and in-memory networks."""

from __future__ import annotations

import asyncio
import errno

import pytest

from pyippdme import CommandName, IppDmeClient, IppDmeMachine, VirtualCMM
from pyippdme.exceptions import IppDmeConnectionError
from pyippdme.protocol.network import (
    TCP_NETWORK,
    MemoryNetwork,
    Network,
    StreamReaderLike,
    StreamWriterLike,
)
from pyippdme.spy import Spy


async def _echo_once(reader: StreamReaderLike, writer: StreamWriterLike) -> None:
    line = await reader.readuntil(b"\n")
    writer.write(line.upper())
    await writer.drain()
    writer.close()


@pytest.fixture(params=["memory", "tcp"])
def any_network(request: pytest.FixtureRequest) -> Network:
    return MemoryNetwork() if request.param == "memory" else TCP_NETWORK


async def test_echo_round_trip_on_every_network(any_network: Network) -> None:
    listener = await any_network.start_server(_echo_once, "127.0.0.1", 0)
    try:
        reader, writer = await any_network.open_connection("127.0.0.1", listener.port)
        writer.write(b"hello\n")
        await writer.drain()
        assert await reader.readuntil(b"\n") == b"HELLO\n"
        assert await reader.read() == b""
        writer.close()
        await writer.wait_closed()
    finally:
        await listener.close()


async def test_connecting_to_nothing_is_refused_on_every_network(any_network: Network) -> None:
    listener = await any_network.start_server(_echo_once, "127.0.0.1", 0)
    port = listener.port
    await listener.close()
    with pytest.raises(OSError):  # noqa: PT011 (the exact subclass differs per platform)
        await any_network.open_connection("127.0.0.1", port)


async def test_binding_a_used_port_fails_on_every_network(any_network: Network) -> None:
    listener = await any_network.start_server(_echo_once, "127.0.0.1", 0)
    try:
        with pytest.raises(OSError) as excinfo:  # noqa: PT011
            await any_network.start_server(_echo_once, "127.0.0.1", listener.port)
        assert excinfo.value.errno == errno.EADDRINUSE
    finally:
        await listener.close()


async def test_memory_network_assigns_distinct_ports() -> None:
    network = MemoryNetwork()
    first = await network.start_server(_echo_once, "x", 0)
    second = await network.start_server(_echo_once, "x", 0)
    assert first.port != second.port


async def test_memory_network_ignores_the_host() -> None:
    network = MemoryNetwork()
    listener = await network.start_server(_echo_once, "0.0.0.0", 1294)
    reader, writer = await network.open_connection("anything", 1294)
    writer.write(b"x\n")
    assert await reader.readuntil(b"\n") == b"X\n"
    writer.close()
    await listener.close()


async def test_memory_networks_are_isolated_from_each_other() -> None:
    one, other = MemoryNetwork(), MemoryNetwork()
    listener = await one.start_server(_echo_once, "x", 1294)
    try:
        with pytest.raises(ConnectionRefusedError):
            await other.open_connection("x", 1294)
    finally:
        await listener.close()


async def test_memory_closing_a_listener_frees_its_port() -> None:
    network = MemoryNetwork()
    listener = await network.start_server(_echo_once, "x", 1294)
    await listener.close()
    await listener.close()  # closing twice is harmless
    again = await network.start_server(_echo_once, "x", 1294)
    await again.close()


async def test_memory_serve_forever_stops_when_cancelled() -> None:
    network = MemoryNetwork()
    listener = await network.start_server(_echo_once, "x", 1294)
    task = asyncio.create_task(listener.serve_forever())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    with pytest.raises(ConnectionRefusedError):
        await network.open_connection("x", 1294)


async def test_memory_peername_is_reported_on_both_sides() -> None:
    network = MemoryNetwork()
    seen: list[object] = []

    async def handler(reader: StreamReaderLike, writer: StreamWriterLike) -> None:
        del reader
        seen.append(writer.get_extra_info("peername"))
        writer.close()

    listener = await network.start_server(handler, "x", 1294)
    _reader, writer = await network.open_connection("somewhere", 1294)
    assert writer.get_extra_info("peername") == ("somewhere", 1294)
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert seen
    assert seen[0] is not None
    assert writer.get_extra_info("no-such-key", "fallback") == "fallback"
    writer.close()
    await listener.close()


async def test_memory_writes_after_the_peer_closed_are_dropped_quietly() -> None:
    network = MemoryNetwork()
    closed = asyncio.Event()

    async def handler(reader: StreamReaderLike, writer: StreamWriterLike) -> None:
        del reader
        writer.close()
        closed.set()

    listener = await network.start_server(handler, "x", 1294)
    _reader, writer = await network.open_connection("x", 1294)
    await closed.wait()
    writer.write(b"ignored\n")
    await writer.drain()
    writer.close()
    await listener.close()


async def test_memory_drain_after_own_close_raises() -> None:
    network = MemoryNetwork()
    listener = await network.start_server(_echo_once, "x", 1294)
    _reader, writer = await network.open_connection("x", 1294)
    writer.close()
    with pytest.raises(ConnectionResetError):
        await writer.drain()
    await listener.close()


async def test_memory_reader_enforces_the_buffer_limit() -> None:
    network = MemoryNetwork()

    async def handler(reader: StreamReaderLike, writer: StreamWriterLike) -> None:
        del reader
        writer.write(b"x" * 100)
        await writer.drain()
        writer.close()

    listener = await network.start_server(handler, "x", 1294)
    reader, writer = await network.open_connection("x", 1294, limit=10)
    with pytest.raises(asyncio.LimitOverrunError):
        await reader.readuntil(b"\r\n")
    writer.close()
    await listener.close()


async def test_client_reports_a_refused_memory_connection_as_a_connection_error() -> None:
    with pytest.raises(IppDmeConnectionError):
        await IppDmeClient.connect("x", 1294, network=MemoryNetwork())


async def test_server_defaults_to_tcp() -> None:
    assert VirtualCMM().network is TCP_NETWORK


async def test_client_and_server_talk_over_a_memory_network_without_a_tcp_port() -> None:
    network = MemoryNetwork()
    server = VirtualCMM(network=network)
    port = await server.start()
    try:
        machine = await IppDmeMachine.connect("virtual", port, network=network)
        try:
            await machine.start_session()
            (data,) = await machine.client.call(CommandName.GET_MACHINE_CLASS)
            assert "_VirtualCMM" in data.to_wire()
        finally:
            await machine.close()
    finally:
        await server.close()
    assert server.port is None


async def test_spy_relays_between_client_and_server_on_a_memory_network() -> None:
    network = MemoryNetwork()
    server = VirtualCMM(network=network)
    server_port = await server.start()
    messages: list[bytes] = []
    spy = Spy("virtual", server_port, network=network, on_message=lambda m: messages.append(m.line))
    spy_port = await spy.start("virtual", 0)
    try:
        client = await IppDmeClient.connect("virtual", spy_port, network=network)
        try:
            await client.start_session()
            await client.call(CommandName.GET_DME_VERSION)
        finally:
            await client.close()
        await asyncio.sleep(0.05)
    finally:
        await spy.close()
        await server.close()
    assert any(b"GetDMEVersion" in line for line in messages)
