# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for pyippdme.spy: the transparent relay behind ``ippdme spy``."""

from __future__ import annotations

import asyncio

import pytest

from pyippdme import CommandName, IppDmeClient, VirtualCMM
from pyippdme.exceptions import IppDmeConnectionError
from pyippdme.spy import Spy, SpyDirection, SpyMessage


async def _start_server() -> tuple[VirtualCMM, int]:
    server = VirtualCMM()
    port = await server.start("127.0.0.1", 0)
    return server, port


async def _start_spy(target_port: int, **hooks: object) -> tuple[Spy, int]:
    spy = Spy("127.0.0.1", target_port, **hooks)  # type: ignore[arg-type]
    port = await spy.start("127.0.0.1", 0)
    return spy, port


async def test_relays_a_full_session_unchanged() -> None:
    server, server_port = await _start_server()
    spy, spy_port = await _start_spy(server_port)
    try:
        client = await IppDmeClient.connect("127.0.0.1", spy_port)
        try:
            await client.start_session()
            (version,) = await client.call(CommandName.GET_DME_VERSION)
            assert version is not None
        finally:
            await client.close()
    finally:
        await spy.close()
        await server.close()


async def test_on_message_reports_both_directions_verbatim() -> None:
    server, server_port = await _start_server()
    messages: list[SpyMessage] = []
    spy, spy_port = await _start_spy(server_port, on_message=messages.append)
    try:
        client = await IppDmeClient.connect("127.0.0.1", spy_port)
        try:
            await client.start_session()
        finally:
            await client.close()
        await asyncio.sleep(0.05)  # let the last relayed bytes/hook calls land
    finally:
        await spy.close()
        await server.close()

    to_server = [m for m in messages if m.direction == SpyDirection.TO_SERVER]
    to_client = [m for m in messages if m.direction == SpyDirection.TO_CLIENT]
    assert to_server == [
        SpyMessage(1, SpyDirection.TO_SERVER, m.timestamp, b"00001 StartSession()\r\n")
        for m in to_server
    ]
    assert any(m.line == b"00001 %\r\n" for m in to_client)


async def test_on_connect_and_on_disconnect_fire_with_connection_id_and_peer() -> None:
    server, server_port = await _start_server()
    events: list[tuple[str, int, str]] = []
    spy, spy_port = await _start_spy(
        server_port,
        on_connect=lambda cid, peer: events.append(("connect", cid, peer)),
        on_disconnect=lambda cid, peer: events.append(("disconnect", cid, peer)),
    )
    try:
        client = await IppDmeClient.connect("127.0.0.1", spy_port)
        await client.close()
        await asyncio.sleep(0.05)
    finally:
        await spy.close()
        await server.close()

    assert len(events) == 2
    assert events[0][0] == "connect"
    assert events[0][1] == 1
    assert events[1] == ("disconnect", 1, events[0][2])


async def test_each_connection_gets_a_distinct_connection_id() -> None:
    server, server_port = await _start_server()
    connect_ids: list[int] = []
    spy, spy_port = await _start_spy(
        server_port, on_connect=lambda cid, _peer: connect_ids.append(cid)
    )
    try:
        first = await IppDmeClient.connect("127.0.0.1", spy_port)
        second = await IppDmeClient.connect("127.0.0.1", spy_port)
        await asyncio.sleep(0.05)
        await first.close()
        await second.close()
    finally:
        await spy.close()
        await server.close()

    assert sorted(connect_ids) == [1, 2]


async def test_a_raising_hook_does_not_break_the_relay() -> None:
    server, server_port = await _start_server()

    def _bad_hook(_message: SpyMessage) -> None:
        raise RuntimeError("boom")

    spy, spy_port = await _start_spy(server_port, on_message=_bad_hook)
    try:
        client = await IppDmeClient.connect("127.0.0.1", spy_port)
        try:
            await client.start_session()
            (version,) = await client.call(CommandName.GET_DME_VERSION)
            assert version is not None
        finally:
            await client.close()
    finally:
        await spy.close()
        await server.close()


async def test_port_property_before_and_after_start() -> None:
    spy = Spy("127.0.0.1", 1)
    assert spy.port is None
    bound = await spy.start("127.0.0.1", 0)
    try:
        assert spy.port == bound
    finally:
        await spy.close()
    assert spy.port is None


async def _connect_and_start_session(port: int) -> None:
    client = await IppDmeClient.connect("127.0.0.1", port)
    try:
        await client.start_session()
    finally:
        await client.close()


async def test_connecting_when_the_real_server_is_unreachable_just_drops_the_client() -> None:
    # Nothing listens on this port - the spy's outbound connection must fail
    # cleanly (logged, not raised) rather than crashing the accept loop.
    spy, spy_port = await _start_spy(1)
    try:
        with pytest.raises(IppDmeConnectionError):
            await _connect_and_start_session(spy_port)
    finally:
        await spy.close()
