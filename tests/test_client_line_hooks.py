# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for IppDmeClient's on_line_sent/on_line_received observability hooks."""

from __future__ import annotations

from pyippdme import CommandName, IppDmeClient
from pyippdme.client.model import IppDmeMachine


async def test_on_line_sent_reports_every_command_including_its_tag(server_port: int) -> None:
    sent: list[str] = []
    client = await IppDmeClient.connect("127.0.0.1", server_port, on_line_sent=sent.append)
    try:
        await client.start_session()
        await client.call(CommandName.GET_DME_VERSION)
    finally:
        await client.close()

    assert sent == ["00001 StartSession()", "00002 GetDMEVersion()"]


async def test_on_line_received_reports_every_response_without_the_terminator(
    server_port: int,
) -> None:
    received: list[str] = []
    client = await IppDmeClient.connect("127.0.0.1", server_port, on_line_received=received.append)
    try:
        await client.start_session()
        await client.call(CommandName.GET_DME_VERSION)
    finally:
        await client.close()

    assert received == [
        "00001 &",
        "00001 %",
        "00002 &",
        '00002 # DMEVersion("2.5")',
        "00002 %",
    ]
    assert not any(line.endswith("\r") or line.endswith("\n") for line in received)


async def test_a_raising_hook_does_not_break_the_connection(server_port: int) -> None:
    def _bad_hook(_text: str) -> None:
        raise RuntimeError("boom")

    client = await IppDmeClient.connect(
        "127.0.0.1", server_port, on_line_sent=_bad_hook, on_line_received=_bad_hook
    )
    try:
        await client.start_session()
        (version,) = await client.call(CommandName.GET_DME_VERSION)
        assert version is not None
    finally:
        await client.close()


async def test_no_hooks_given_is_the_default_and_works_unchanged(server_port: int) -> None:
    client = await IppDmeClient.connect("127.0.0.1", server_port)
    try:
        await client.start_session()
        (version,) = await client.call(CommandName.GET_DME_VERSION)
        assert version is not None
    finally:
        await client.close()


async def test_ippdme_machine_connect_forwards_the_same_hooks(server_port: int) -> None:
    sent: list[str] = []
    received: list[str] = []
    machine = await IppDmeMachine.connect(
        "127.0.0.1", server_port, on_line_sent=sent.append, on_line_received=received.append
    )
    try:
        await machine.start_session()
    finally:
        await machine.close()

    assert sent == ["00001 StartSession()"]
    assert received == ["00001 &", "00001 %"]
