# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Behaviour the standard prescribes for the protocol layer: errors, aborting, XML payloads."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from pyippdme import CommandName, IppDmeClient, IppDmeServer
from pyippdme.exceptions import IppDmeServerError
from pyippdme.protocol.ast import Argument, Xml
from pyippdme.protocol.errors import ErrorCode, ErrorSeverity, ServerError
from pyippdme.protocol.network import MemoryNetwork
from pyippdme.protocol.parser import parse_command, parse_response
from pyippdme.server.classes import register_dme_class, register_server_class
from pyippdme.server.registry import CommandContext, MachineState

# -- XML payloads --------------------------------------------------------------


def test_xml_with_parentheses_parses_as_one_payload() -> None:
    response = parse_response('00001 # Tool.Id(<a b="(x)">t(1)</a>)')
    assert response.to_wire() == '00001 # Tool.Id(<a b="(x)">t(1)</a>)'
    command = parse_command("00002 Foo(<a>(</a>)")
    assert command.method.xml == Xml("<a>(</a>")


def test_a_bare_xml_response_may_contain_parentheses() -> None:
    assert parse_response("00001 # <a>(x)</a>").to_wire() == "00001 # <a>(x)</a>"


def test_angle_brackets_inside_a_string_are_not_xml() -> None:
    command = parse_command('00001 Foo("a(<b>)", X(1))')
    assert command.method.xml is None
    assert len(command.method.args) == 2


# -- raw lines -------------------------------------------------------------------


@pytest.fixture
async def raw_server(network: MemoryNetwork) -> AsyncIterator[tuple[MemoryNetwork, int]]:
    server: IppDmeServer[MachineState] = IppDmeServer(
        command_classes=[register_server_class, register_dme_class], network=network
    )
    port = await server.start("x", 0)
    try:
        yield network, port
    finally:
        await server.close()


async def _exchange(network: MemoryNetwork, port: int, *lines: bytes) -> list[str]:
    """Send ``lines`` and return every response line up to the last one's ``%``."""
    reader, writer = await network.open_connection("x", port)
    responses: list[str] = []
    try:
        for line in lines:
            writer.write(line)
            await writer.drain()
            while True:
                text = (await asyncio.wait_for(reader.readuntil(b"\r\n"), 1.0)).decode("ascii")
                responses.append(text.removesuffix("\r\n"))
                if text.endswith(" %\r\n"):
                    break
    finally:
        writer.close()
    return responses


@pytest.mark.parametrize(
    ("line", "number", "tag"),
    [
        (b"00001 GetDMEVersion(\t)\r\n", "0007", "00001"),  # illegal character
        (b"00001 GetDMEVersion(\xe9)\r\n", "0007", "00001"),
        (b"0000X GetDMEVersion()\r\n", "0001", "00000"),  # illegal tag
        (b"00000 GetDMEVersion()\r\n", "0001", "00000"),
        (b"E0000 GetDMEVersion()\r\n", "0001", "00000"),
        (b"00001GetDMEVersion()\r\n", "0002", "00001"),  # no space at position 6
        (b"00001 Get DME Version\r\n", "0008", "00001"),  # protocol error
    ],
)
async def test_a_malformed_line_gets_the_matching_predefined_error(
    raw_server: tuple[MemoryNetwork, int], line: bytes, number: str, tag: str
) -> None:
    """5.1, 5.4.2 and Annex B: errors 0001, 0002, 0007 and 0008."""
    network, port = raw_server
    responses = await _exchange(network, port, line)
    assert responses[0].startswith(f"{tag} ! Error(")
    assert f",{number}," in responses[0]
    assert responses[-1] == f"{tag} %"


async def test_after_a_protocol_error_the_client_must_clear_it(
    raw_server: tuple[MemoryNetwork, int],
) -> None:
    network, port = raw_server
    responses = await _exchange(
        network,
        port,
        b"00001 StartSession()\r\n",
        b"00002 Foo Bar\r\n",
        b"00003 GetDMEVersion()\r\n",
        b"00004 ClearAllErrors()\r\n",
        b"00005 GetDMEVersion()\r\n",
    )
    assert any(r.startswith("00003 ! Error(2,0514,") for r in responses)  # "Use ClearAllErrors"
    assert any(r.startswith("00005 # ") for r in responses)


async def test_every_error_names_the_command_that_caused_it(
    raw_server: tuple[MemoryNetwork, int],
) -> None:
    """5.6.3: F3 names the method that caused the error; the grammar wants a non-empty string."""
    network, port = raw_server
    responses = await _exchange(
        network,
        port,
        b"00001 GetDMEVersion()\r\n",  # no session yet
        b"00002 StartSession()\r\n",
        b"00003 NoSuchCommand()\r\n",
        b"00004 GetDMEVersion()\r\n",  # 0514, the error above is still active
    )
    errors = [r for r in responses if " ! Error(" in r]
    assert len(errors) == 3
    for error in errors:
        assert '""' not in error
    assert '"GetDMEVersion"' in errors[0]
    assert '"NoSuchCommand"' in errors[1]
    assert '"GetDMEVersion"' in errors[2]


# -- AbortE and pending commands ---------------------------------------------------


class _Gate:
    """Test handlers: ``Slow`` waits for ``release`` and then fails with a severe error."""

    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.started = asyncio.Event()

    async def slow(self, _ctx: CommandContext[MachineState], _args: tuple[Argument, ...]) -> None:
        self.started.set()
        await self.release.wait()
        raise ServerError(
            ErrorSeverity.CRITICAL, ErrorCode.BAD_ARGUMENT, "Slow", "Failed on purpose"
        )

    async def hang(self, _ctx: CommandContext[MachineState], _args: tuple[Argument, ...]) -> None:
        self.started.set()
        await asyncio.sleep(30)


@pytest.fixture
async def gated(network: MemoryNetwork) -> AsyncIterator[tuple[IppDmeClient, _Gate]]:
    gate = _Gate()
    server: IppDmeServer[MachineState] = IppDmeServer(
        command_classes=[register_server_class, register_dme_class], network=network
    )
    server.registry.register("Slow", gate.slow)
    server.registry.register("Hang", gate.hang)
    port = await server.start("x", 0)
    client = await IppDmeClient.connect("x", port, network=network)
    await client.start_session()
    try:
        yield client, gate
    finally:
        await client.close()
        await server.close()


async def test_a_severe_error_aborts_the_commands_still_waiting(
    gated: tuple[IppDmeClient, _Gate],
) -> None:
    """5.6: "In case of error severity equal or greater 2 the server will abort all pending"."""
    client, gate = gated
    slow = client.send("Slow")
    await gate.started.wait()
    queued = client.send(CommandName.GET_DME_VERSION)
    await asyncio.wait_for(queued.wait_ack(), 1.0)
    gate.release.set()
    with pytest.raises(IppDmeServerError) as excinfo:
        await slow.wait_complete()
    assert excinfo.value.error.number == "0509"
    with pytest.raises(IppDmeServerError) as excinfo:
        await asyncio.wait_for(queued.wait_complete(), 1.0)
    assert excinfo.value.error.number == "0006"  # "Transaction aborted"
    assert excinfo.value.error.cause == "GetDMEVersion"


async def test_abort_e_with_nothing_running_still_requires_clear_all_errors(
    gated: tuple[IppDmeClient, _Gate],
) -> None:
    """6.3.1 (AbortE): the next command must be ClearAllErrors, even if nothing was aborted."""
    client, _gate = gated
    await client.call(CommandName.ABORT_E)
    with pytest.raises(IppDmeServerError) as excinfo:
        await client.call(CommandName.GET_DME_VERSION)
    assert excinfo.value.error.number == "0514"
    await client.clear_all_errors()
    assert await client.call(CommandName.GET_DME_VERSION)


async def test_abort_e_rejects_waiting_commands_and_cancels_the_running_one(
    gated: tuple[IppDmeClient, _Gate],
) -> None:
    client, gate = gated
    running = client.send("Hang")
    await gate.started.wait()
    waiting = client.send(CommandName.GET_DME_VERSION)
    await asyncio.wait_for(waiting.wait_ack(), 1.0)
    await client.call(CommandName.ABORT_E)
    for txn in (running, waiting):
        with pytest.raises(IppDmeServerError) as excinfo:
            await asyncio.wait_for(txn.wait_complete(), 1.0)
        assert excinfo.value.error.number == "0006"
