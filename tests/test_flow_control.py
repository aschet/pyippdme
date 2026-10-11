# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""The server delays the acknowledgement while too many commands wait (5.4.3)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from pyippdme import VirtualCMM
from pyippdme.protocol.network import MemoryNetwork
from pyippdme.protocol.transport import LineTransport
from pyippdme.types.csy import InMemoryCsyStore


@pytest.fixture
async def raw(network: MemoryNetwork) -> AsyncIterator[tuple[VirtualCMM, LineTransport]]:
    server = VirtualCMM(network=network, csy_store=InMemoryCsyStore(), max_pending=1)
    port = await server.start()
    reader, writer = await network.open_connection("x", port, limit=2**20)
    transport = LineTransport(reader, writer)
    try:
        yield server, transport
    finally:
        await transport.close()
        await server.close()


async def _send(transport: LineTransport, line: str) -> None:
    transport.write_line(f"{line}\r\n".encode())
    await transport.drain()


async def _read(transport: LineTransport) -> str:
    return (await asyncio.wait_for(transport.read_line(), 2.0)).strip()


async def test_acknowledgements_wait_until_there_is_room(
    raw: tuple[VirtualCMM, LineTransport],
) -> None:
    _, t = raw
    await _send(t, "00001 StartSession()")
    assert await _read(t) == "00001 &"
    assert await _read(t) == "00001 %"
    # A long command runs; one more may wait in the queue; the third is not acknowledged yet.
    await _send(t, "00002 Home()")
    await _send(t, "00003 GetDMEVersion()")
    await _send(t, "00004 GetDMEVersion()")
    lines = []
    while True:
        try:
            lines.append(await asyncio.wait_for(t.read_line(), 0.5))
        except TimeoutError:
            break
        if lines[-1].strip() == "00004 %":
            break
    text = [ln.strip() for ln in lines]
    # Every command is acknowledged before its own data and completion, in order.
    for tag in ("00002", "00003", "00004"):
        assert text.index(f"{tag} &") < text.index(f"{tag} %")
    assert text.index("00003 &") < text.index("00004 &")


async def test_a_prioritized_command_is_never_delayed(
    raw: tuple[VirtualCMM, LineTransport],
) -> None:
    _, t = raw
    await _send(t, "00001 StartSession()")
    await _read(t)
    await _read(t)
    for tag in ("00002", "00003", "00004", "00005"):
        await _send(t, f"{tag} GetDMEVersion()")
    await _send(t, "00006 GetErrStatusE()")
    seen = []
    while True:
        line = (await asyncio.wait_for(t.read_line(), 2.0)).strip()
        seen.append(line)
        if line == "00006 %":
            break
    assert "00006 &" in seen


async def test_the_third_command_is_not_acknowledged_while_two_wait_behind_a_running_one(
    raw: tuple[VirtualCMM, LineTransport],
) -> None:
    server, t = raw
    release = asyncio.Event()

    async def slow(_ctx: object, _args: object) -> None:
        await release.wait()

    server.registry.register("SlowOne", slow, arguments=())
    await _send(t, "00001 StartSession()")
    await _read(t)
    await _read(t)
    await _send(t, "00002 SlowOne()")
    assert await _read(t) == "00002 &"
    await _send(t, "00003 GetDMEVersion()")
    assert await _read(t) == "00003 &"  # one may wait (max_pending is 1)
    await _send(t, "00004 GetDMEVersion()")
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(t.read_line(), 0.3)  # no Ack: the server is not ready for it
    release.set()
    rest = []
    while True:
        line = (await asyncio.wait_for(t.read_line(), 2.0)).strip()
        rest.append(line)
        if line == "00004 %":
            break
    assert rest.index("00004 &") < rest.index("00004 %")
    assert rest.index("00002 %") < rest.index("00004 &")  # it came when there was room


async def test_abort_e_answers_the_commands_that_were_not_yet_acknowledged(
    raw: tuple[VirtualCMM, LineTransport],
) -> None:
    server, t = raw
    never = asyncio.Event()

    async def slow(_ctx: object, _args: object) -> None:
        await never.wait()

    server.registry.register("SlowOne", slow, arguments=())
    await _send(t, "00001 StartSession()")
    await _read(t)
    await _read(t)
    await _send(t, "00002 SlowOne()")
    await _send(t, "00003 GetDMEVersion()")
    await _send(t, "00004 GetDMEVersion()")
    await _send(t, "00005 AbortE()")
    seen: list[str] = []
    while True:
        line = (await asyncio.wait_for(t.read_line(), 2.0)).strip()
        seen.append(line)
        if line == "00005 %":
            break
    for tag in ("00003", "00004"):
        assert f"{tag} &" in seen  # acknowledged first ...
        assert any(ln.startswith(f"{tag} !") for ln in seen)  # ... then aborted
        first_ack = seen.index(f"{tag} &")
        first_error = next(i for i, ln in enumerate(seen) if ln.startswith(f"{tag} !"))
        assert first_ack < first_error
