# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for pyippdme.cli.spy: the console/file-logging consumer of pyippdme.spy.Spy."""

from __future__ import annotations

import asyncio
import contextlib
import io
import re
import socket
from pathlib import Path

from pyippdme import CommandName, IppDmeClient
from pyippdme.cli.script import run_script_lines
from pyippdme.cli.session_log import strip_tag
from pyippdme.cli.spy import run

#: A session-log line: ``<timestamp> <marker> <rest>``.
_SESSION_LOG_LINE_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3} (?P<marker>\S+) (?P<rest>.*)$"
)


def test_strip_tag_drops_the_leading_tag_and_keeps_the_rest() -> None:
    assert strip_tag("00001 StartSession()") == "StartSession()"
    assert strip_tag('00002 # DMEVersion("2.5")') == '# DMEVersion("2.5")'
    assert strip_tag("00003 %") == "%"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


async def _spy_a_simple_session(tcp_server_port: int, **run_kwargs: object) -> int:
    """Start a spy (per ``run_kwargs``), run one StartSession()+GetDMEVersion() through it."""
    spy_port = _free_port()
    task = asyncio.create_task(
        run("127.0.0.1", tcp_server_port, "127.0.0.1", spy_port, **run_kwargs)  # type: ignore[arg-type]
    )
    try:
        await asyncio.sleep(0.05)
        client = await IppDmeClient.connect("127.0.0.1", spy_port)
        await client.start_session()
        await client.call(CommandName.GET_DME_VERSION)
        await client.close()
        await asyncio.sleep(0.05)
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    return spy_port


async def test_session_log_is_interleaved_timestamped_marked_and_tag_kept(
    tcp_server_port: int, tmp_path: Path
) -> None:
    log_path = tmp_path / "session.log"
    await _spy_a_simple_session(tcp_server_port, session_log=str(log_path))

    lines = log_path.read_text().splitlines()
    matches = [_SESSION_LOG_LINE_RE.match(line) for line in lines]
    assert all(matches), lines

    markers_and_rest = [(m["marker"], m["rest"]) for m in matches if m is not None]
    assert markers_and_rest[0][0] == "CONNECTED"
    assert markers_and_rest[-1][0] == "DISCONNECTED"
    assert [pair for pair in markers_and_rest if pair[0] in (">", "<")] == [
        (">", "00001 StartSession()"),
        ("<", "00001 &"),
        ("<", "00001 %"),
        (">", "00002 GetDMEVersion()"),
        ("<", "00002 &"),
        ("<", '00002 # DMEVersion("2.5")'),
        ("<", "00002 %"),
    ]


async def test_commands_file_is_replayable_by_ippdme_script(
    tcp_server_port: int, tmp_path: Path
) -> None:
    commands_path = tmp_path / "commands.txt"
    await _spy_a_simple_session(tcp_server_port, commands_file=str(commands_path))

    assert commands_path.read_text().splitlines() == ["StartSession()", "GetDMEVersion()"]

    replay_client = await IppDmeClient.connect("127.0.0.1", tcp_server_port)
    try:
        output = io.StringIO()
        await run_script_lines(replay_client, commands_path, output)
        assert "0501" not in output.getvalue()  # every replayed line was a real command
    finally:
        await replay_client.close()


async def test_session_log_and_commands_file_work_together(
    tcp_server_port: int, tmp_path: Path
) -> None:
    log_path = tmp_path / "session.log"
    commands_path = tmp_path / "commands.txt"
    await _spy_a_simple_session(
        tcp_server_port, session_log=str(log_path), commands_file=str(commands_path)
    )

    assert commands_path.read_text().splitlines() == ["StartSession()", "GetDMEVersion()"]
    assert any(line.endswith("StartSession()") for line in log_path.read_text().splitlines())
