# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for pyippdme.cli._interaction, the REPL/TUI's shared command-line runner."""

from __future__ import annotations

from pyippdme import IppDmeClient
from pyippdme.cli._interaction import (
    Acked,
    Completed,
    Failed,
    ParseFailed,
    Received,
    format_error,
    run_command_line,
)


async def test_run_command_line_happy_path(started_client: IppDmeClient) -> None:
    events = [e async for e in run_command_line(started_client, "GetDMEVersion()")]
    assert isinstance(events[0], Acked)
    assert isinstance(events[1], Received)
    assert "2.5" in events[1].payload.to_wire()
    assert isinstance(events[-1], Completed)


async def test_run_command_line_parse_error(started_client: IppDmeClient) -> None:
    (event,) = [e async for e in run_command_line(started_client, "not valid ippdme")]
    assert isinstance(event, ParseFailed)


async def test_run_command_line_server_error(client: IppDmeClient) -> None:
    # No StartSession() first: every command is rejected with 0008.
    events = [e async for e in run_command_line(client, "GetDMEVersion()")]
    (event,) = [e for e in events if isinstance(e, Failed)]
    assert event.error.number == "0008"
    assert "0008" in format_error(event.error)
