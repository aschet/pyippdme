# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for pyippdme.cli.script: the shared engine behind ``ippdme client --file``."""

from __future__ import annotations

import io
from pathlib import Path

from pyippdme.cli.script import run_script_lines, start_embedded_server
from pyippdme.client import IppDmeClient


async def test_run_script_lines_transcript(tcp_server_port: int, tmp_path: Path) -> None:
    script = tmp_path / "session.iscript"
    script.write_text("# start a session\nStartSession()\n\nGoTo(X(10), Y(20))\nGet(X(), Y())\n")
    output = io.StringIO()

    client = await IppDmeClient.connect("127.0.0.1", tcp_server_port)
    try:
        await run_script_lines(client, script, output)
    finally:
        await client.close()

    text = output.getvalue()
    lines = text.splitlines()
    assert lines[0] == "# start a session"
    assert "> StartSession()" in lines
    assert lines[lines.index("> StartSession()") + 1] == "&"
    assert "" in lines  # the blank line was preserved
    assert "> GoTo(X(10), Y(20))" in text
    assert "> Get(X(), Y())" in text
    assert "# X(10),Y(20)" in text


async def test_run_script_lines_against_an_embedded_virtualcmm(tmp_path: Path) -> None:
    script = tmp_path / "session.iscript"
    script.write_text("StartSession()\nGetMachineClass()\n")
    output = io.StringIO()

    host, port, server = await start_embedded_server()
    try:
        client = await IppDmeClient.connect(host, port, network=server.network)
        try:
            await run_script_lines(client, script, output)
        finally:
            await client.close()
    finally:
        await server.close()

    assert "_VirtualCMM" in output.getvalue()


async def test_run_script_lines_reports_parse_and_server_errors(
    tcp_server_port: int, tmp_path: Path
) -> None:
    script = tmp_path / "session.iscript"
    script.write_text("StartSession()\nnot a valid method call!!\nGetSupportedArguments(1)\n")
    output = io.StringIO()

    client = await IppDmeClient.connect("127.0.0.1", tcp_server_port)
    try:
        await run_script_lines(client, script, output)
    finally:
        await client.close()

    text = output.getvalue()
    assert "Parse error" in text
    assert "0509" in text  # GetSupportedArguments(1): expects a single string


async def test_run_script_lines_continues_after_server_error(
    tcp_server_port: int, tmp_path: Path
) -> None:
    script = tmp_path / "session.iscript"
    script.write_text(
        "StartSession()\nGetSupportedArguments(1)\nClearAllErrors()\nGetDMEVersion()\n"
    )
    output = io.StringIO()

    client = await IppDmeClient.connect("127.0.0.1", tcp_server_port)
    try:
        await run_script_lines(client, script, output)
    finally:
        await client.close()

    text = output.getvalue()
    assert "> GetDMEVersion()" in text
    assert 'DMEVersion("2.5")' in text


async def test_run_script_lines_reuses_an_already_connected_client(
    tcp_server_port: int, tmp_path: Path
) -> None:
    """The .run meta-command's engine: same transcript, but no new connection is opened."""
    script = tmp_path / "session.iscript"
    script.write_text("StartSession()\nGoTo(X(10), Y(20))\nGet(X(), Y())\n")
    output = io.StringIO()

    client = await IppDmeClient.connect("127.0.0.1", tcp_server_port)
    try:
        ok = await run_script_lines(client, script, output)
        assert ok is True
        # The connection is still usable afterwards - run_script_lines didn't close it.
        (data,) = await client.call("GetDMEVersion")
        assert data is not None
    finally:
        await client.close()

    text = output.getvalue()
    assert "> StartSession()" in text
    assert "# X(10),Y(20)" in text
