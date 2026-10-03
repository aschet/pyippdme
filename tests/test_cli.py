# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from pyippdme.cli import client, script
from pyippdme.cli.main import _build_parser, main
from pyippdme.cli.script import run_line
from pyippdme.client import IppDmeClient
from pyippdme.protocol.commands import CommandName
from pyippdme.protocol.transport import DEFAULT_PORT
from pyippdme.simulation import DEFAULT_COMMAND_CLASSES
from pyippdme.simulation.classes import (
    register_cartcmm_class,
    register_dme_class,
    register_server_class,
)


def test_client_subcommand_defaults() -> None:
    args = _build_parser().parse_args(["client", "127.0.0.1"])
    assert args.command == "client"
    assert args.host == "127.0.0.1"
    assert args.port == DEFAULT_PORT
    assert args.virtual is False
    assert args.file is None


def test_client_subcommand_with_host_and_port() -> None:
    args = _build_parser().parse_args(["client", "example.org", "1234"])
    assert args.host == "example.org"
    assert args.port == 1234


def test_client_subcommand_virtual_flag() -> None:
    args = _build_parser().parse_args(["client", "--virtual"])
    assert args.virtual is True
    assert args.host is None


def test_client_subcommand_file_flag() -> None:
    args = _build_parser().parse_args(["client", "127.0.0.1", "-f", "session.iscript"])
    assert args.file == "session.iscript"


def test_client_subcommand_session_log_and_commands_file_flags() -> None:
    args = _build_parser().parse_args(
        ["client", "127.0.0.1", "--session-log", "session.log", "--commands-file", "commands.txt"]
    )
    assert args.session_log == "session.log"
    assert args.commands_file == "commands.txt"


def test_client_subcommand_session_log_and_commands_file_default_to_none() -> None:
    args = _build_parser().parse_args(["client", "127.0.0.1"])
    assert args.session_log is None
    assert args.commands_file is None


def test_tui_subcommand_virtual_flag() -> None:
    args = _build_parser().parse_args(["tui", "--virtual"])
    assert args.virtual is True


@pytest.mark.parametrize("subcommand", ["client", "tui"])
def test_virtual_and_host_together_is_rejected(
    subcommand: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("sys.argv", ["ippdme", subcommand, "example.org", "--virtual"])
    with pytest.raises(SystemExit):
        main()


def test_client_without_host_or_virtual_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["ippdme", "client"])
    with pytest.raises(SystemExit):
        main()


def test_tui_without_host_or_virtual_is_allowed() -> None:
    # Unlike the client, the TUI has its own host/port fields + Connect button.
    args = _build_parser().parse_args(["tui"])
    assert args.host is None
    assert args.virtual is False


def test_serve_subcommand_defaults() -> None:
    args = _build_parser().parse_args(["serve"])
    assert args.command == "serve"
    assert args.host == "127.0.0.1"
    assert args.port == DEFAULT_PORT
    assert args.verbose is False


def test_serve_subcommand_options() -> None:
    args = _build_parser().parse_args(["serve", "--host", "0.0.0.0", "--port", "9999", "-v"])
    assert args.host == "0.0.0.0"
    assert args.port == 9999
    assert args.verbose is True


def test_serve_subcommand_components_default_to_full() -> None:
    args = _build_parser().parse_args(["serve"])
    assert args.components == DEFAULT_COMMAND_CLASSES
    assert args.csy_dir is None
    assert args.session_log is None
    assert args.commands_file is None


def test_serve_subcommand_session_log_and_commands_file_flags() -> None:
    args = _build_parser().parse_args(
        ["serve", "--session-log", "session.log", "--commands-file", "commands.txt"]
    )
    assert args.session_log == "session.log"
    assert args.commands_file == "commands.txt"


def test_serve_subcommand_components_flag_selects_a_subset() -> None:
    args = _build_parser().parse_args(["serve", "--components", "server,dme,cartcmm"])
    assert args.components == (register_server_class, register_dme_class, register_cartcmm_class)


def test_serve_subcommand_unknown_component_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["ippdme", "serve", "--components", "bogus"])
    with pytest.raises(SystemExit):
        main()


def test_no_subcommand_errors() -> None:
    with pytest.raises(SystemExit):
        _build_parser().parse_args([])


async def test_run_line_send_and_receive(
    tcp_server_port: int, capsys: pytest.CaptureFixture[str]
) -> None:
    client = await IppDmeClient.connect("127.0.0.1", tcp_server_port)
    try:
        assert await run_line(client, "StartSession()", sys.stdout) is True
        out = capsys.readouterr().out
        assert "&" in out
        assert "%" in out

        assert await run_line(client, "GetDMEVersion()", sys.stdout) is True
        out = capsys.readouterr().out
        assert 'DMEVersion("2.5")' in out
    finally:
        await client.close()


async def test_run_line_reports_server_errors(
    tcp_server_port: int, capsys: pytest.CaptureFixture[str]
) -> None:
    client = await IppDmeClient.connect("127.0.0.1", tcp_server_port)
    try:
        await run_line(client, "GetDMEVersion()", sys.stdout)  # no session yet -> 0008
        assert "0008" in capsys.readouterr().out
    finally:
        await client.close()


async def test_run_line_parse_error(
    tcp_server_port: int, capsys: pytest.CaptureFixture[str]
) -> None:
    client = await IppDmeClient.connect("127.0.0.1", tcp_server_port)
    try:
        await run_line(client, "not a valid method call!!", sys.stdout)
        assert "Parse error" in capsys.readouterr().out
    finally:
        await client.close()


async def test_start_embedded_server_gives_a_working_virtualcmm() -> None:
    host, port, server = await script.start_embedded_server()
    try:
        assert host == script.VIRTUAL_HOST
        assert server.port == port

        dme_client = await IppDmeClient.connect(host, port, network=server.network)
        try:
            await dme_client.start_session()
            (data,) = await dme_client.call(CommandName.GET_MACHINE_CLASS)
            assert "_VirtualCMM" in data.to_wire()
        finally:
            await dme_client.close()
    finally:
        await server.close()


def _fake_input(lines: list[str]) -> object:
    """Build a drop-in replacement for builtins.input(): plays back ``lines``, then EOFError."""
    it = iter(lines)

    def _next(prompt: str = "") -> str:
        del prompt
        try:
            return next(it)
        except StopIteration:
            raise EOFError from None

    return _next


async def test_client_run_interactive_end_to_end(
    tcp_server_port: int, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "builtins.input",
        _fake_input(["StartSession()", "GetDMEVersion()"]),
    )

    await client.run("127.0.0.1", tcp_server_port)  # then EOFError from _fake_input ends the loop

    out = capsys.readouterr().out
    assert f"Connected to 127.0.0.1:{tcp_server_port}" in out
    assert '"2.5"' in out


async def test_client_run_with_virtual_starts_and_tears_down_its_own_server(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "builtins.input",
        _fake_input(["StartSession()", "GetMachineClass()"]),
    )

    await client.run(None, DEFAULT_PORT, virtual=True)

    out = capsys.readouterr().out
    assert "Started an in-process VirtualCMM" in out
    assert "_VirtualCMM" in out


async def test_client_run_exits_on_eof(
    tcp_server_port: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _raise_eof(prompt: str = "") -> str:
        del prompt
        raise EOFError

    monkeypatch.setattr("builtins.input", _raise_eof)

    await client.run("127.0.0.1", tcp_server_port)  # must return, not hang or raise


async def test_client_run_exits_on_keyboard_interrupt(
    tcp_server_port: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _raise_interrupt(prompt: str = "") -> str:
        del prompt
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", _raise_interrupt)

    await client.run("127.0.0.1", tcp_server_port)  # must return, not hang or raise


async def test_client_run_with_file_runs_it_non_interactively(
    tcp_server_port: int, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    script = tmp_path / "session.iscript"
    script.write_text("StartSession()\nGoTo(X(10), Y(20))\nGet(X(), Y())\n")

    await client.run("127.0.0.1", tcp_server_port, script=script)

    out = capsys.readouterr().out
    assert "> StartSession()" in out
    assert "# X(10),Y(20)" in out


async def test_client_run_with_file_and_virtual(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    script = tmp_path / "session.iscript"
    script.write_text("StartSession()\nGetMachineClass()\n")

    await client.run(None, DEFAULT_PORT, virtual=True, script=script)

    out = capsys.readouterr().out
    assert "Started an in-process VirtualCMM" in out
    assert "_VirtualCMM" in out


async def test_client_run_session_log_records_wire_traffic_and_connection_events(
    tcp_server_port: int, tmp_path: Path
) -> None:
    script = tmp_path / "session.iscript"
    script.write_text("StartSession()\nGetDMEVersion()\n")
    log_path = tmp_path / "session.log"

    await client.run("127.0.0.1", tcp_server_port, script=script, session_log=str(log_path))

    logged = log_path.read_text()
    assert "CONNECTED" in logged
    assert "DISCONNECTED" in logged
    assert any(line.endswith("StartSession()") for line in logged.splitlines())
    assert '"2.5"' in logged


async def test_client_run_commands_file_records_interactively_typed_commands(
    tcp_server_port: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("builtins.input", _fake_input(["StartSession()", "GetDMEVersion()"]))
    commands_path = tmp_path / "commands.txt"

    await client.run("127.0.0.1", tcp_server_port, commands_file=str(commands_path))

    assert commands_path.read_text().splitlines() == ["StartSession()", "GetDMEVersion()"]


async def test_client_run_commands_file_records_commands_from_a_script(
    tcp_server_port: int, tmp_path: Path
) -> None:
    script = tmp_path / "session.iscript"
    script.write_text("# a comment, not a command\nStartSession()\n\nGetDMEVersion()\n")
    commands_path = tmp_path / "commands.txt"

    await client.run("127.0.0.1", tcp_server_port, script=script, commands_file=str(commands_path))

    # Comments/blanks from the source script are not recorded, only real commands.
    assert commands_path.read_text().splitlines() == ["StartSession()", "GetDMEVersion()"]
