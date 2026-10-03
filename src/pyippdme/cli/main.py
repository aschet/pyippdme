# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""``ippdme`` command-line entry point."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
from collections.abc import Callable
from pathlib import Path

from pyippdme.cli import virtual
from pyippdme.cli.session_log import SessionLog, strip_tag
from pyippdme.exceptions import IppDmeConnectionError
from pyippdme.protocol.transport import DEFAULT_PORT
from pyippdme.server.registry import CommandRegistry, component_name
from pyippdme.simulation import DEFAULT_COMMAND_CLASSES
from pyippdme.types.csy import FileCsyStore

#: Maps a ``--components`` name to its registration function, derived from
#: :data:`~pyippdme.simulation.DEFAULT_COMMAND_CLASSES` itself rather than a
#: separately hand-maintained list, so it can't drift from what the library
#: actually knows how to register.
_COMPONENTS_BY_NAME: dict[str, Callable[[CommandRegistry], None]] = {
    component_name(fn): fn for fn in DEFAULT_COMMAND_CLASSES
}


def _parse_components(value: str) -> tuple[Callable[[CommandRegistry], None], ...]:
    names = [name.strip() for name in value.split(",") if name.strip()]
    try:
        return tuple(_COMPONENTS_BY_NAME[name] for name in names)
    except KeyError as exc:
        choices = ", ".join(sorted(_COMPONENTS_BY_NAME))
        raise argparse.ArgumentTypeError(
            f"unknown component {exc.args[0]!r} - choose from {choices}"
        ) from None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ippdme", description="I++ DME (VDMA 8722) client/server toolkit"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    client_parser = subparsers.add_parser(
        "client",
        help="barebones tool for sending commands to a server - typed interactively, "
        "or from a file with --file",
    )
    client_parser.add_argument("host", nargs="?", help="server host to connect to")
    client_parser.add_argument(
        "port", nargs="?", type=int, default=DEFAULT_PORT, help="server port"
    )
    client_parser.add_argument(
        "--virtual",
        action="store_true",
        help="start an in-process VirtualCMM and connect to it instead of "
        "host/port - no separate 'ippdme serve' needed",
    )
    client_parser.add_argument(
        "-f",
        "--file",
        help="run every command in this file instead of prompting interactively, one per line",
    )
    virtual.add_arguments(client_parser)
    client_parser.add_argument(
        "--session-log",
        help="also record every wire line sent/received, plus connect/disconnect, to this "
        "file (unified format shared with serve/spy - see pyippdme.cli.session_log)",
    )
    client_parser.add_argument(
        "--commands-file",
        help="also record every command actually run (typed, or from --file) to this file, "
        "one per line - a valid 'ippdme client --file' input, for replay",
    )

    subparsers.add_parser(
        "gui",
        add_help=False,
        help="virtual CMM with a visible digital twin in a Qt window (needs pyippdme[gui]); "
        "run 'ippdme gui --help' for its options",
    )

    subparsers.add_parser(
        "client-gui",
        add_help=False,
        help="Qt window to send commands to a server from dialogs (needs pyippdme[gui]); "
        "run 'ippdme client-gui --help' for its options",
    )

    tui_parser = subparsers.add_parser(
        "tui", help="full-screen terminal UI for sending commands to a server (needs pyippdme[tui])"
    )
    virtual.add_arguments(tui_parser)
    tui_parser.add_argument("host", nargs="?", help="server host to connect to on startup")
    tui_parser.add_argument("port", nargs="?", type=int, default=DEFAULT_PORT, help="server port")
    tui_parser.add_argument(
        "--virtual",
        action="store_true",
        help="start an in-process VirtualCMM and connect to it instead of "
        "host/port - no separate 'ippdme serve' needed",
    )

    serve_parser = subparsers.add_parser(
        "serve",
        help="run a simulated CMM server (VirtualCMM) - reports itself as virtual "
        "(6.4.1) and persists named coordinate systems to disk",
    )
    serve_parser.add_argument(
        "--host", default="127.0.0.1", help="address to bind (default: 127.0.0.1)"
    )
    serve_parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="port to bind")
    serve_parser.add_argument("-v", "--verbose", action="store_true", help="enable info logging")
    serve_parser.add_argument(
        "--components",
        type=_parse_components,
        default=DEFAULT_COMMAND_CLASSES,
        metavar="NAME[,NAME...]",
        help="comma-separated command classes to simulate (default: all) - choices: "
        + ", ".join(sorted(_COMPONENTS_BY_NAME))
        + " - GetMachineClass() (6.4.1) is derived from this selection",
    )
    virtual.add_arguments(serve_parser)
    serve_parser.add_argument(
        "--csy-dir", help="directory for persisted coordinate systems (default: ~/.pyippdme/csy)"
    )
    serve_parser.add_argument(
        "--session-log",
        help="also record every wire line received/sent, plus connect/disconnect, to this "
        "file (unified format shared with client/spy - see pyippdme.cli.session_log)",
    )
    serve_parser.add_argument(
        "--commands-file",
        help="also record the connecting client's own commands (tag-stripped) to this file - "
        "a valid 'ippdme client --file' input, for replaying the session",
    )

    spy_parser = subparsers.add_parser(
        "spy",
        help="transparent proxy: relay traffic between a client and a real server "
        "unchanged, logging each message to the console with a timestamp",
    )
    spy_parser.add_argument("host", help="real server host to forward to")
    spy_parser.add_argument(
        "port", nargs="?", type=int, default=DEFAULT_PORT, help="real server port"
    )
    spy_parser.add_argument(
        "--listen-host", default="127.0.0.1", help="address to bind (default: 127.0.0.1)"
    )
    spy_parser.add_argument(
        "--listen-port", type=int, default=DEFAULT_PORT, help="port to bind (default: 1294)"
    )
    spy_parser.add_argument(
        "--session-log",
        help="also record every relayed message, plus connect/disconnect, to this file "
        "(unified format shared with client/serve - see pyippdme.cli.session_log)",
    )
    spy_parser.add_argument(
        "--commands-file",
        help="also write client-to-server messages (tag-stripped) to this file - "
        "a valid 'ippdme client --file' input, for replaying the session",
    )

    return parser


async def _run_client(
    host: str | None,
    port: int,
    virtual: bool,
    file: str | None,
    session_log: str | None,
    commands_file: str | None,
) -> None:
    from pyippdme.cli.client import run

    script = None
    if file is not None:
        script = Path(file)
        if not script.is_file():
            raise SystemExit(f"Script file not found: {script}")

    try:
        await run(
            host,
            port,
            virtual=virtual,
            script=script,
            session_log=session_log,
            commands_file=commands_file,
        )
    except IppDmeConnectionError as exc:
        raise SystemExit(f"Connection failed: {exc}") from exc


def _run_gui(argv: list[str]) -> None:
    try:
        from pyippdme.gui.app import main as gui_main
    except ImportError as exc:
        raise SystemExit(
            "The GUI needs the optional 'gui' dependency group: pip install pyippdme[gui]"
        ) from exc
    raise SystemExit(gui_main(argv))


def _run_client_gui(argv: list[str]) -> None:
    try:
        from pyippdme.gui.client_window import main as client_gui_main
    except ImportError as exc:
        raise SystemExit(
            "The GUI needs the optional 'gui' dependency group: pip install pyippdme[gui]"
        ) from exc
    raise SystemExit(client_gui_main(argv))


def _run_tui(host: str | None, port: int, virtual: bool) -> None:
    try:
        from pyippdme.cli.tui import run
    except ImportError as exc:
        raise SystemExit(
            "The TUI needs the optional 'tui' dependency group: pip install pyippdme[tui]"
        ) from exc
    run(host, port, virtual=virtual)


async def _run_serve(
    host: str,
    port: int,
    verbose: bool,
    components: tuple[Callable[[CommandRegistry], None], ...],
    csy_dir: str | None,
    session_log: str | None,
    commands_file: str | None,
    options: virtual.SimulationOptions,
) -> None:
    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    csy_store = FileCsyStore(Path(csy_dir)) if csy_dir else None

    with contextlib.ExitStack() as files:
        log = SessionLog(
            files.enter_context(Path(session_log).open("w", buffering=1))
            if session_log is not None
            else None
        )
        commands = (
            files.enter_context(Path(commands_file).open("w", buffering=1))
            if commands_file is not None
            else None
        )

        def on_line_received(text: str) -> None:
            log.to_server(text)
            if commands is not None:
                print(strip_tag(text), file=commands)

        server = virtual.build_server(
            options,
            csy_store=csy_store,
            command_classes=components,
            on_line_received=on_line_received,
            on_line_sent=log.to_client,
            on_connect=log.connected,
            on_disconnect=log.disconnected,
        )
        kind = "digital twin" if options.uses_twin else "VirtualCMM"
        print(f"pyippdme {kind} server listening on {host}:{port}")
        await server.serve_forever(host, port)


async def _run_spy(
    host: str,
    port: int,
    listen_host: str,
    listen_port: int,
    session_log: str | None,
    commands_file: str | None,
) -> None:
    from pyippdme.cli.spy import run

    await run(
        host,
        port,
        listen_host,
        listen_port,
        session_log=session_log,
        commands_file=commands_file,
    )


def main() -> None:
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "gui":
        _run_gui(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "client-gui":
        _run_client_gui(sys.argv[2:])
    parser = _build_parser()
    args = parser.parse_args()
    if args.command in ("client", "tui"):
        virtual.configure(virtual.options_from_args(args))
    if args.command in ("client", "tui") and args.virtual and args.host is not None:
        parser.error("argument host: not allowed with argument --virtual")
    if args.command == "client" and not args.virtual and args.host is None:
        # Unlike the TUI (which has its own host/port fields + Connect button),
        # the client has no meta-command to connect after starting - it must
        # know where to connect up front.
        parser.error("argument host is required unless --virtual is given")
    try:
        if args.command == "client":
            asyncio.run(
                _run_client(
                    args.host,
                    args.port,
                    args.virtual,
                    args.file,
                    args.session_log,
                    args.commands_file,
                )
            )
        elif args.command == "tui":
            _run_tui(args.host, args.port, args.virtual)
        elif args.command == "serve":
            asyncio.run(
                _run_serve(
                    args.host,
                    args.port,
                    args.verbose,
                    args.components,
                    args.csy_dir,
                    args.session_log,
                    args.commands_file,
                    virtual.options_from_args(args),
                )
            )
        elif args.command == "spy":
            asyncio.run(
                _run_spy(
                    args.host,
                    args.port,
                    args.listen_host,
                    args.listen_port,
                    args.session_log,
                    args.commands_file,
                )
            )
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
