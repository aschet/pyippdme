# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""A barebones tool for driving an I++ DME server by hand: ``ippdme client``.

Deliberately minimal - just enough to see whether a server is actually
answering, not a full shell: no meta-commands, no tab-completion, no
reconnecting mid-session. The connection (host/port, or ``--virtual`` for
an in-process :class:`~pyippdme.simulation.virtual_cmm.VirtualCMM`) is fixed
for the process's whole lifetime. Two modes, chosen by whether a script
file is given:

- Interactive (no file): type a bare method call
  (``GoTo(X(10), Y(20))``), press enter, see its Ack/Data/Done/Error as it
  arrives; :kbd:`Ctrl-D`/:kbd:`Ctrl-C` exits.
- Non-interactive (``-f``/``--file``): run every command in the file, one
  per line, printing the same Ack/Data/Done/Error transcript.

Unsolicited (``E0000``) events print as they arrive in both modes, for
the whole session - there's no toggle to turn them off.

No dependency beyond the core library itself (no ``prompt_toolkit`` or
similar), so importing this - and therefore ``pyippdme`` itself - never
pulls in more than a library user actually needs. Line editing comes for
free from the standard library's ``readline`` module in interactive mode,
when it's available (POSIX; imported defensively, so its absence on
Windows is a graceful degradation, not an error - ``input()`` still works
everywhere, and Windows' own console host already provides *within-
session* up-arrow recall independent of ``readline``, so the only thing
actually lost there is history persisting *across* separate runs).
Reading input runs in a worker thread (:func:`asyncio.to_thread`) so the
event loop - and with it, unsolicited-event printing - keeps running
while a prompt is up.

The richer, full-featured interactive experience is
:mod:`pyippdme.cli.tui` (``ippdme tui``); this is intentionally not that.
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TextIO

try:
    import readline
except ImportError:  # pragma: no cover - platforms without readline (e.g. Windows)
    readline = None  # type: ignore[assignment]

from pyippdme.cli.script import describe_address, run_line, run_script_lines, start_embedded_server
from pyippdme.cli.session_log import SessionLog
from pyippdme.client import IppDmeClient
from pyippdme.protocol.network import TCP_NETWORK
from pyippdme.server import IppDmeServer
from pyippdme.simulation.state import SimulationState

_HISTORY_FILE = Path.home() / ".ippdme_history"
_HISTORY_LENGTH = 1000


async def run(
    host: str | None,
    port: int,
    *,
    virtual: bool = False,
    script: Path | None = None,
    session_log: str | None = None,
    commands_file: str | None = None,
) -> None:
    """Connect once, then either run ``script`` non-interactively or prompt for commands.

    ``session_log`` records every wire line sent/received, plus connect/
    disconnect, in the unified format all of ``client``/``serve``/``spy``
    share; see :mod:`pyippdme.cli.session_log`. ``commands_file`` records
    just the commands actually run (typed interactively, or read from
    ``script``), one per line, in exactly the syntax this client accepts -
    so the session can be replayed later with ``--file``.
    """
    embedded_server: IppDmeServer[SimulationState] | None = None
    if virtual:
        host, port, embedded_server = await start_embedded_server()
        print("Started an in-process VirtualCMM")
    assert host is not None  # noqa: S101 (argparse requires host unless --virtual)

    peer = describe_address(host, port)
    with contextlib.ExitStack() as files:
        log = SessionLog(
            files.enter_context(Path(session_log).open("w", buffering=1))
            if session_log is not None
            else None
        )
        client = await IppDmeClient.connect(
            host,
            port,
            network=TCP_NETWORK if embedded_server is None else embedded_server.network,
            on_line_sent=log.to_server,
            on_line_received=log.to_client,
        )
        log.connected(peer)
        print(f"Connected to {peer}")
        events_task = asyncio.create_task(_print_events(client))
        try:
            commands_handle = (
                files.enter_context(Path(commands_file).open("w", buffering=1))
                if commands_file is not None
                else None
            )
            on_command = None if commands_handle is None else _make_recorder(commands_handle)

            if script is not None:
                await run_script_lines(client, script, sys.stdout, on_command=on_command)
            else:
                await _interactive_loop(client, "> ", sys.stdout, on_command)
        finally:
            events_task.cancel()
            await client.close()
            log.disconnected(peer)
            if embedded_server is not None:
                await embedded_server.close()


def _make_recorder(commands_handle: TextIO) -> Callable[[str], None]:
    def record(text: str) -> None:
        print(text, file=commands_handle)

    return record


async def _print_events(client: IppDmeClient) -> None:
    async for event in client.unsolicited_events():
        print(f"[event] {event.tag.to_wire()} # {event.data.to_wire()}")


async def _interactive_loop(
    client: IppDmeClient,
    prompt: str,
    output: TextIO,
    on_command: Callable[[str], None] | None,
) -> None:
    if readline is not None:
        readline.set_history_length(_HISTORY_LENGTH)
        with contextlib.suppress(OSError):
            readline.read_history_file(_HISTORY_FILE)
    try:
        while True:
            try:
                line = await asyncio.to_thread(input, prompt)
            except (EOFError, KeyboardInterrupt):
                print()
                return
            line = line.strip()
            if not line:
                continue
            if on_command is not None:
                on_command(line)
            if not await run_line(client, line, output):
                return  # connection lost - nothing further could succeed
    finally:
        if readline is not None:
            with contextlib.suppress(OSError):
                readline.write_history_file(_HISTORY_FILE)
