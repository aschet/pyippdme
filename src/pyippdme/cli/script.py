# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

r"""Shared engine for running a file of commands: parsing, execution, transcript.

Used by ``ippdme client --file`` (:mod:`pyippdme.cli.client`) to run a
script of commands non-interactively through an already-open connection,
and by that same module's ``--virtual`` flag (:func:`start_embedded_server`)
to do so against an in-process ``VirtualCMM`` instead of a real host/port.

Inspired by the NIST/I++ DME reference test suite's paired ``.prg``
(commands)/``.res`` (responses)/``.txt`` (description) files - a
repeatable, scriptable way to exercise a server without typing commands
into the client by hand (see :mod:`pyippdme.simulation.classes.scanning_class`'s module
docstring for this project's general stance on that suite: useful for real
captured wire syntax, never as behavioral ground truth). This project's own
version deliberately isn't byte-compatible with that format - no fixed
5-digit tag column, no ``\\`` line separators, no paired second file - since
:class:`~pyippdme.client.IppDmeClient` already allocates tags itself and a
single interleaved transcript (command, then its response, in reading
order) is easier for a person to follow than cross-referencing two files
by tag number.

A script is a plain text file, one command per line, in the same syntax
the client accepts (``StartSession()``, ``GoTo(X(10), Y(20))``, ...). Blank
lines and ``#``-prefixed comment lines are allowed and are copied into the
transcript verbatim, so a single file documents both what was run and what
happened - the ``.txt`` description NIST keeps separate.

A line may also carry a leading wire tag (``00047 GoTo(...)``, ``E0001
OnMoveReportE(...)``) - e.g. a genuine NIST-style ``.prg`` file's fixed
tag column, or any other tag-prefixed transcript - rather than this
project's own bare ``MethodName(...)`` form; a leading tag is simply
ignored if present (see :func:`_strip_leading_tag`) instead of being
required to be stripped first. The connection's own client re-allocates a
fresh tag when it actually sends the command regardless, so the original
tag's value is never used for anything.

Limitation: a script that starts a daemon (``OnMoveReport``/
``OnMoveReportE``, see :mod:`pyippdme.cli._interaction`) and later
``StopDaemon(<tag>)``s it is only reliably replayable if it's run in full,
in its original order, by this same client implementation - the ``tag``
baked into that ``StopDaemon`` call is a snapshot of whichever
``EventTag`` the *original* run happened to allocate, and a fresh run
only reallocates the same value by re-running the identical sequence of
daemon-starting commands (``EventTag``/``Tag`` are independent counters,
5.4.1). A script captured from a different client (e.g. via
:mod:`pyippdme.spy`) isn't guaranteed to follow this project's own
allocation scheme at all, and editing, truncating, or partially replaying
such a script silently breaks the correspondence. Replay in general also
assumes the server answers identically each run, which holds for this
library's own simulated servers but not for real hardware. Neither is
something this module (or any commands-file format) can detect or fix.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import TextIO

from pyippdme.cli._interaction import (
    Acked,
    Completed,
    ConnectionLost,
    Failed,
    ParseFailed,
    Received,
    format_error,
    run_command_line,
)
from pyippdme.client import IppDmeClient
from pyippdme.server import IppDmeServer
from pyippdme.simulation.state import SimulationState

#: Lines starting with this (after stripping leading whitespace) are
#: comments: copied into the transcript verbatim, never sent as commands.
COMMENT_PREFIX = "#"

#: A wire tag (5.4.1: a 5-digit ``Tag``, or ``E`` + 4 digits for
#: an ``EventTag``) followed by whitespace, at the very start of a line.
#: No built-in command name matches either shape (a real command name
#: must start with a letter per the grammar, and none of this project's
#: happen to look like ``E0001``), so stripping this is unambiguous.
_LEADING_TAG_RE = re.compile(r"^(?:\d{5}|E\d{4})\s+")


def _strip_leading_tag(text: str) -> str:
    """Drop a leading wire tag from ``text``, if it has one; otherwise return it unchanged."""
    return _LEADING_TAG_RE.sub("", text, count=1)


async def start_embedded_server() -> tuple[str, int, IppDmeServer[SimulationState]]:
    """Start an in-process VirtualCMM; return ``(host, port, server)`` to connect to it.

    Used by ``ippdme client --virtual`` (see :mod:`pyippdme.cli.client`);
    ``ippdme tui --virtual`` has its own separate copy of this (see
    :mod:`pyippdme.cli.tui`), since the TUI's app-lifecycle logging makes
    sharing this one awkward, not because the logic itself differs.
    """
    from pyippdme.simulation.virtual_cmm import VirtualCMM

    server = VirtualCMM()
    port = await server.start()
    return "127.0.0.1", port, server


async def run_script_lines(
    client: IppDmeClient,
    script_path: Path,
    output: TextIO,
    *,
    on_command: Callable[[str], None] | None = None,
) -> bool:
    """Run every line of ``script_path`` through an already-connected ``client``.

    Used by ``ippdme client --file`` (:func:`pyippdme.cli.client.run`) to
    run a script through the connection that entry point already opened.
    Comment/blank lines are echoed as-is. A parse error or a server error
    is written to the transcript and execution continues with the next
    line (mirroring what a NIST-style ``.res`` file records: everything
    that happened, including failures); returns ``False`` if the
    connection was lost partway through, since nothing further could
    succeed.

    ``on_command``, if given, is called with each command's own text
    (comments/blank lines excluded) right before it runs - used by
    ``ippdme client``'s ``--commands-file`` to record every command actually
    run, however it was reached (typed interactively or read from a
    script), as a file replayable later the same way.
    """
    for line in script_path.read_text().splitlines():
        text = line.strip()
        if not text or text.startswith(COMMENT_PREFIX):
            print(line, file=output)
            continue
        text = _strip_leading_tag(text)
        print(f"> {text}", file=output)
        if on_command is not None:
            on_command(text)
        if not await run_line(client, text, output):
            return False
    return True


async def run_line(client: IppDmeClient, text: str, output: TextIO) -> bool:
    """Run one command line, writing its transcript. Returns False if the connection died.

    The shared per-line engine behind both :func:`run_script_lines` and
    :mod:`pyippdme.cli.client`'s interactive prompt - one place that knows
    how to turn a typed/scripted line into Ack/Data/Done/Error output.
    """
    async for event in run_command_line(client, text):
        match event:
            case ParseFailed(error):
                print(f"Parse error: {error}", file=output)
            case Acked():
                print("&", file=output)
            case Received(payload):
                print(f"# {payload.to_wire()}", file=output)
            case Completed():
                print("%", file=output)
            case Failed(error):
                print(format_error(error), file=output)
            case ConnectionLost(error):
                print(f"Connection lost: {error}", file=output)
                return False
    return True
