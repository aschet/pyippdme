# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Shared "run one typed-in command line" logic for the client and TUI.

Both front-ends do the exact same thing when the user types a command:
parse it, send it, and report the Ack/Data/Done/Error/connection-loss
sequence - they differ only in *how* each of those is displayed (plain
``print()`` text for the client, Rich markup written to a scrolling log
widget for the TUI; see :mod:`pyippdme.cli.client`/:mod:`pyippdme.cli.tui`).
This module owns the interaction itself and yields a typed
:data:`CommandEvent` for each step, leaving formatting entirely to the
caller, rather than each front-end re-implementing the send/await/format
sequence with its own hard-coded output calls.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass

from pyippdme.client import IppDmeClient
from pyippdme.exceptions import IppDmeConnectionError, IppDmeError, IppDmeServerError
from pyippdme.protocol.ast import DataPayload
from pyippdme.protocol.commands import CommandName
from pyippdme.protocol.errors import ServerError, describe
from pyippdme.protocol.parser import parse_method

#: Commands that start a daemon (5.5.2) and so must be sent
#: tagged with an :class:`~pyippdme.protocol.ast.EventTag`, not a plain
#: numbered ``Tag`` - this library only ever implements one (see
#: :mod:`pyippdme.simulation.classes.mover_class`'s module docstring). The wire
#: grammar itself allows either tag kind for any command, so this can't be
#: inferred from the name alone; it's protocol/library-specific knowledge,
#: hardcoded here rather than derived from anything.
_DAEMON_STARTING_COMMANDS = frozenset({CommandName.ON_MOVE_REPORT, CommandName.ON_MOVE_REPORT_E})


@dataclass(frozen=True, slots=True)
class ParseFailed:
    """``text`` was not a valid method call; nothing was sent."""

    error: IppDmeError


@dataclass(frozen=True, slots=True)
class Acked:
    """The server accepted the command (``&``)."""


@dataclass(frozen=True, slots=True)
class Received:
    """One data response line (``#``) arrived."""

    payload: DataPayload


@dataclass(frozen=True, slots=True)
class Completed:
    """The transaction finished successfully (``%``)."""


@dataclass(frozen=True, slots=True)
class Failed:
    """The server reported an error (``!Error(...)``); the transaction is done."""

    error: ServerError


@dataclass(frozen=True, slots=True)
class ConnectionLost:
    """The connection dropped before the transaction finished."""

    error: IppDmeConnectionError


#: One step of running a single command line; see :func:`run_command_line`.
CommandEvent = ParseFailed | Acked | Received | Completed | Failed | ConnectionLost


def format_error(error: ServerError) -> str:
    """Render a :class:`~pyippdme.protocol.errors.ServerError` as one plain-text line."""
    description = describe(error.number)
    suffix = f" ({description[1]})" if description else ""
    return f"! Error({int(error.severity)},{error.number},{error.cause!r},{error.text!r}){suffix}"


async def run_command_line(client: IppDmeClient, text: str) -> AsyncIterator[CommandEvent]:
    """Parse, send, and await one command line, yielding each event as it happens.

    A recognized daemon-starting command (:data:`_DAEMON_STARTING_COMMANDS`)
    is sent tagged with a fresh ``EventTag`` instead of the usual numbered
    ``Tag`` - without this, e.g. ``OnMoveReportE(...)`` would be rejected
    outright (a daemon needs an ``EventTag``, 5.5.2), and a later
    ``StopDaemon(<that tag>)`` in the same live session would have nothing
    to reference.

    This lets a *live* session start and stop a daemon at all; it does
    **not** make a captured commands file's ``StopDaemon(E0047)`` reliably
    replayable later. ``EventTag`` and ``Tag`` are independent counters
    (5.4.1); a fresh replay only reallocates the same ``E0047`` if it
    re-runs the exact same daemon-starting commands in the exact same
    order the original session did. A file captured from a *different*
    client (e.g. via :mod:`pyippdme.spy`) may not follow this project's own
    :meth:`~pyippdme.client.IppDmeClient.allocate_event_tag` scheme at all,
    and even a same-client capture breaks if the file is edited, truncated,
    or replayed out of order - none of which the format can detect. Replay
    in general also assumes the server responds identically each run,
    which holds for this library's own simulated servers but not for real
    hardware, where actual contact points, timing, and error conditions
    vary run to run. Both are limitations of any commands-file format,
    regardless of implementation.
    """
    try:
        method = parse_method(text)
    except IppDmeError as exc:
        yield ParseFailed(exc)
        return

    tag = client.allocate_event_tag() if method.name in _DAEMON_STARTING_COMMANDS else None
    txn = client.send(method.name, *method.args, tag=tag)
    await client._transport.drain()
    try:
        await txn.wait_ack()
        yield Acked()
        data = await txn.wait_complete()
        for item in data:
            yield Received(item)
        yield Completed()
    except IppDmeServerError as exc:
        yield Failed(exc.error)
    except IppDmeConnectionError as exc:
        yield ConnectionLost(exc)
