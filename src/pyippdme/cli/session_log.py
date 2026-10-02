# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""One unified wire-traffic log format, shared by ``ippdme client``/``serve``/``spy``.

Every ``--session-log`` file uses exactly the same format, regardless of
which subcommand wrote it. One line per event, always timestamped:

* ``<timestamp> > <tag> <method>(...)`` - a line traveling client -> server.
* ``<timestamp> < <tag> <response>`` - a line traveling server -> client.
* ``<timestamp> CONNECTED <peer>`` / ``<timestamp> DISCONNECTED <peer>``.

Direction is always relative to the underlying protocol traffic (client ->
server vs. server -> client), never to which side the current process
happens to be playing. So ``ippdme serve``'s session log uses ``>`` for
what it *receives* (a client's commands) and ``<`` for what it *sends*
(its own responses) - the opposite of which of its own hooks
(``on_line_received``/``on_line_sent``) that corresponds to - so that the
same line, read out of context, always means the same thing whether it
came from ``ippdme client`` or ``ippdme serve`` sitting on the other end
of the same wire. Per 5.8, only one client is ever connected to
a server at a time (a server "available for use by other clients" is
described as something that happens only *after* the current client
disconnects, "the next client" - never concurrently) - so unlike
:class:`~pyippdme.spy.Spy`'s own hooks, none of this needs a connection
identifier to disambiguate whose traffic a line belongs to.
"""

from __future__ import annotations

from datetime import datetime
from typing import TextIO


class SessionLog:
    """Writes one ``--session-log`` file; safe to use with no destination (a no-op then)."""

    def __init__(self, handle: TextIO | None) -> None:
        self._handle = handle

    def to_server(self, text: str) -> None:
        """Record one line that traveled client -> server (a command)."""
        self._write(">", text)

    def to_client(self, text: str) -> None:
        """Record one line that traveled server -> client (a response)."""
        self._write("<", text)

    def connected(self, peer: str) -> None:
        self._event("CONNECTED", peer)

    def disconnected(self, peer: str) -> None:
        self._event("DISCONNECTED", peer)

    def _write(self, marker: str, text: str) -> None:
        if self._handle is None:
            return
        print(f"{_timestamp()} {marker} {text}", file=self._handle)

    def _event(self, label: str, peer: str) -> None:
        if self._handle is None:
            return
        print(f"{_timestamp()} {label} {peer}", file=self._handle)


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def strip_tag(text: str) -> str:
    """Drop the leading ``<tag> `` every wire line starts with.

    Every command/response/event line is ``tag SP rest CRLF`` (5.2) - a
    plain split on the first space, not a re-parse, since the
    point of capturing traffic is to tolerate whatever was actually sent,
    not to additionally require it satisfy the strict grammar
    :mod:`pyippdme.protocol.parser` enforces. Unlike
    :mod:`pyippdme.cli.script`'s ``_strip_leading_tag``, this assumes a tag
    is always present (real captured traffic), not merely optional.
    """
    _tag, _sep, rest = text.partition(" ")
    return rest
