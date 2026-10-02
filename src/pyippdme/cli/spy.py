# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""A transparent proxy between a client and a real server: ``ippdme spy``.

A thin console-logging (and, optionally, file-logging) consumer of
:class:`~pyippdme.spy.Spy` - see that module for what actually relays
traffic and why it never decodes a line through the protocol parser. This
module only adds the parts specific to being a CLI command: printing each
relayed message and connection event to the console with a timestamp, and
optionally also recording traffic to files - ``--session-log`` in the
unified format all of ``client``/``serve``/``spy`` share (see
:mod:`pyippdme.cli.session_log`), ``--commands-file`` in this project's
bare, tag-stripped, replayable form.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from pathlib import Path
from typing import TextIO

from pyippdme.cli.session_log import SessionLog, strip_tag
from pyippdme.spy import Spy, SpyDirection, SpyMessage

_DIRECTION_MARKER = {
    SpyDirection.TO_SERVER: ">",
    SpyDirection.TO_CLIENT: "<",
}


def _decoded_text(message: SpyMessage) -> str:
    return message.line.decode("ascii", errors="replace").rstrip("\r\n")


def _print_message(message: SpyMessage) -> None:
    marker = _DIRECTION_MARKER[message.direction]
    timestamp = message.timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"{timestamp} [{message.connection_id}] {marker} {_decoded_text(message)}")


def _print_connect(connection_id: int, peer: str) -> None:
    print(f"[{connection_id}] connected from {peer}")


def _print_disconnect(connection_id: int, peer: str) -> None:
    print(f"[{connection_id}] disconnected ({peer})")


def _build_message_hook(
    log: SessionLog, commands_file: TextIO | None
) -> Callable[[SpyMessage], None]:
    def on_message(message: SpyMessage) -> None:
        _print_message(message)
        decoded = _decoded_text(message)
        if message.direction == SpyDirection.TO_SERVER:
            log.to_server(decoded)
            # Tag-stripped unconditionally: meant to be replayed by
            # `ippdme client --file`, which expects a bare method call per
            # line (or now accepts, and ignores, a leading tag - see
            # cli.script's own module docstring - but stripping here keeps
            # this file's own format the simplest possible one).
            if commands_file is not None:
                print(strip_tag(decoded), file=commands_file)
        else:
            log.to_client(decoded)

    return on_message


async def run(
    host: str,
    port: int,
    listen_host: str,
    listen_port: int,
    *,
    session_log: str | None = None,
    commands_file: str | None = None,
) -> None:
    """Start a spy forwarding to ``host``:``port``, listening on ``listen_host``:``listen_port``.

    ``session_log`` records every relayed message (both directions,
    tags included) plus connect/disconnect, in the unified format; see
    :mod:`pyippdme.cli.session_log`. ``commands_file`` (client-to-server
    messages only, tag-stripped, one per line) is written in exactly the
    format ``ippdme client --file`` expects, so a captured session can be
    replayed later that way.
    """
    with contextlib.ExitStack() as files:
        log = SessionLog(
            files.enter_context(Path(session_log).open("w", buffering=1))
            if session_log is not None
            else None
        )
        commands = _open_if_given(files, commands_file)

        def on_connect(connection_id: int, peer: str) -> None:
            _print_connect(connection_id, peer)
            log.connected(peer)

        def on_disconnect(connection_id: int, peer: str) -> None:
            _print_disconnect(connection_id, peer)
            log.disconnected(peer)

        spy = Spy(
            host,
            port,
            on_message=_build_message_hook(log, commands),
            on_connect=on_connect,
            on_disconnect=on_disconnect,
        )
        print(f"pyippdme spy listening on {listen_host}:{listen_port}, forwarding to {host}:{port}")
        await spy.serve_forever(listen_host, listen_port)


def _open_if_given(files: contextlib.ExitStack, path: str | None) -> TextIO | None:
    if path is None:
        return None
    return files.enter_context(Path(path).open("w", buffering=1))
