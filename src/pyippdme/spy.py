# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

r"""A transparent I++ DME proxy (``ippdme spy``'s underlying implementation).

:class:`Spy` binds a listening socket and, for each incoming client
connection, opens a new outbound connection to the real server and relays
bytes in both directions unchanged - it never decodes a line through
:mod:`pyippdme.protocol.parser`/re-encodes it, deliberately, so nothing here
can normalize or subtly alter what either side actually sent (a
byte-for-byte relay is what a spy is for; round-tripping through
the AST would only be able to approximate that). Line boundaries (``\r\n``,
per 5.1) are still respected, since that is what lets each
message be reported to :attr:`on_message` as a discrete event
rather than an arbitrary chunk of bytes.

:attr:`on_message`/:attr:`on_connect`/:attr:`on_disconnect` are
observe-only: they are told what was relayed, but cannot change, delay, or
block it. To filter, alter, delay or multiply messages, pass an ``intercept``
function (:data:`Interceptor`); without one, every line is forwarded
unchanged. ``ippdme spy`` (:mod:`pyippdme.cli.spy`) is a
thin console-logging consumer of this same class, not a separate
implementation - embedding a spy in your own tooling (recording a session,
feeding traffic to an analyzer, rewriting commands, ...) uses exactly this API.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from enum import StrEnum

from pyippdme.protocol.network import (
    TCP_NETWORK,
    Listener,
    Network,
    StreamReaderLike,
    StreamWriterLike,
)
from pyippdme.protocol.transport import DEFAULT_PORT, READ_LIMIT

logger = logging.getLogger("pyippdme.spy")


class SpyDirection(StrEnum):
    """Which way a relayed message travelled."""

    TO_SERVER = "to_server"
    TO_CLIENT = "to_client"


class SpyOutcome(StrEnum):
    """What a :class:`Spy` did with a received line."""

    FORWARDED = "forwarded"
    ALTERED = "altered"
    DROPPED = "dropped"


@dataclass(frozen=True, slots=True)
class SpyMessage:
    r"""One line received by :class:`Spy` (including its trailing ``\r\n``).

    ``line`` is the line as it arrived. ``forwarded`` is what was sent on in
    its place: ``None`` if it was forwarded unchanged, otherwise the lines an
    ``intercept`` function returned (empty if the message was dropped).
    """

    connection_id: int
    direction: SpyDirection
    timestamp: datetime
    line: bytes
    forwarded: tuple[bytes, ...] | None = None

    @property
    def outcome(self) -> SpyOutcome:
        if self.forwarded is None:
            return SpyOutcome.FORWARDED
        if not self.forwarded:
            return SpyOutcome.DROPPED
        return SpyOutcome.ALTERED


#: A hook is a plain, synchronous callable - matching this project's existing
#: convention for lightweight observation points (see
#: :data:`~pyippdme.server.registry.PropertySetHandler`). One that needs to do
#: async work of its own (write to a file, push to a queue, ...) should spawn
#: its own ``asyncio.create_task(...)`` rather than this class supporting a
#: parallel async-callable signature for the same purpose. A hook that
#: raises is logged and otherwise ignored - a bug in observation code must
#: not take down the relay itself.
MessageHook = Callable[[SpyMessage], None]
ConnectionHook = Callable[[int, str], None]

#: Decides what to forward in place of a received line. It is awaited before
#: anything is sent on, so it may also delay a message. Return the same bytes
#: to forward unchanged, other bytes to alter the message, ``None``, empty bytes
#: or an empty sequence to drop it, or several lines to replace it with more than
#: one. Returned lines should end in ``\r\n`` (5.1). Lines of one direction
#: are processed one at a time, in order. Dropping or altering a command
#: can leave the client waiting for a response the server never sends; the
#: spy does not invent one. An interceptor that raises is logged and the
#: connection is closed.
Interceptor = Callable[[SpyMessage], Awaitable[bytes | Sequence[bytes] | None]]


def _call_hook(hook: Callable[..., None] | None, *args: object) -> None:
    if hook is None:
        return
    try:
        hook(*args)
    except Exception:
        logger.exception("Unhandled exception in a Spy hook")


class Spy:
    """Relay every byte between a client and the real server at ``host``:``port``.

    See the module docstring for what "unchanged" and "observe-only" mean
    here. Method names/shapes deliberately match
    :class:`~pyippdme.server.IppDmeServer`'s own (:meth:`start`/
    :meth:`serve_forever`/:meth:`close`/:attr:`port`), so embedding a spy
    alongside other asyncio work looks the same as embedding a server::

        spy = Spy("192.168.1.50", 1294, on_message=lambda m: print(m))
        port = await spy.start("127.0.0.1", 0)
        ...
        await spy.close()
    """

    def __init__(
        self,
        host: str,
        port: int,
        *,
        network: Network = TCP_NETWORK,
        intercept: Interceptor | None = None,
        on_message: MessageHook | None = None,
        on_connect: ConnectionHook | None = None,
        on_disconnect: ConnectionHook | None = None,
    ) -> None:
        self._network = network
        self._intercept = intercept
        self._host = host
        self._port = port
        self._on_message = on_message
        self._on_connect = on_connect
        self._on_disconnect = on_disconnect
        self._listener: Listener | None = None
        self._next_connection_id = 1

    async def start(self, host: str = "127.0.0.1", port: int = 0) -> int:
        """Bind and start accepting connections without blocking; return the bound port.

        ``port=0`` lets the OS pick a free port, returned here so the
        caller can connect to it (embedding a spy in a test, say). Call
        :meth:`close` when done.
        """
        self._listener = await self._network.start_server(
            self._on_client, host, port, limit=READ_LIMIT
        )
        return self._listener.port

    @property
    def port(self) -> int | None:
        """The port currently bound by :meth:`start`/:meth:`serve_forever`, or ``None``."""
        if self._listener is None:
            return None
        return self._listener.port

    async def serve_forever(
        self,
        host: str = "0.0.0.0",  # noqa: S104 # nosec B104
        port: int = DEFAULT_PORT,
    ) -> None:
        await self.start(host, port)
        assert self._listener is not None  # noqa: S101 (set by start() just above)
        await self._listener.serve_forever()

    async def close(self) -> None:
        if self._listener is not None:
            await self._listener.close()
            self._listener = None

    async def _on_client(self, reader: StreamReaderLike, writer: StreamWriterLike) -> None:
        connection_id = self._next_connection_id
        self._next_connection_id += 1
        client_peer = _peer(writer)

        try:
            upstream_reader, upstream_writer = await self._network.open_connection(
                self._host, self._port, limit=READ_LIMIT
            )
        except OSError:
            logger.exception(
                "Spy connection %d: could not reach %s:%d", connection_id, self._host, self._port
            )
            await _close(writer)
            return

        _call_hook(self._on_connect, connection_id, client_peer)
        to_server = asyncio.create_task(
            self._pump(reader, upstream_writer, connection_id, SpyDirection.TO_SERVER)
        )
        to_client = asyncio.create_task(
            self._pump(upstream_reader, writer, connection_id, SpyDirection.TO_CLIENT)
        )
        _done, pending = await asyncio.wait(
            {to_server, to_client}, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        for task in pending:
            with contextlib.suppress(asyncio.CancelledError):
                await task

        await _close(writer)
        await _close(upstream_writer)
        _call_hook(self._on_disconnect, connection_id, client_peer)

    async def _pump(
        self,
        reader: StreamReaderLike,
        writer: StreamWriterLike,
        connection_id: int,
        direction: SpyDirection,
    ) -> None:
        try:
            while True:
                try:
                    line = await reader.readuntil(b"\r\n")
                except asyncio.IncompleteReadError:
                    return  # connection closed, possibly mid-line - nothing further to relay
                except asyncio.LimitOverrunError:
                    logger.error(
                        "Spy connection %d: a line exceeded the buffer size, stopping this side",
                        connection_id,
                    )
                    return
                message = SpyMessage(connection_id, direction, datetime.now(), line)
                if self._intercept is not None:
                    try:
                        result = await self._intercept(message)
                    except Exception:
                        logger.exception(
                            "Spy connection %d: an interceptor raised, closing the connection",
                            connection_id,
                        )
                        return
                    lines = _as_lines(result)
                    if lines != (line,):
                        message = replace(message, forwarded=lines)
                for out in message.forwarded if message.forwarded is not None else (line,):
                    writer.write(out)
                await writer.drain()
                _call_hook(self._on_message, message)
        except (ConnectionError, OSError):
            return


def _as_lines(result: bytes | Sequence[bytes] | None) -> tuple[bytes, ...]:
    if result is None:
        return ()
    if isinstance(result, bytes):
        return (result,) if result else ()
    return tuple(line for line in result if line)


async def _close(writer: StreamWriterLike) -> None:
    writer.close()
    with contextlib.suppress(OSError):
        await writer.wait_closed()


def _peer(writer: StreamWriterLike) -> str:
    peername = writer.get_extra_info("peername")
    if not peername:
        return "<unknown>"
    host, port = peername[0], peername[1]
    return f"{host}:{port}"
