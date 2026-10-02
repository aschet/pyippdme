# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

r"""A transparent I++ DME proxy (``ippdme spy``'s underlying implementation).

:class:`Spy` binds a listening socket and, for each incoming client
connection, opens a new outbound connection to the real server and relays
bytes in both directions unchanged - it never decodes a line through
:mod:`pyippdme.protocol.parser`/re-encodes it, deliberately, so nothing here
can normalize or subtly alter what either side actually sent (a
byte-for-byte relay is the whole point of a spy; round-tripping through
the AST would only be able to approximate that). Line boundaries (``\r\n``,
per 5.1) are still respected, since that is what lets each
forwarded message be reported to :attr:`on_message` as a discrete event
rather than an arbitrary chunk of bytes.

:attr:`on_message`/:attr:`on_connect`/:attr:`on_disconnect` are
observe-only: they are told what was relayed, but cannot change, delay, or
block it - forwarding unchanged is a hard invariant of this class, not
something a hook can opt out of. ``ippdme spy`` (:mod:`pyippdme.cli.spy`) is a
thin console-logging consumer of this same class, not a separate
implementation - embedding a spy in your own tooling (recording a session,
feeding traffic to an analyzer, ...) uses exactly this API.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from pyippdme.protocol.transport import DEFAULT_PORT, READ_LIMIT

logger = logging.getLogger("pyippdme.spy")


class SpyDirection(StrEnum):
    """Which way a relayed message travelled."""

    TO_SERVER = "to_server"
    TO_CLIENT = "to_client"


@dataclass(frozen=True, slots=True)
class SpyMessage:
    r"""One line relayed by :class:`Spy`, exactly as forwarded (including its trailing ``\r\n``)."""

    connection_id: int
    direction: SpyDirection
    timestamp: datetime
    line: bytes


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
        on_message: MessageHook | None = None,
        on_connect: ConnectionHook | None = None,
        on_disconnect: ConnectionHook | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._on_message = on_message
        self._on_connect = on_connect
        self._on_disconnect = on_disconnect
        self._asyncio_server: asyncio.Server | None = None
        self._next_connection_id = 1

    async def start(self, host: str = "127.0.0.1", port: int = 0) -> int:
        """Bind and start accepting connections without blocking; return the bound port.

        ``port=0`` lets the OS pick a free port, returned here so the
        caller can connect to it (embedding a spy in a test, say). Call
        :meth:`close` when done.
        """
        self._asyncio_server = await asyncio.start_server(
            self._on_client, host, port, limit=READ_LIMIT
        )
        return int(self._asyncio_server.sockets[0].getsockname()[1])

    @property
    def port(self) -> int | None:
        """The port currently bound by :meth:`start`/:meth:`serve_forever`, or ``None``."""
        if self._asyncio_server is None:
            return None
        return int(self._asyncio_server.sockets[0].getsockname()[1])

    async def serve_forever(
        self,
        host: str = "0.0.0.0",  # noqa: S104 # nosec B104
        port: int = DEFAULT_PORT,
    ) -> None:
        await self.start(host, port)
        assert self._asyncio_server is not None  # noqa: S101 (set by start() just above)
        async with self._asyncio_server:
            await self._asyncio_server.serve_forever()

    async def close(self) -> None:
        if self._asyncio_server is not None:
            self._asyncio_server.close()
            await self._asyncio_server.wait_closed()
            self._asyncio_server = None

    async def _on_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        connection_id = self._next_connection_id
        self._next_connection_id += 1
        client_peer = _peer(writer)

        try:
            upstream_reader, upstream_writer = await asyncio.open_connection(
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
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
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
                writer.write(line)
                await writer.drain()
                _call_hook(
                    self._on_message, SpyMessage(connection_id, direction, datetime.now(), line)
                )
        except (ConnectionError, OSError):
            return


async def _close(writer: asyncio.StreamWriter) -> None:
    writer.close()
    with contextlib.suppress(OSError):
        await writer.wait_closed()


def _peer(writer: asyncio.StreamWriter) -> str:
    peername = writer.get_extra_info("peername")
    if not peername:
        return "<unknown>"
    host, port = peername[0], peername[1]
    return f"{host}:{port}"
