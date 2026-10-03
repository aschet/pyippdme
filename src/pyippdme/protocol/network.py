# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Where connections come from: TCP, or an in-process network.

The I++ DME protocol itself runs over TCP (5.1). Everything in this package
that opens or accepts a connection (:class:`~pyippdme.client.IppDmeClient`,
:class:`~pyippdme.server.IppDmeServer`, :class:`~pyippdme.spy.Spy`, the raw-data
binary socket) does so through a :class:`Network`, which mirrors
:func:`asyncio.open_connection` and :func:`asyncio.start_server`.

:data:`TCP_NETWORK` is the default. A :class:`MemoryNetwork` keeps every
connection inside the process, so a server and its clients can talk without
opening a port::

    network = MemoryNetwork()
    server = VirtualCMM(network=network)
    port = await server.start()
    client = await IppDmeClient.connect("virtual", port, network=network)

Host names are ignored by a :class:`MemoryNetwork`; connections are matched by
port only.
"""

from __future__ import annotations

import asyncio
import errno
from collections.abc import Awaitable, Callable
from typing import Any, Protocol, TypeAlias


class StreamReaderLike(Protocol):
    """The part of :class:`asyncio.StreamReader` this package uses."""

    async def readuntil(self, _separator: bytes = b"\n", /) -> bytes: ...

    async def read(self, _n: int = -1, /) -> bytes: ...


class StreamWriterLike(Protocol):
    """The part of :class:`asyncio.StreamWriter` this package uses."""

    def write(self, data: bytes) -> None: ...

    async def drain(self) -> None: ...

    def close(self) -> None: ...

    async def wait_closed(self) -> None: ...

    def get_extra_info(self, name: str, default: Any = None) -> Any: ...


ClientHandler: TypeAlias = Callable[[StreamReaderLike, StreamWriterLike], Awaitable[None]]

#: asyncio's own default stream buffer size.
DEFAULT_LIMIT = 2**16


class Listener(Protocol):
    """A bound server endpoint returned by :meth:`Network.start_server`."""

    @property
    def port(self) -> int: ...

    async def serve_forever(self) -> None: ...

    async def close(self) -> None: ...


class Network(Protocol):
    """Opens outgoing and accepts incoming byte-stream connections."""

    async def open_connection(
        self, host: str, port: int, *, limit: int = DEFAULT_LIMIT
    ) -> tuple[StreamReaderLike, StreamWriterLike]:
        """Connect to ``host``:``port``; raise :class:`OSError` if nothing is listening."""
        ...

    async def start_server(
        self, on_client: ClientHandler, host: str, port: int, *, limit: int = DEFAULT_LIMIT
    ) -> Listener:
        """Listen on ``host``:``port``; ``port=0`` picks a free one (see :attr:`Listener.port`)."""
        ...


class _TcpListener:
    def __init__(self, server: asyncio.Server) -> None:
        self._server = server

    @property
    def port(self) -> int:
        return int(self._server.sockets[0].getsockname()[1])

    async def serve_forever(self) -> None:
        async with self._server:
            await self._server.serve_forever()

    async def close(self) -> None:
        self._server.close()
        await self._server.wait_closed()


class TcpNetwork:
    """Real TCP sockets, via asyncio."""

    async def open_connection(
        self, host: str, port: int, *, limit: int = DEFAULT_LIMIT
    ) -> tuple[StreamReaderLike, StreamWriterLike]:
        return await asyncio.open_connection(host, port, limit=limit)

    async def start_server(
        self, on_client: ClientHandler, host: str, port: int, *, limit: int = DEFAULT_LIMIT
    ) -> Listener:
        return _TcpListener(await asyncio.start_server(on_client, host, port, limit=limit))


TCP_NETWORK: Network = TcpNetwork()


class _MemoryWriter:
    def __init__(self, peer_reader: asyncio.StreamReader, peername: tuple[str, int]) -> None:
        self._peer_reader = peer_reader
        self._peername = peername
        self._peer: _MemoryWriter | None = None
        self._closed = False

    def write(self, data: bytes) -> None:
        if self._closed or self._peer is None or self._peer._closed:
            return
        self._peer_reader.feed_data(bytes(data))

    async def drain(self) -> None:
        if self._closed:
            raise ConnectionResetError("Connection closed")
        await asyncio.sleep(0)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._peer_reader.feed_eof()

    async def wait_closed(self) -> None:
        return None

    def get_extra_info(self, name: str, default: Any = None) -> Any:
        if name == "peername":
            return self._peername
        return default


class _MemoryListener:
    def __init__(
        self, network: MemoryNetwork, port: int, on_client: ClientHandler, limit: int
    ) -> None:
        self._network = network
        self._port = port
        self.on_client = on_client
        self.limit = limit
        self._closed = False

    @property
    def port(self) -> int:
        return self._port

    async def serve_forever(self) -> None:
        try:
            await asyncio.Event().wait()
        finally:
            await self.close()

    async def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._network._listeners.pop(self._port, None)


class MemoryNetwork:
    """Connections that stay inside the process: no sockets, no ports opened.

    Each instance is its own isolated network. Ports are plain numbers
    scoped to the instance; hosts are ignored.
    """

    _FIRST_EPHEMERAL_PORT = 49152

    def __init__(self) -> None:
        self._listeners: dict[int, _MemoryListener] = {}
        self._next_ephemeral_port = self._FIRST_EPHEMERAL_PORT
        self._handler_tasks: set[asyncio.Task[None]] = set()

    def _allocate_port(self) -> int:
        while self._next_ephemeral_port in self._listeners:
            self._next_ephemeral_port += 1
        port = self._next_ephemeral_port
        self._next_ephemeral_port += 1
        return port

    async def start_server(
        self, on_client: ClientHandler, host: str, port: int, *, limit: int = DEFAULT_LIMIT
    ) -> Listener:
        if port == 0:
            port = self._allocate_port()
        elif port in self._listeners:
            raise OSError(errno.EADDRINUSE, f"Address already in use: {host}:{port}")
        listener = _MemoryListener(self, port, on_client, limit)
        self._listeners[port] = listener
        return listener

    async def open_connection(
        self, host: str, port: int, *, limit: int = DEFAULT_LIMIT
    ) -> tuple[StreamReaderLike, StreamWriterLike]:
        listener = self._listeners.get(port)
        if listener is None:
            raise ConnectionRefusedError(errno.ECONNREFUSED, f"Nothing listening on {host}:{port}")
        client_port = self._allocate_port()
        client_reader = asyncio.StreamReader(limit=limit)
        server_reader = asyncio.StreamReader(limit=listener.limit)
        client_writer = _MemoryWriter(server_reader, (host, port))
        server_writer = _MemoryWriter(client_reader, ("memory", client_port))
        client_writer._peer = server_writer
        server_writer._peer = client_writer

        async def run_handler() -> None:
            await listener.on_client(server_reader, server_writer)

        task = asyncio.create_task(run_handler())
        self._handler_tasks.add(task)
        task.add_done_callback(self._handler_tasks.discard)
        return client_reader, client_writer
