# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Line-oriented transport over a byte stream (5.1, 5.8)."""

from __future__ import annotations

import asyncio
import contextlib

from pyippdme.exceptions import IppDmeConnectionError
from pyippdme.protocol.network import TCP_NETWORK, Network, StreamReaderLike, StreamWriterLike

#: The standard recommends port 1294 for I++ DME clients and servers (5.0).
DEFAULT_PORT = 1294

#: The asyncio stream buffer must be at least as large as
#: pyippdme.protocol.codec.MAX_LINE_LENGTH, or a legitimately long line (e.g.
#: a Tool.Id()/raw point cloud XML payload) would be rejected here before
#: ever reaching that check. Used for both the client (see connect()) and
#: the server (see IppDmeServer.serve_forever()).
READ_LIMIT = 9 * 1024 * 1024


class LineTransport:
    """Reads/writes ``<CR><LF>``-terminated ASCII lines over an asyncio-style stream."""

    def __init__(self, reader: StreamReaderLike, writer: StreamWriterLike) -> None:
        self._reader = reader
        self._writer = writer

    @classmethod
    async def connect(
        cls, host: str, port: int = DEFAULT_PORT, *, network: Network = TCP_NETWORK
    ) -> LineTransport:
        try:
            reader, writer = await network.open_connection(host, port, limit=READ_LIMIT)
        except OSError as exc:
            raise IppDmeConnectionError(f"Could not connect to {host}:{port}: {exc}") from exc
        return cls(reader, writer)

    async def read_line(self) -> str:
        try:
            data = await self._reader.readuntil(b"\r\n")
        except asyncio.IncompleteReadError as exc:
            raise IppDmeConnectionError(
                "Connection closed before a complete line was received"
            ) from exc
        except asyncio.LimitOverrunError as exc:
            raise IppDmeConnectionError("Line exceeded the maximum buffer size") from exc
        try:
            return data.decode("ascii")
        except UnicodeDecodeError as exc:
            raise IppDmeConnectionError(f"Received non-ASCII data: {data!r}") from exc

    def write_line(self, data: bytes) -> None:
        self._writer.write(data)

    async def drain(self) -> None:
        await self._writer.drain()

    async def close(self) -> None:
        self._writer.close()
        with contextlib.suppress(OSError):
            await self._writer.wait_closed()

    @property
    def peer(self) -> str:
        peername = self._writer.get_extra_info("peername")
        if not peername:
            return "<unknown>"
        host, port = peername[0], peername[1]
        return f"{host}:{port}"
