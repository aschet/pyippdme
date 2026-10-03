# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Run an I++ DME client on a background thread, for front ends that own the main thread.

The counterpart of :class:`pyippdme.twin.host.ServerHost`: a GUI keeps its thread, the asyncio
loop runs here. Every command runs as its own task, so a long ``GoTo`` does not block ``AbortE``
or a status query sent meanwhile. Callbacks run on the loop thread; a Qt front end hands them over
with a signal.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable, Coroutine
from concurrent.futures import Future
from typing import Any, TypeVar

from pyippdme.cli._interaction import CommandEvent, ConnectionLost, run_command_line
from pyippdme.client import IppDmeClient
from pyippdme.protocol.network import TCP_NETWORK, MemoryNetwork, Network
from pyippdme.server import IppDmeServer

T = TypeVar("T")

#: Called with the command text and each step of its transaction.
EventHandler = Callable[[str, CommandEvent], None]


class ClientHost:
    """One connection to a server, driven from any thread."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, name="ippdme-client", daemon=True)
        self._thread.start()
        self._client: IppDmeClient | None = None
        self._embedded: IppDmeServer[Any] | None = None

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def submit(self, coro: Coroutine[Any, Any, T]) -> Future[T]:
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    @property
    def connected(self) -> bool:
        return self._client is not None

    @property
    def embedded(self) -> bool:
        return self._embedded is not None

    async def _connect(self, host: str, port: int, network: Network) -> None:
        await self._disconnect()
        self._client = await IppDmeClient.connect(host, port, network=network)

    def connect(self, host: str, port: int) -> Future[None]:
        """Connect over TCP; the future fails with the connection error."""
        return self.submit(self._connect(host, port, TCP_NETWORK))

    def connect_embedded(self, server: IppDmeServer[Any] | None = None) -> Future[None]:
        """Start a server in this process (default: the CLI's ``--virtual`` machine) and connect."""

        async def start() -> None:
            await self._disconnect()
            if server is None:
                from pyippdme.cli.virtual import create_embedded

                self._embedded = create_embedded(MemoryNetwork())
            else:
                self._embedded = server
            port = await self._embedded.start()
            self._client = await IppDmeClient.connect(
                "virtual", port, network=self._embedded.network
            )

        return self.submit(start())

    async def _disconnect(self) -> None:
        client, self._client = self._client, None
        embedded, self._embedded = self._embedded, None
        if client is not None:
            await client.close()
        if embedded is not None:
            await embedded.close()

    def disconnect(self) -> Future[None]:
        return self.submit(self._disconnect())

    def run(self, text: str, on_event: EventHandler) -> Future[None]:
        """Send one command line; ``on_event`` gets each step. Commands may overlap."""

        async def go() -> None:
            client = self._client
            if client is None:
                raise RuntimeError("not connected")
            async for event in run_command_line(client, text):
                on_event(text, event)
                if isinstance(event, ConnectionLost):
                    self._client = None

        return self.submit(go())

    def run_sequence(self, lines: list[str], on_event: EventHandler) -> Future[bool]:
        """Send the lines one after the other; stops at the first error. True if all succeeded."""
        from pyippdme.cli._interaction import Failed, ParseFailed

        async def go() -> bool:
            client = self._client
            if client is None:
                raise RuntimeError("not connected")
            for text in lines:
                async for event in run_command_line(client, text):
                    on_event(text, event)
                    if isinstance(event, ConnectionLost):
                        self._client = None
                    if isinstance(event, ConnectionLost | Failed | ParseFailed):
                        return False
            return True

        return self.submit(go())

    def shutdown(self) -> None:
        try:
            self.disconnect().result(timeout=10)
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)
            self._loop.close()
