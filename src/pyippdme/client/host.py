# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Run an I++ DME client on a background thread, for front ends that own the main thread.

The counterpart of :class:`pyippdme.server.host.ServerHost`: a GUI keeps its thread, the asyncio
loop runs here. Every command runs as its own task, so a long ``GoTo`` does not block ``AbortE``
or a status query sent meanwhile. Callbacks run on the loop thread; a Qt front end hands them over
with a signal.
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from concurrent.futures import Future
from typing import Any, TypeVar

from pyippdme.client import IppDmeClient
from pyippdme.client.interaction import CommandEvent, ConnectionLost, run_command_line
from pyippdme.client.model import IppDmeMachine
from pyippdme.client.optical import SensorInfo, acquire, read_sensor_info
from pyippdme.loop import LoopThread
from pyippdme.protocol.network import TCP_NETWORK, MemoryNetwork, Network
from pyippdme.rawdata.formats import ScanData
from pyippdme.server import IppDmeServer

T = TypeVar("T")

#: Called with the command text and each step of its transaction.
EventHandler = Callable[[str, CommandEvent], None]


class ClientHost:
    """One connection to a server, driven from any thread."""

    def __init__(self) -> None:
        self._thread = LoopThread("ippdme-client")
        self._client: IppDmeClient | None = None
        self._embedded: IppDmeServer[Any] | None = None
        self._address: tuple[str, Network] = ("", TCP_NETWORK)

    def submit(self, coro: Coroutine[Any, Any, T]) -> Future[T]:
        return self._thread.submit(coro)

    @property
    def connected(self) -> bool:
        return self._client is not None

    @property
    def embedded(self) -> bool:
        return self._embedded is not None

    async def _connect(self, host: str, port: int, network: Network) -> None:
        await self._disconnect()
        self._client = await IppDmeClient.connect(host, port, network=network)
        self._address = (host, network)

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
            self._address = ("virtual", self._embedded.network)

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

    def sensor_info(self) -> Future[SensorInfo]:
        """Return the active tool and how its raw data is delivered."""

        async def go() -> SensorInfo:
            return await read_sensor_info(self._machine())

        return self.submit(go())

    def acquire(self, **options: Any) -> Future[ScanData]:
        """Run ``DataAcquire`` with the active tool and fetch the points."""

        async def go() -> ScanData:
            host, network = self._address
            return await acquire(self._machine(), host, network=network, **options)

        return self.submit(go())

    def _machine(self) -> IppDmeMachine:
        if self._client is None:
            raise RuntimeError("not connected")
        return IppDmeMachine(self._client)

    def run_sequence(self, lines: list[str], on_event: EventHandler) -> Future[bool]:
        """Send the lines one after the other; stops at the first error. True if all succeeded."""
        from pyippdme.client.interaction import Failed, ParseFailed

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
            self._thread.close()
