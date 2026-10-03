# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Run an I++ DME server in a background thread, for front ends that own the main thread."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Coroutine
from concurrent.futures import Future
from typing import Any, TypeVar

from pyippdme.server import IppDmeServer

T = TypeVar("T")


class ServerHost:
    """An asyncio loop on its own thread that serves one :class:`~pyippdme.server.IppDmeServer`.

    A Qt (or any other) GUI keeps its main thread; the server, the motion model
    and the simulation run here. Use :meth:`submit` to run a coroutine, e.g. the
    server's ``key_press``, from the GUI thread.
    """

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, name="ippdme-server", daemon=True)
        self._thread.start()
        self._server: IppDmeServer[Any] | None = None

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def submit(self, coro: Coroutine[Any, Any, T]) -> Future[T]:
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    @property
    def running(self) -> bool:
        return self._server is not None and self._server.port is not None

    @property
    def port(self) -> int | None:
        return self._server.port if self._server is not None else None

    def start(self, server: IppDmeServer[Any], host: str, port: int) -> int:
        """Listen on ``host:port`` (``port=0`` picks a free one); returns the bound port."""
        if self.running:
            raise RuntimeError("the server is already running")
        bound = self.submit(server.start(host, port)).result(timeout=10)
        self._server = server
        return bound

    def stop(self) -> None:
        if self._server is not None:
            server, self._server = self._server, None
            self.submit(server.close()).result(timeout=10)

    def shutdown(self) -> None:
        self.stop()
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
        self._loop.close()
