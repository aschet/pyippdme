# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Run an I++ DME server in a background thread, for front ends that own the main thread."""

from __future__ import annotations

from collections.abc import Coroutine
from concurrent.futures import Future
from typing import Any, TypeVar

from pyippdme.loop import LoopThread
from pyippdme.server import IppDmeServer

__all__ = ["ServerHost"]

T = TypeVar("T")


class ServerHost:
    """An asyncio loop on its own thread that serves one :class:`~pyippdme.server.IppDmeServer`.

    A Qt (or any other) GUI keeps its main thread; the server, the motion model
    and the simulation run here. Use :meth:`submit` to run a coroutine, e.g. the
    server's ``key_press``, from the GUI thread.
    """

    def __init__(self) -> None:
        self._thread = LoopThread("ippdme-server")
        self._server: IppDmeServer[Any] | None = None

    def submit(self, coro: Coroutine[Any, Any, T]) -> Future[T]:
        return self._thread.submit(coro)

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
        self._thread.close()
