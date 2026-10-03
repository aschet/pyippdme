# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""An asyncio loop on its own thread, so that a GUI (or any blocking program) keeps its thread."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Coroutine
from concurrent.futures import Future
from typing import Any, TypeVar

__all__ = ["LoopThread"]

T = TypeVar("T")


class LoopThread:
    """Run coroutines from any thread.

    Used by :class:`~pyippdme.client.host.ClientHost` and
    :class:`~pyippdme.server.host.ServerHost`, and usable for your own hosts.
    """

    def __init__(self, name: str = "ippdme") -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def submit(self, coro: Coroutine[Any, Any, T]) -> Future[T]:
        """Start ``coro`` on the loop; the future gives its result or exception."""
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def close(self) -> None:
        """Stop the loop and join its thread; nothing may be submitted afterwards."""
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
        self._loop.close()
