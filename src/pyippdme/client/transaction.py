# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Client-side transaction bookkeeping (5.4.3, 5.5)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from pyippdme.exceptions import IppDmeConnectionError, IppDmeServerError
from pyippdme.protocol.ast import DataPayload, TagLike
from pyippdme.protocol.errors import ServerError


@dataclass(slots=True)
class Transaction:
    """Tracks one client-initiated transaction from Ack through Done.

    A transaction started with an :class:`~pyippdme.protocol.ast.EventTag`
    (e.g. an ``OnMoveReportE(...)`` daemon) keeps delivering data through
    :meth:`events` after :meth:`wait_complete` returns, for as long as the
    daemon runs; a plain-:class:`~pyippdme.protocol.ast.Tag` transaction
    collects all of its data up front and is done once :meth:`wait_complete`
    returns.
    """

    tag: TagLike
    _acked: asyncio.Event = field(default_factory=asyncio.Event, repr=False, compare=False)
    _completed: asyncio.Event = field(default_factory=asyncio.Event, repr=False, compare=False)
    _error: ServerError | None = field(default=None, repr=False, compare=False)
    _connection_error: IppDmeConnectionError | None = field(default=None, repr=False, compare=False)
    _data: list[DataPayload] = field(default_factory=list, repr=False, compare=False)
    #: Live feed of this transaction's own data responses, for :meth:`stream`;
    #: ``None`` marks the end (Done, Error, or connection loss).
    _incoming: asyncio.Queue[DataPayload | None] = field(
        default_factory=asyncio.Queue, repr=False, compare=False
    )
    _events: asyncio.Queue[DataPayload | None] = field(
        default_factory=asyncio.Queue, repr=False, compare=False
    )

    def _on_ack(self) -> None:
        self._acked.set()

    def _on_data(self, payload: DataPayload) -> None:
        if self._completed.is_set():
            self._events.put_nowait(payload)
        else:
            self._data.append(payload)
            self._incoming.put_nowait(payload)

    def _on_done(self) -> None:
        self._completed.set()
        self._incoming.put_nowait(None)

    def _on_error(self, error: ServerError) -> None:
        self._error = error
        self._acked.set()
        self._completed.set()
        self._incoming.put_nowait(None)

    def _on_close(self) -> None:
        self._events.put_nowait(None)

    def _on_connection_lost(self, error: IppDmeConnectionError) -> None:
        self._connection_error = error
        self._acked.set()
        self._completed.set()
        self._incoming.put_nowait(None)
        self._events.put_nowait(None)

    async def wait_ack(self) -> None:
        """Wait for the acknowledgement response, raising if it was an error."""
        await self._acked.wait()
        if self._connection_error is not None:
            raise self._connection_error
        if self._error is not None:
            raise IppDmeServerError(self._error)

    async def wait_complete(self) -> tuple[DataPayload, ...]:
        """Wait for the transaction-complete response and return the data collected so far."""
        await self._completed.wait()
        if self._connection_error is not None:
            raise self._connection_error
        if self._error is not None:
            raise IppDmeServerError(self._error)
        return tuple(self._data)

    async def stream(self) -> AsyncIterator[DataPayload]:
        """Iterate this transaction's own data responses as they arrive, until Done/Error.

        Unlike :meth:`wait_complete`, which buffers everything and returns
        once the transaction finishes, this yields each response as soon as
        it is received - for a handler that streams results incrementally
        (e.g. ``ScanOnLine``, see :mod:`pyippdme.simulation.classes.scanning_class`),
        so a caller can react to (or interrupt after) each point instead of
        waiting for the whole scan to finish. Raises the same errors
        :meth:`wait_complete` would, once the stream ends.
        """
        while True:
            item = await self._incoming.get()
            if item is None:
                break
            yield item
        if self._connection_error is not None:
            raise self._connection_error
        if self._error is not None:
            raise IppDmeServerError(self._error)

    async def events(self) -> AsyncIterator[DataPayload]:
        """Iterate data responses arriving after :meth:`wait_complete` (daemon events)."""
        while True:
            item = await self._events.get()
            if item is None:
                return
            yield item
