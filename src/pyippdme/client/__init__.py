# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Async I++ DME client (section 5)."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator

from pyippdme.client.call import record_transaction
from pyippdme.client.transaction import Transaction
from pyippdme.exceptions import IppDmeConnectionError, IppDmeError, IppDmeProtocolError
from pyippdme.protocol.ast import (
    AckResponse,
    Argument,
    Command,
    DataPayload,
    DataResponse,
    DoneResponse,
    ErrorResponse,
    EventTag,
    Method,
    NamedValue,
    Response,
    Tag,
    TagLike,
)
from pyippdme.protocol.codec import decode_response_line, encode_line
from pyippdme.protocol.commands import CommandName
from pyippdme.protocol.errors import ErrorSeverity, ServerError
from pyippdme.protocol.hooks import LineHook as LineHook
from pyippdme.protocol.hooks import call_line_hook
from pyippdme.protocol.network import TCP_NETWORK, Network
from pyippdme.protocol.transport import DEFAULT_PORT, LineTransport

logger = logging.getLogger("pyippdme.client")


class IppDmeClient:
    """A connection to an I++ DME server.

    Use :meth:`connect` to open a connection; the client then reads
    responses in a background task and routes them to the
    :class:`~pyippdme.client.transaction.Transaction` returned by :meth:`send`, or
    to :meth:`unsolicited_events` for ``E0000`` events (5.5.1).
    """

    def __init__(
        self,
        transport: LineTransport,
        *,
        on_line_sent: LineHook | None = None,
        on_line_received: LineHook | None = None,
    ) -> None:
        self._transport = transport
        self._on_line_sent = on_line_sent
        self._on_line_received = on_line_received
        self._pending: dict[str, Transaction] = {}
        self._next_command_tag = 1
        self._next_event_tag = 1
        self._unsolicited: asyncio.Queue[DataResponse | None] = asyncio.Queue()
        #: 5.4.3: "A client is not allowed to start a new transaction until it received
        #: the acknowledgement response of the most recent sent command line."
        self._ack_gate = asyncio.Lock()
        self._reader_task = asyncio.create_task(self._reader_loop())

    @classmethod
    async def connect(
        cls,
        host: str,
        port: int = DEFAULT_PORT,
        *,
        network: Network = TCP_NETWORK,
        on_line_sent: LineHook | None = None,
        on_line_received: LineHook | None = None,
    ) -> IppDmeClient:
        transport = await LineTransport.connect(host, port, network=network)
        return cls(transport, on_line_sent=on_line_sent, on_line_received=on_line_received)

    async def close(self) -> None:
        self._reader_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._reader_task
        await self._transport.close()
        self._fail_all_pending(IppDmeConnectionError("Connection closed"))

    # -- low-level command/transaction API -----------------------------------

    def _allocate_command_tag(self) -> Tag:
        for _ in range(Tag.MAX):
            n = self._next_command_tag
            self._next_command_tag = n + 1 if n < Tag.MAX else 1
            tag = Tag.of(n)
            if tag.to_wire() not in self._pending:
                return tag
        raise IppDmeError("No free command tags available")

    def allocate_event_tag(self) -> EventTag:
        """Reserve the next free :class:`~pyippdme.protocol.ast.EventTag`.

        For a caller that needs to :meth:`send` a daemon-starting command
        itself (see :meth:`start_daemon` for the common case that just wants
        one and doesn't care which); public because knowing *which* commands
        start a daemon is protocol/library knowledge this class doesn't
        have (see :mod:`pyippdme.client.interaction`, which does).
        """
        for _ in range(EventTag.MAX):
            n = self._next_event_tag
            self._next_event_tag = n + 1 if n < EventTag.MAX else 1
            tag = EventTag.of(n)
            if tag.to_wire() not in self._pending:
                return tag
        raise IppDmeError("No free event tags available")

    def send(self, name: str, *args: Argument, tag: TagLike | None = None) -> Transaction:
        """Send a command and return a :class:`Transaction` to await responses on.

        Pass an explicit ``tag`` (an :class:`~pyippdme.protocol.ast.EventTag`)
        to start a daemon whose events should be consumed via
        :meth:`Transaction.events` after the initial transaction completes.
        """
        chosen_tag = tag if tag is not None else self._allocate_command_tag()
        txn = Transaction(chosen_tag)
        self._pending[chosen_tag.to_wire()] = txn
        record_transaction(txn)
        command = Command(chosen_tag, Method(name, args))
        if self._on_line_sent is not None:
            call_line_hook(self._on_line_sent, command.to_wire())
        self._transport.write_line(encode_line(command))
        return txn

    async def _send_acknowledged(
        self, name: str, args: tuple[Argument, ...], tag: TagLike | None = None
    ) -> Transaction:
        """Send a command once the previous one was acknowledged, and wait for its Ack.

        Commands for prioritized execution (name ending in ``E``, 5.7) go
        out at once: they exist to get past a queue the server may still be
        working through, such as an ``AbortE()`` during a scan.
        """
        if name.endswith("E"):
            txn = self.send(name, *args, tag=tag)
            await self._transport.drain()
            await txn.wait_ack()
            return txn
        async with self._ack_gate:
            txn = self.send(name, *args, tag=tag)
            await self._transport.drain()
            await txn.wait_ack()
        return txn

    async def call(self, name: str, *args: Argument) -> tuple[DataPayload, ...]:
        """Send a command and wait for it to complete, returning its data.

        Commands wait for the Ack of the one before them (5.4.3), so
        calls made at the same time are sent one after another.
        """
        txn = await self._send_acknowledged(name, args)
        return await txn.wait_complete()

    async def call_streaming(self, name: str, *args: Argument) -> AsyncIterator[DataPayload]:
        """Send a command and yield its data responses as they arrive.

        Unlike :meth:`call`, which waits for the whole transaction to finish
        and returns everything at once, this is for handlers that stream
        results incrementally (e.g. ``ScanOnLine``); see
        :meth:`~pyippdme.client.transaction.Transaction.stream`.
        """
        txn = await self._send_acknowledged(name, args)
        async for payload in txn.stream():
            yield payload

    async def start_daemon(self, name: str, *args: Argument) -> Transaction:
        """Send an event-tagged command that starts a daemon (5.5.2)."""
        txn = await self._send_acknowledged(name, args, self.allocate_event_tag())
        await txn.wait_complete()
        return txn

    async def stop_daemon(self, event_tag: EventTag) -> None:
        await self.call(CommandName.STOP_DAEMON, event_tag)
        txn = self._pending.pop(event_tag.to_wire(), None)
        if txn is not None:
            txn._on_close()

    def unsolicited_events(self) -> AsyncIterator[DataResponse]:
        """Iterate unsolicited server events (tag ``E0000``, 5.5.1)."""
        return self._unsolicited_iterator()

    async def _unsolicited_iterator(self) -> AsyncIterator[DataResponse]:
        while True:
            item = await self._unsolicited.get()
            if item is None:
                return
            yield item

    # -- convenience wrappers for the mandatory Server/DME classes -----------

    async def start_session(self) -> None:
        await self.call(CommandName.START_SESSION)

    async def end_session(self) -> None:
        await self.call(CommandName.END_SESSION)

    async def clear_all_errors(self) -> None:
        await self.call(CommandName.CLEAR_ALL_ERRORS)

    async def get_prop(self, *names: str) -> tuple[DataPayload, ...]:
        return await self.call(CommandName.GET_PROP, *(NamedValue(n, ()) for n in names))

    async def set_prop(self, *properties: NamedValue) -> None:
        await self.call(CommandName.SET_PROP, *properties)

    # -- background response routing ------------------------------------------

    async def _reader_loop(self) -> None:
        try:
            while True:
                line = await self._transport.read_line()
                if self._on_line_received is not None:
                    call_line_hook(self._on_line_received, line.removesuffix("\r\n"))
                try:
                    response = decode_response_line(line)
                except IppDmeProtocolError:
                    continue
                self._dispatch(response)
        except IppDmeConnectionError as exc:
            self._fail_all_pending(exc)

    def _dispatch(self, response: Response) -> None:
        tag = response.tag
        if isinstance(tag, EventTag) and tag.is_unsolicited:
            if isinstance(response, DataResponse):
                self._unsolicited.put_nowait(response)
            return

        txn = self._pending.get(tag.to_wire())
        if txn is None:
            return

        if isinstance(response, AckResponse):
            txn._on_ack()
        elif isinstance(response, DoneResponse):
            one_shot = isinstance(tag, EventTag) and txn.received_data
            txn._on_done()
            if isinstance(tag, Tag):
                self._pending.pop(tag.to_wire(), None)
            elif one_shot:
                # 5.5.1/5.5.2: a one-shot event is sent before the transaction completes,
                # and its daemon dies after firing.
                self._pending.pop(tag.to_wire(), None)
                txn._on_close()
        elif isinstance(response, DataResponse):
            txn._on_data(response.data)
        elif isinstance(response, ErrorResponse):
            error = ServerError(
                severity=ErrorSeverity(response.severity),
                number=response.number,
                cause=response.cause,
                text=response.text,
            )
            txn._on_error(error)
            self._pending.pop(tag.to_wire(), None)

    def _fail_all_pending(self, error: IppDmeConnectionError) -> None:
        for txn in self._pending.values():
            txn._on_connection_lost(error)
        self._pending.clear()
        self._unsolicited.put_nowait(None)
