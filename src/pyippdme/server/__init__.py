# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Async I++ DME server (section 5).

Execution model (5.4.3 "transactions may overlap", 5.7 "Prioritized
Execution"): a command whose name does **not** end in ``E`` goes onto a
single-consumer queue and is executed strictly in the order received - "the
standard command queue". A command whose name **does** end in ``E`` bypasses
that queue and starts executing immediately, concurrently with whatever the
queue is currently working on - this is the literal rule the standard gives for
telling prioritized commands apart ("commands for the prioritized execution
end with an upper-case E... to be independent from the transport layer").
``AbortE()`` is additionally handled at the connection level rather than as
a registry handler, since aborting requires reaching into this connection's
own queue/task bookkeeping (cancel whatever is currently executing, reject
whatever is still queued) - not something a generic per-command handler can
do.

A command's :class:`~pyippdme.server.registry.CommandContext` carries a
per-transaction :data:`~pyippdme.server.backend.CancellationToken` that
``AbortE()`` sets before cancelling its task, so a streaming handler (see
``ScanOnLine``/``ScanOnCircle`` in :mod:`pyippdme.simulation.classes.scanning_class`)
gets both signals: the token, for cooperative/explicit checks, and
``asyncio.CancelledError`` at its next await point.

A bare :class:`IppDmeServer` registers no commands (``command_classes``
defaults to ``()``) and builds a plain
:class:`~pyippdme.server.registry.MachineState` per connection. See
:class:`~pyippdme.simulation.virtual_cmm.VirtualCMM` for a server with
simulated command classes.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass
from typing import Generic

from pyippdme.exceptions import IppDmeConnectionError, IppDmeProtocolError
from pyippdme.protocol.ast import (
    AckResponse,
    Argument,
    DataPayload,
    DataResponse,
    DoneResponse,
    ErrorResponse,
    Response,
    TagLike,
)
from pyippdme.protocol.codec import decode_command_line, encode_line
from pyippdme.protocol.commands import CommandName
from pyippdme.protocol.errors import DEFAULT_ERRORS, ErrorCode, ErrorSeverity, ServerError
from pyippdme.protocol.hooks import LineHook, call_line_hook
from pyippdme.protocol.network import (
    TCP_NETWORK,
    Listener,
    Network,
    StreamReaderLike,
    StreamWriterLike,
)
from pyippdme.protocol.transport import DEFAULT_PORT, READ_LIMIT, LineTransport
from pyippdme.server.backend import CancellationToken, MachineBackend
from pyippdme.server.registry import (
    DEFAULT_MACHINE_CLASS,
    CommandContext,
    CommandRegistry,
    HandlerResult,
    MachineState,
    StateT,
)
from pyippdme.server.surface import SampleSurface
from pyippdme.types.csy import CsyStore, InMemoryCsyStore

logger = logging.getLogger("pyippdme.server")

#: Commands a client may send while a severity>=2 error is active (5.6.1/5.6.2).
#: StartSession and EndSession are included even though 5.6.1 does not name
#: them explicitly: StartSession's own definition (6.3) states it
#: "implicitly executes ClearAllErrors()", and the "Session" section states
#: that "sending an EndSession() followed by a StartSession will start a new
#: session in any case" - i.e. this pair is meant to be the guaranteed
#: recovery path. Without this exemption, any severity>=2 error (most of the
#: predefined errors, including e.g. ErrorCode.UNSUPPORTED_COMMAND) would
#: permanently lock a connection out of ever (re)starting a session.
_ERROR_EXEMPT_COMMANDS = frozenset(
    {
        CommandName.GET_ERR_STATUS_E,
        CommandName.GET_XTD_ERR_STATUS,
        CommandName.CLEAR_ALL_ERRORS,
        CommandName.START_SESSION,
        CommandName.END_SESSION,
    }
)
#: Commands allowed before a session has been started (5.8, 6.3 "Session").
_SESSION_EXEMPT_COMMANDS = frozenset({CommandName.START_SESSION, CommandName.END_SESSION})
#: "Transaction aborted", the error AbortE() reports for whatever it cancels
#: (Annex B); its text comes straight from DEFAULT_ERRORS rather
#: than being repeated here, so the two can't drift apart.
_TRANSACTION_ABORTED_TEXT = DEFAULT_ERRORS[ErrorCode.TRANSACTION_ABORTED][1]


@dataclass(slots=True)
class _ActiveTransaction:
    task: asyncio.Task[None]
    cancel: CancellationToken


class _ServerConnection(Generic[StateT]):
    def __init__(
        self,
        transport: LineTransport,
        registry: CommandRegistry,
        backend: MachineBackend | None,
        csy_store: CsyStore,
        machine_class: str,
        state_factory: Callable[[], StateT],
        network: Network,
        sample_surface: SampleSurface | None = None,
        on_line_received: LineHook | None = None,
        on_line_sent: LineHook | None = None,
    ) -> None:
        self._transport = transport
        self._network = network
        self._registry = registry
        self._backend = backend
        self._csy_store = csy_store
        self._sample_surface = sample_surface
        self._on_line_received = on_line_received
        self._on_line_sent = on_line_sent
        self.state = state_factory()
        self.state.machine_class = machine_class
        self._queue: asyncio.Queue[tuple[TagLike, str, tuple[Argument, ...]]] = asyncio.Queue()
        self._active: dict[str, _ActiveTransaction] = {}
        self._worker_task: asyncio.Task[None] | None = None

    async def run(self) -> None:
        self._worker_task = asyncio.create_task(self._worker_loop())
        try:
            while True:
                line = await self._transport.read_line()
                if self._on_line_received is not None:
                    call_line_hook(self._on_line_received, line.removesuffix("\r\n"))
                await self._handle_line(line)
        except IppDmeConnectionError:
            pass
        finally:
            self._worker_task.cancel()
            for txn in list(self._active.values()):
                txn.task.cancel()
            await self._transport.close()

    async def _handle_line(self, line: str) -> None:
        try:
            command = decode_command_line(line)
        except IppDmeProtocolError as exc:
            logger.warning("Discarding unparseable line from %s: %s", self._transport.peer, exc)
            return
        tag, name, args = command.tag, command.method.name, command.method.args

        # Ack is sent immediately and in receipt order regardless of queueing:
        # this is a single ordered TCP stream, so that ordering is free.
        self._write(AckResponse(tag))
        await self._transport.drain()

        if name == CommandName.ABORT_E:
            await self._handle_abort_e(tag)
        elif name.endswith("E"):
            self._start(tag, name, args)  # prioritized: bypasses the standard queue (5.7)
        else:
            await self._queue.put((tag, name, args))

    async def _worker_loop(self) -> None:
        """Run the standard command queue: exactly one non-prioritized command at a time."""
        while True:
            tag, name, args = await self._queue.get()
            task = self._start(tag, name, args)
            with contextlib.suppress(asyncio.CancelledError):
                await task

    def _start(self, tag: TagLike, name: str, args: tuple[Argument, ...]) -> asyncio.Task[None]:
        cancel = asyncio.Event()
        task = asyncio.create_task(self._run_transaction(tag, name, args, cancel))
        key = tag.to_wire()
        self._active[key] = _ActiveTransaction(task, cancel)

        def _forget(_task: asyncio.Task[None], key: str = key) -> None:
            self._active.pop(key, None)

        task.add_done_callback(_forget)
        return task

    async def _run_transaction(
        self, tag: TagLike, name: str, args: tuple[Argument, ...], cancel: CancellationToken
    ) -> None:
        try:
            await self._execute(tag, name, args, cancel)
        except asyncio.CancelledError:
            await self._send_error(
                tag, ErrorSeverity.ERROR, ErrorCode.TRANSACTION_ABORTED, _TRANSACTION_ABORTED_TEXT
            )

    async def _handle_abort_e(self, tag: TagLike) -> None:
        # "not start any pending commands (those for which an Ack has been
        # sent but for which execution has not yet started)": reject
        # everything still sitting in the standard queue.
        pending: list[tuple[TagLike, str, tuple[Argument, ...]]] = []
        while not self._queue.empty():
            pending.append(self._queue.get_nowait())
        for pending_tag, _name, _args in pending:
            await self._send_error(
                pending_tag,
                ErrorSeverity.ERROR,
                ErrorCode.TRANSACTION_ABORTED,
                _TRANSACTION_ABORTED_TEXT,
            )

        # "stop executing any currently executing commands": cancel everything
        # in flight (both the queue's current item and any concurrently
        # running prioritized commands), and wait for them to actually unwind
        # before completing AbortE itself.
        active = list(self._active.values())
        for txn in active:
            txn.cancel.set()
            txn.task.cancel()
        if active:
            await asyncio.gather(*(txn.task for txn in active), return_exceptions=True)

        self._write(DoneResponse(tag))
        await self._transport.drain()

    async def _execute(
        self, tag: TagLike, name: str, args: tuple[Argument, ...], cancel: CancellationToken
    ) -> None:
        if not self.state.session_active and name not in _SESSION_EXEMPT_COMMANDS:
            await self._send_error(
                tag, ErrorSeverity.CRITICAL, ErrorCode.PROTOCOL_ERROR, "No session active"
            )
            return
        active_error = self.state.active_error
        if (
            active_error is not None
            and active_error.severity.requires_clear_all_errors
            and name not in _ERROR_EXEMPT_COMMANDS
        ):
            await self._send_error(
                tag,
                ErrorSeverity.ERROR,
                ErrorCode.USE_CLEAR_ALL_ERRORS,
                "Use ClearAllErrors to continue",
            )
            return

        handler = self._registry.get(name)
        if handler is None:
            await self._send_error(
                tag, ErrorSeverity.CRITICAL, ErrorCode.UNSUPPORTED_COMMAND, "Unsupported command"
            )
            return

        ctx = CommandContext(
            tag=tag,
            state=self.state,
            registry=self._registry,
            backend=self._backend,
            csy_store=self._csy_store,
            cancel=cancel,
            sample_surface=self._sample_surface,
            network=self._network,
            emit_event=self._emit_event,
        )
        try:
            result = await handler(ctx, args)
            # A handler returning a value that violates the HandlerResult
            # contract (e.g. a bare NamedValue instead of Items(...)) must
            # fail the same way a handler *raising* does, not escape
            # unhandled: left uncaught, an exception here would propagate out
            # of _worker_loop's `await task` (only CancelledError is
            # suppressed there) and kill it silently, wedging every command
            # this connection sends afterwards with no response, ever.
            await self._send_data(tag, result)
        except ServerError as err:
            if err.severity.requires_clear_all_errors:
                self.state.active_error = err
            await self._send_error(tag, err.severity, err.number, err.text, cause=err.cause)
            return
        except Exception:
            # A handler bug must not take down the whole connection: log it and
            # report it to the client as a regular (if generic) error response.
            logger.exception("Unhandled exception in handler for %s", name)
            await self._send_error(
                tag,
                ErrorSeverity.CRITICAL,
                ErrorCode.ERROR_PROCESSING_METHOD,
                "Error processing method",
                cause=name,
            )
            return

        self._write(DoneResponse(tag))
        await self._transport.drain()

    async def _send_data(self, tag: TagLike, result: HandlerResult) -> None:
        if result is None:
            return
        if isinstance(result, DataPayload):
            self._write(DataResponse(tag, result))
            await self._transport.drain()
        elif isinstance(result, AsyncIterator):
            # Explicit aclose() (rather than contextlib.aclosing, whose type
            # is narrower than the AsyncIterator HandlerResult promises)
            # ensures a cancelled/interrupted stream still runs the async
            # generator's own cleanup, e.g. stopping real hardware motion.
            try:
                async for payload in result:
                    self._write(DataResponse(tag, payload))
                    await self._transport.drain()
            finally:
                aclose = getattr(result, "aclose", None)
                if aclose is not None:
                    with contextlib.suppress(Exception):
                        await aclose()
        else:
            for payload in result:
                self._write(DataResponse(tag, payload))
                await self._transport.drain()

    async def _send_error(
        self, tag: TagLike, severity: ErrorSeverity, number: str, text: str, *, cause: str = ""
    ) -> None:
        if severity.requires_clear_all_errors and self.state.active_error is None:
            self.state.active_error = ServerError(severity, number, cause, text)
        self._write(ErrorResponse(tag, int(severity), number, cause, text))
        await self._transport.drain()
        self._write(DoneResponse(tag))
        await self._transport.drain()

    def _write(self, node: Response) -> None:
        if self._on_line_sent is not None:
            call_line_hook(self._on_line_sent, node.to_wire())
        self._transport.write_line(encode_line(node))

    async def _emit_event(self, tag: TagLike, payload: DataPayload) -> None:
        """Send one data message tagged with ``tag`` outside the normal request/response flow.

        For an ``OnMoveReport``/``OnMoveReportE`` daemon's reports (see
        :mod:`pyippdme.simulation.classes.mover_class`), which arrive tagged
        with the daemon's own ``EventTag``, not whatever command triggered
        this particular report.
        """
        self._write(DataResponse(tag, payload))
        await self._transport.drain()


class IppDmeServer(Generic[StateT]):
    """An I++ DME server that registers no command classes by default.

    ``IppDmeServer()`` alone registers nothing - not even ``Server``/``DME``
    - and needs no per-connection state beyond the bare
    :class:`~pyippdme.server.registry.MachineState`. Pass ``command_classes``
    to register any mix of :mod:`pyippdme.server.classes` (the two mandatory
    ones), :mod:`pyippdme.simulation.classes` (this library's own bundled
    reference simulation - see :class:`~pyippdme.simulation.virtual_cmm.VirtualCMM`
    for a server pre-loaded with all of them), or your own - each element is
    a ``register(registry)`` function applied in order::

        from pyippdme.server.classes import register_dme_class, register_server_class

        server = IppDmeServer(command_classes=[register_server_class, register_dme_class])

        @server.registry.command_proprietary("XX", "MyCommand")
        async def _my_command(ctx, args):
            ...

    Section 6.1 requires a two-character company namespace for any command
    not defined by the standard, so prefer
    :meth:`~pyippdme.server.registry.CommandRegistry.command_proprietary` over
    :meth:`~pyippdme.server.registry.CommandRegistry.command` for your own.
    The standard mandates ``Server`` and ``DME`` for a conformant
    implementation; this is not enforced here, so intentionally-partial or
    test servers remain possible.

    To keep only *some* commands from a class you otherwise want (e.g. a
    ``Scanning`` backend that only ever does ``ScanOnLine``, not
    ``ScanOnCircle``/``ScanOnHelix``), use
    :func:`~pyippdme.server.registry.register_subset` instead of hand-writing
    a matching :meth:`~pyippdme.server.registry.CommandRegistry.unregister`
    call per command you don't want - :func:`~pyippdme.server.registry.commands_of`
    answers "what would this class register" if you need that list for
    anything else::

        server = IppDmeServer(command_classes=[register_server_class, register_dme_class])
        register_subset(server.registry, register_scanning_class, keep={CommandName.SCAN_ON_LINE})

    For a *new* command class (built-in or your own) where you want a
    method's own parameters typed instead of a raw argument tuple, and its
    registered commands always kept truthfully in sync with what
    ``GetSupportedCommands()`` reports without a separate call, write it as
    a plain object with ``@command``/``@command_proprietary``-decorated
    methods and pass it to :func:`~pyippdme.server.dispatch.register_object`
    instead; see :mod:`pyippdme.server.dispatch`.

    Pass ``backend`` for the ``Scanning`` class's real hardware control (see
    :mod:`pyippdme.server.backend`) - ``None`` (the default) is only a
    problem if a registered handler actually needs one and none was given.
    Pass ``csy_store`` to persist named coordinate systems across restarts,
    e.g. ``FileCsyStore()``; see :mod:`pyippdme.types.csy` (the in-memory
    default does not). Pass ``machine_class`` to override the string every
    connection's ``GetMachineClass()`` reports (6.4.1). Pass ``sample_surface``
    to give ``PtMeas`` (6.12.1) a real synthetic part to measure against
    instead of always reporting exactly the commanded position; see
    :mod:`pyippdme.server.surface`. Pass ``state_factory`` if any registered
    command class needs more per-connection state than
    :class:`~pyippdme.server.registry.MachineState` itself provides (see
    :class:`~pyippdme.simulation.state.SimulationState` for this library's
    own bundled simulation's extension) - it must build the same concrete
    type ``StateT`` this server is parameterized over. Pass ``network`` to
    listen somewhere other than TCP, e.g. a
    :class:`~pyippdme.protocol.network.MemoryNetwork` for connections that stay
    inside the process. Pass ``on_line_received``/``on_line_sent``/``on_connect``/``on_disconnect``
    for observe-only wire-level visibility (e.g. ``ippdme serve
    --session-log``); see :mod:`pyippdme.protocol.hooks`.
    """

    def __init__(
        self,
        *,
        backend: MachineBackend | None = None,
        csy_store: CsyStore | None = None,
        machine_class: str = DEFAULT_MACHINE_CLASS,
        command_classes: Sequence[Callable[[CommandRegistry], None]] = (),
        state_factory: Callable[[], StateT] = MachineState,  # type: ignore[assignment]
        network: Network = TCP_NETWORK,
        sample_surface: SampleSurface | None = None,
        on_line_received: LineHook | None = None,
        on_line_sent: LineHook | None = None,
        on_connect: Callable[[str], None] | None = None,
        on_disconnect: Callable[[str], None] | None = None,
    ) -> None:
        self.registry = CommandRegistry()
        for register_class in command_classes:
            register_class(self.registry)
        self.backend = backend
        self._state_factory = state_factory
        self.csy_store: CsyStore = csy_store if csy_store is not None else InMemoryCsyStore()
        self.machine_class = machine_class
        #: Where this server's listener (and anything a handler has to open) comes from.
        self.network = network
        #: The synthetic part ``PtMeas`` measures against, if any; see
        #: :mod:`pyippdme.server.surface`. ``None`` (the default) keeps
        #: ``PtMeas`` reporting the commanded position exactly.
        self.sample_surface = sample_surface
        #: Observe-only wire-line hooks (see :mod:`pyippdme.protocol.hooks`),
        #: named from this server's own point of view: ``on_line_received``
        #: fires for a line a client sent it (a command), ``on_line_sent``
        #: for a line it sent back (a response) - the mirror image of
        #: :class:`~pyippdme.client.IppDmeClient`'s hooks of the same name.
        #: Per 5.8, only one client is ever connected at a time,
        #: so unlike :class:`~pyippdme.spy.Spy`'s hooks these carry no
        #: connection identifier to disambiguate.
        self.on_line_received = on_line_received
        self.on_line_sent = on_line_sent
        #: Called with the peer's address (``host:port``) on each new
        #: connection/disconnection.
        self.on_connect = on_connect
        self.on_disconnect = on_disconnect
        self._listener: Listener | None = None

    async def start(self, host: str = "127.0.0.1", port: int = 0) -> int:
        """Bind and start accepting connections without blocking; return the bound port.

        Unlike :meth:`serve_forever` (which blocks the caller until
        cancelled), this returns as soon as the listening socket is up -
        for embedding the server in-process alongside other work in the
        same event loop (a REPL/TUI driving it directly, a test fixture,
        ...) rather than running it as a standalone process. ``port=0``
        (the default) lets the network pick a free port, returned here so
        the caller can connect to it. Call :meth:`close` when done.
        """
        self._listener = await self.network.start_server(
            self._on_client, host, port, limit=READ_LIMIT
        )
        return self._listener.port

    @property
    def port(self) -> int | None:
        """The port currently bound by :meth:`start`/:meth:`serve_forever`, or ``None``."""
        if self._listener is None:
            return None
        return self._listener.port

    async def serve_forever(
        self,
        host: str = "0.0.0.0",  # noqa: S104 # nosec B104
        port: int = DEFAULT_PORT,
    ) -> None:
        await self.start(host, port)
        assert self._listener is not None  # noqa: S101 (set by start() just above)
        await self._listener.serve_forever()

    async def close(self) -> None:
        if self._listener is not None:
            await self._listener.close()
            self._listener = None

    async def _on_client(self, reader: StreamReaderLike, writer: StreamWriterLike) -> None:
        transport = LineTransport(reader, writer)
        logger.info("Client connected: %s", transport.peer)
        if self.on_connect is not None:
            call_line_hook(self.on_connect, transport.peer)
        connection = _ServerConnection(
            transport,
            self.registry,
            self.backend,
            self.csy_store,
            self.machine_class,
            self._state_factory,
            self.network,
            self.sample_surface,
            self.on_line_received,
            self.on_line_sent,
        )
        await connection.run()
        logger.info("Client disconnected: %s", transport.peer)
        if self.on_disconnect is not None:
            call_line_hook(self.on_disconnect, transport.peer)
