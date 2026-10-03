# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Handles for the commands the object model sends: await the result, or check the Ack first.

A method of :class:`~pyippdme.client.model.IppDmeMachine` starts its command as
soon as it is called and returns a :class:`Call` (or, for a command that
streams data, a :class:`StreamCall`)::

    await machine.cart_cmm.go_to(x=10)             # waits until the command is done

    call = machine.cart_cmm.pt_meas(x=1, y=2, z=3)  # command sent
    await call.acknowledged()                       # the server accepted it
    position = await call                           # the typed result

An Ack only means the server received the command. A command that then fails
(an unsupported command, a bad argument, no session, ...) is reported after the
Ack, so it raises :class:`~pyippdme.exceptions.IppDmeServerError` when the call
is awaited, not from :meth:`Call.acknowledged`.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
from collections.abc import AsyncIterator, Callable, Coroutine, Generator, Iterator
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Generic, ParamSpec, TypeVar

from pyippdme.client.transaction import Transaction
from pyippdme.exceptions import IppDmeError

T = TypeVar("T")
P = ParamSpec("P")


class _Capture:
    """Collects the transactions a handle's command sends."""

    def __init__(self) -> None:
        self.first: asyncio.Future[Transaction] = asyncio.get_running_loop().create_future()

    def add(self, transaction: Transaction) -> None:
        if not self.first.done():
            self.first.set_result(transaction)


_current_capture: ContextVar[_Capture | None] = ContextVar("pyippdme_capture", default=None)


def record_transaction(transaction: Transaction) -> None:
    """Hand a command the client just sent to the handle that started it, if any."""
    capture = _current_capture.get()
    if capture is not None:
        capture.add(transaction)


@contextlib.contextmanager
def unrecorded() -> Iterator[None]:
    """Keep commands sent inside the block (a method's setup step) out of its handle's Ack."""
    token = _current_capture.set(None)
    try:
        yield
    finally:
        _current_capture.reset(token)


def _start(coroutine: Coroutine[Any, Any, T], capture: _Capture) -> asyncio.Task[T]:
    token = _current_capture.set(capture)
    try:
        return asyncio.ensure_future(coroutine)
    finally:
        _current_capture.reset(token)


class _Handle:
    _capture: _Capture
    _task: asyncio.Task[Any]
    _ack_failed = False

    async def _first_transaction(self) -> Transaction | None:
        if not self._capture.first.done():
            await asyncio.wait(
                {self._capture.first, self._task}, return_when=asyncio.FIRST_COMPLETED
            )
        if not self._capture.first.done():
            self._task.result()  # nothing was sent; surface why the method ended
            return None
        return self._capture.first.result()

    async def transaction(self) -> Transaction:
        """Return the :class:`~pyippdme.client.transaction.Transaction` of the sent command.

        For the raw responses (:meth:`~pyippdme.client.transaction.Transaction.wait_complete`,
        :meth:`~pyippdme.client.transaction.Transaction.stream`) when the method's own
        parsed result is not what you need. This is the command the method exists
        for, not any setup command it sends first.
        """
        transaction = await self._first_transaction()
        if transaction is None:
            raise IppDmeError("The method ended without sending a command")
        return transaction

    async def acknowledged(self) -> None:
        """Wait until the server acknowledged the command (5.4).

        The Ack means the command was received; a failure is reported after it
        and raises when the call is awaited. This raises only if the server
        answered with an Error instead of an Ack
        (:class:`~pyippdme.exceptions.IppDmeServerError`) or the connection was
        lost first (:class:`~pyippdme.exceptions.IppDmeConnectionError`). The Ack
        is that of the command the method exists for, not of any setup command it
        sends first.
        """
        transaction = await self._first_transaction()
        if transaction is None:
            return
        try:
            await transaction.wait_ack()
        except IppDmeError:
            self._ack_failed = True
            if self._task.done():
                self._on_done(self._task)
            raise

    def _on_done(self, task: asyncio.Task[Any]) -> None:
        # The rejection was already reported through acknowledged(); don't
        # let asyncio complain that the task's own copy of it went unread.
        if self._ack_failed and not task.cancelled():
            task.exception()


class Call(_Handle, Generic[T]):
    """A command that has been sent; await it for its result."""

    def __init__(self, coroutine: Coroutine[Any, Any, T]) -> None:
        self._capture = _Capture()
        self._task: asyncio.Task[T] = _start(coroutine, self._capture)
        self._task.add_done_callback(self._on_done)

    def __await__(self) -> Generator[Any, None, T]:
        return self._task.__await__()


@dataclass(slots=True)
class _Item(Generic[T]):
    value: T


@dataclass(slots=True)
class _Failure:
    error: BaseException


class StreamCall(_Handle, Generic[T]):
    """A command that streams data; iterate it with ``async for``."""

    def __init__(self, source: AsyncIterator[T]) -> None:
        self._capture = _Capture()
        self._queue: asyncio.Queue[_Item[T] | _Failure | None] = asyncio.Queue()
        self._task: asyncio.Task[None] = _start(self._produce(source), self._capture)
        self._task.add_done_callback(self._on_done)

    async def _produce(self, source: AsyncIterator[T]) -> None:
        try:
            async for value in source:
                self._queue.put_nowait(_Item(value))
        except Exception as exc:
            self._queue.put_nowait(_Failure(exc))
        else:
            self._queue.put_nowait(None)

    def __aiter__(self) -> AsyncIterator[T]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[T]:
        try:
            while True:
                item = await self._queue.get()
                if item is None:
                    return
                if isinstance(item, _Failure):
                    raise item.error
                yield item.value
        finally:
            if not self._task.done():
                self._task.cancel()


def call_handle(func: Callable[P, Coroutine[Any, Any, T]]) -> Callable[P, Call[T]]:
    """Make a coroutine function start its command when called and return a :class:`Call`."""

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Call[T]:
        return Call(func(*args, **kwargs))

    return wrapper


def stream_handle(func: Callable[P, AsyncIterator[T]]) -> Callable[P, StreamCall[T]]:
    """Make an async generator function start its command when called, returning a StreamCall."""

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> StreamCall[T]:
        return StreamCall(func(*args, **kwargs))

    return wrapper
