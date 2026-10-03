# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Server-side command dispatch: registry, per-connection state, and handler contract.

A :class:`CommandHandler` receives the parsed argument tuple of the incoming
method call and a :class:`CommandContext`, and returns the data to send back
(5.4.2/5.4.3): nothing (just Ack+Done), a single payload, a
sequence of payloads each sent as its own ``#`` data response line before
the final ``%`` (e.g. ``EnumNameSpaces()`` sending "each namespace in a
separate message"), or an async iterator of payloads sent as they are
produced rather than materialized up front (used by streaming commands like
``ScanOnLine``/``ScanOnCircle``, so a client-issued ``AbortE()`` can
interrupt delivery mid-stream; see :mod:`pyippdme.server.backend`).

:class:`MachineState` is deliberately minimal: only the session/error/
machine-identity bookkeeping the mandatory ``Server``/``DME`` classes
(:mod:`pyippdme.server.classes`) need. A command class that owns further
per-connection state (a simulated position, a fake tool catalog, ...)
extends it by subclassing, not by ``MachineState`` itself growing a field
per class that might not even be registered - see
:class:`~pyippdme.simulation.state.SimulationState` for this library's own
bundled simulation's extension, and :class:`~pyippdme.server.IppDmeServer`'s
``state_factory`` for how a server is told to build one. :class:`CommandContext`
is generic over the concrete state type for exactly this reason: a handler
written against a specific state subclass (``ctx: CommandContext[SimulationState]``)
gets full static type-checking on ``ctx.state``'s extra fields, while the
registry itself stores every handler erased to ``CommandContext[Any]`` - it
never needs to know or care which concrete state type a given server was
built with.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Generic, TypeAlias, TypeVar

from pyippdme.protocol.ast import Argument, DataPayload, EventTag, NamedValue, TagLike
from pyippdme.protocol.errors import ServerError
from pyippdme.protocol.namespace import proprietary_name
from pyippdme.protocol.network import TCP_NETWORK, Network
from pyippdme.protocol.signature import Parameter
from pyippdme.server.backend import CancellationToken, MachineBackend
from pyippdme.server.motion import MotionModel
from pyippdme.server.surface import RawSensor, SampleSurface
from pyippdme.types.csy import CsyStore

HandlerResult: TypeAlias = DataPayload | Sequence[DataPayload] | AsyncIterator[DataPayload] | None

#: One command class's attempt to handle a single ``SetProp(<name>(<value>))``
#: argument. Returns ``True`` if it recognized ``arg.name`` and handled it
#: (including by raising a :class:`~pyippdme.protocol.errors.ServerError`);
#: ``False`` lets the next resolver, or finally the generic per-connection
#: property store, try instead. See :meth:`CommandRegistry.register_property_resolver`.
PropertySetHandler: TypeAlias = Callable[["CommandContext[Any]", NamedValue], bool]

#: The ``GetProp`` counterpart of :data:`PropertySetHandler`: returns the
#: resolved ``NamedValue`` if this resolver recognized ``arg.name``, or
#: ``None`` to let the next resolver (or the generic store) try.
PropertyGetHandler: TypeAlias = Callable[["CommandContext[Any]", NamedValue], NamedValue | None]

#: One command class's attempt to list the direct children of a property
#: reference (``EnumProp``/``EnumAllProp``, 6.3.1.1). Returns each child as
#: ``(name, type)`` where ``type`` is one of Table 20's own vocabulary -
#: ``"Number"``, ``"String"``, or ``"Property"`` (the last meaning the
#: child itself has further children an ``EnumAllProp`` should recurse
#: into) - or ``None`` if this class doesn't recognize ``reference`` at
#: all, letting the next resolver try. See
#: :meth:`CommandRegistry.register_property_children`.
PropertyChildrenHandler: TypeAlias = Callable[
    ["CommandContext[Any]", str], tuple[tuple[str, str], ...] | None
]

#: One command class's attempt to stop a ``StopDaemon``/``StopAllDaemons``
#: (6.3.1) target: given the event tag ``StopDaemon`` named (``None`` for
#: ``StopAllDaemons``, "every daemon"), stop and forget whatever daemon(s)
#: it owns that match, returning ``True`` if it recognized/handled at least
#: one. See :meth:`CommandRegistry.register_daemon_stopper`.
DaemonStopHandler: TypeAlias = Callable[["CommandContext[Any]", EventTag | None], bool]

#: One command class's reaction to a session-lifecycle event
#: (``StartSession``/``EndSession``/``ClearAllErrors``, 6.3) - e.g. resetting
#: a modal flag that the standard says one of these resets, or releasing
#: resources a session held. See :meth:`CommandRegistry.register_session_start_hook`/
#: :meth:`~CommandRegistry.register_session_end_hook`/
#: :meth:`~CommandRegistry.register_clear_errors_hook`.
SessionLifecycleHandler: TypeAlias = Callable[["CommandContext[Any]"], None]


class PropertyKind(StrEnum):
    """Table 20's ``EnumProp``/``EnumAllProp`` child-type vocabulary."""

    NUMBER = "Number"
    STRING = "String"
    PROPERTY = "Property"


#: The default ``GetMachineClass()`` string (6.4.1's naming scheme), also
#: used as :class:`~pyippdme.server.IppDmeServer`'s default; see
#: :mod:`pyippdme.simulation.virtual_cmm` for the ``_VirtualCMM``-suffixed variant.
DEFAULT_MACHINE_CLASS = "CartCMM_TouchTrigger_Fixed_None_None_None_None"


@dataclass(slots=True)
class MachineState:
    """Per-connection state used by the mandatory ``Server``/``DME`` classes (6.3/6.4).

    Deliberately minimal - just session lifecycle, error handling, machine
    identity, and the generic ``SetProp``/``GetProp`` fallback property
    store, since those are the only things :mod:`pyippdme.server.classes`
    (the two classes every conformant server needs, real or simulated) ever
    touch. A command class that needs more per-connection state than this -
    which is every *other* built-in class, all of them simulated - extends
    it by subclassing, e.g. :class:`~pyippdme.simulation.state.SimulationState`;
    :class:`~pyippdme.server.IppDmeServer`'s ``state_factory`` constructor
    argument is how a server is told to build that subclass instead of a
    plain ``MachineState``. This keeps ``MachineState`` itself from growing
    a field per command class regardless of whether that class is even
    registered on a given server.
    """

    session_active: bool = False
    active_error: ServerError | None = None
    homed: bool = False
    machine_class: str | tuple[str, ...] = DEFAULT_MACHINE_CLASS
    dme_version: str = "2.5"
    #: Generic fallback store for SetProp/GetProp on properties not modeled
    #: individually by a built-in class, keyed by dotted property name.
    properties: dict[str, tuple[Argument, ...]] = field(default_factory=dict)

    def carry_over_from(self, previous: MachineState) -> None:
        """Take over what the server keeps between clients (6.3.1: ``EndSession``/``StartSession``).

        The machine keeps being homed. A subclass adds the state of the
        command classes it owns: the active tool, the active coordinate
        system and the part.
        """
        self.homed = previous.homed


StateT = TypeVar("StateT", bound=MachineState)


@dataclass(slots=True)
class CommandContext(Generic[StateT]):
    """Everything a command handler needs, besides the raw argument tuple."""

    tag: TagLike
    state: StateT
    registry: CommandRegistry
    #: The ``Scanning`` class's real-hardware seam (see
    #: :mod:`pyippdme.server.backend`); ``None`` if the server wasn't given
    #: one - only a problem for a handler that actually needs it (a
    #: ``Scanning``-class one), which should check and raise a clear error
    #: rather than let it surface as an ``AttributeError``.
    backend: MachineBackend | None
    #: Persistence for named coordinate systems (6.5.2); see :mod:`pyippdme.types.csy`.
    csy_store: CsyStore
    #: Set when this specific transaction must stop as soon as possible
    #: (e.g. the client sent ``AbortE()``); see :mod:`pyippdme.server.backend`.
    cancel: CancellationToken
    #: The synthetic part ``PtMeas`` measures against, if any - see
    #: :mod:`pyippdme.server.surface`. ``None`` (the default) keeps
    #: ``PtMeas`` reporting the commanded position exactly, as if nothing
    #: were ever really touched.
    sample_surface: SampleSurface | None = None
    #: The simulated optical sensor ``DataAcquire`` uses, if any; see
    #: :class:`~pyippdme.server.surface.RawSensor`.
    raw_sensor: RawSensor | None = None
    #: What carries out a move (time, limits, collisions); see
    #: :class:`~pyippdme.server.motion.MotionModel`. ``None``: moves are instant.
    motion: MotionModel | None = None
    #: Where a handler that has to open its own connection or listener (the
    #: raw-data binary socket, 6.17.2.1) gets it from; the server's own
    #: :class:`~pyippdme.protocol.network.Network`.
    network: Network = TCP_NETWORK
    #: Sends one extra data message tagged with something other than this
    #: transaction's own ``tag`` - the seam a handler uses to push an
    #: ``OnMoveReport``/``OnMoveReportE`` daemon's report (see
    #: :mod:`pyippdme.simulation.classes.mover_class`) alongside its own normal
    #: response, without that daemon's ``EventTag`` being confused for this
    #: command's completion. ``None`` in a context built outside a real
    #: connection (e.g. a unit test), where there is nowhere for it to go.
    emit_event: Callable[[TagLike, DataPayload], Awaitable[None]] | None = None


CommandHandler: TypeAlias = Callable[
    [CommandContext[Any], tuple[Argument, ...]], Awaitable[HandlerResult]
]


class CommandRegistry:
    """Maps method names to :data:`CommandHandler` callables.

    Populated by default with nothing - :class:`~pyippdme.server.IppDmeServer`
    registers whatever ``command_classes`` it was given, typically starting
    with the mandatory ``Server``/``DME`` classes from
    :mod:`pyippdme.server.classes` (see :mod:`pyippdme.simulation` for this
    library's own bundled simulated classes, none of which are registered
    unless asked for). Additional command classes can be registered the
    same way, either by calling :meth:`register` directly or via the
    :meth:`command` decorator. VDMA 8722 6.1 requires any command not
    defined by the standard to be prefixed with a two-character company
    namespace; :meth:`register_proprietary`/ :meth:`command_proprietary`
    build and validate that for you (see :mod:`pyippdme.protocol.namespace`)
    instead of registering a bare name that might collide with a future
    standard command.

    ``arguments`` is optional, spec-derived metadata (see
    :mod:`pyippdme.protocol.signature`) used only to answer
    ``GetSupportedArguments(CommandName)`` (6.4.1); a command registered
    without it still works normally, ``GetSupportedArguments`` just cannot
    describe it (0506 "Argument not supported"). Pass ``arguments=()``
    (rather than omitting the keyword) for a command that is fully
    described and genuinely takes no arguments at all (``Home()``,
    ``StartSession()``, ...) - :meth:`arguments` distinguishes "no schema
    on record" (``None``) from "on record, and it is empty" (``()``), so
    ``GetSupportedArguments`` can report the latter as a real, supported,
    zero-argument command instead of "not supported".
    """

    def __init__(self) -> None:
        self._handlers: dict[str, CommandHandler] = {}
        self._signatures: dict[str, tuple[Parameter, ...]] = {}
        self._property_setters: list[PropertySetHandler] = []
        self._property_getters: list[PropertyGetHandler] = []
        self._property_children: list[PropertyChildrenHandler] = []
        self._daemon_stoppers: list[DaemonStopHandler] = []
        self._session_start_hooks: list[SessionLifecycleHandler] = []
        self._session_end_hooks: list[SessionLifecycleHandler] = []
        self._clear_errors_hooks: list[SessionLifecycleHandler] = []
        self._home_hooks: list[SessionLifecycleHandler] = []

    def register(
        self,
        name: str,
        handler: CommandHandler,
        *,
        arguments: tuple[Parameter, ...] | None = None,
    ) -> None:
        self._handlers[name] = handler
        if arguments is None:
            self._signatures.pop(name, None)
        else:
            self._signatures[name] = arguments

    def register_proprietary(
        self,
        namespace: str,
        name: str,
        handler: CommandHandler,
        *,
        arguments: tuple[Parameter, ...] | None = None,
    ) -> None:
        """Register a command under a two-letter company namespace (6.1)."""
        self.register(proprietary_name(namespace, name), handler, arguments=arguments)

    def unregister(self, name: str) -> None:
        """Remove a command, e.g. to drop one a chosen built-in class doesn't need.

        A no-op if ``name`` isn't registered.
        """
        self._handlers.pop(name, None)
        self._signatures.pop(name, None)

    def command(
        self, name: str, *, arguments: tuple[Parameter, ...] | None = None
    ) -> Callable[[CommandHandler], CommandHandler]:
        """Register as a decorator; ``arguments`` means the same thing as in :meth:`register`."""

        def decorator(handler: CommandHandler) -> CommandHandler:
            self.register(name, handler, arguments=arguments)
            return handler

        return decorator

    def command_proprietary(
        self, namespace: str, name: str, *, arguments: tuple[Parameter, ...] | None = None
    ) -> Callable[[CommandHandler], CommandHandler]:
        """Register as a decorator; ``arguments`` is as in :meth:`register_proprietary`."""

        def decorator(handler: CommandHandler) -> CommandHandler:
            self.register_proprietary(namespace, name, handler, arguments=arguments)
            return handler

        return decorator

    def register_property_resolver(
        self,
        *,
        setter: PropertySetHandler | None = None,
        getter: PropertyGetHandler | None = None,
    ) -> None:
        """Let a command class handle a subset of ``SetProp``/``GetProp`` properties.

        ``SetProp(Tool.PtMeasPar.Retract.Act(5))`` and
        ``SetProp(Part.Temperature(21))`` both dispatch through the single
        ``Server`` class ``SetProp``/``GetProp`` handlers
        (:mod:`pyippdme.server.classes.server_class`), which is the only place the
        wire format actually names - the standard does not prefix
        commands by class. Rather than every property-owning class
        (:mod:`pyippdme.simulation.classes.tool_class`,
        :mod:`pyippdme.simulation.classes.part_class`, ...) overriding those
        two handlers wholesale, which would make each
        newly-registered class silently clobber the previous one's property
        handling, each contributes a resolver here instead; the ``Server``
        handlers try them in registration order and fall back to the plain
        per-connection property store (:func:`~pyippdme.server._util.generic_set_prop`/
        :func:`~pyippdme.server._util.generic_get_prop`) if none matches.
        """
        if setter is not None:
            self._property_setters.append(setter)
        if getter is not None:
            self._property_getters.append(getter)

    def property_setters(self) -> tuple[PropertySetHandler, ...]:
        return tuple(self._property_setters)

    def property_getters(self) -> tuple[PropertyGetHandler, ...]:
        return tuple(self._property_getters)

    def register_property_children(self, handler: PropertyChildrenHandler) -> None:
        """Let a command class answer ``EnumProp``/``EnumAllProp`` for a subset of references.

        Mirrors :meth:`register_property_resolver`: ``EnumProp``/``EnumAllProp``
        dispatch through the single ``Server`` class handlers (6.3.1.1), and
        each property-owning class contributes what it knows about its own
        references here instead of those handlers hard-coding every class's
        object tree.
        """
        self._property_children.append(handler)

    def property_children_handlers(self) -> tuple[PropertyChildrenHandler, ...]:
        return tuple(self._property_children)

    def register_daemon_stopper(self, handler: DaemonStopHandler) -> None:
        """Let a command class handle ``StopDaemon``/``StopAllDaemons`` (6.3.1) for its own daemons.

        Mirrors :meth:`register_property_resolver`: ``StopDaemon``/
        ``StopAllDaemons`` dispatch through the single ``Server`` class
        handlers (:mod:`pyippdme.server.classes.server_class`), which have no
        daemon of their own to stop - only a class that actually starts one
        (e.g. :mod:`pyippdme.simulation.classes.mover_class`'s
        ``OnMoveReport``/``OnMoveReportE``) does, so ``Server`` dispatches to
        whichever registered stopper(s) recognize the target instead of
        hard-coding a specific daemon-owning class by name.
        """
        self._daemon_stoppers.append(handler)

    def daemon_stoppers(self) -> tuple[DaemonStopHandler, ...]:
        return tuple(self._daemon_stoppers)

    def register_session_start_hook(self, handler: SessionLifecycleHandler) -> None:
        """Let a command class react to ``StartSession()`` (6.3), e.g. resetting a modal flag.

        ``StartSession`` itself (:mod:`pyippdme.server.classes.server_class`)
        knows nothing about any specific class's state - a class whose
        commands establish a documented "reset to this on ``StartSession``"
        default (e.g. ``Alignable_AB``'s ``UseSmallestAngletoAlignTool``)
        registers a hook here instead of ``server_class`` reaching into it
        directly.
        """
        self._session_start_hooks.append(handler)

    def session_start_hooks(self) -> tuple[SessionLifecycleHandler, ...]:
        return tuple(self._session_start_hooks)

    def register_session_end_hook(self, handler: SessionLifecycleHandler) -> None:
        """Let a command class react to ``EndSession()`` (6.3), e.g. releasing held resources.

        Mirrors :meth:`register_session_start_hook`; see e.g.
        :mod:`pyippdme.simulation.classes.rawdata_class`'s buffered
        acquisitions, which ``EndSession()`` "implicitly deletes" (6.15.1).
        """
        self._session_end_hooks.append(handler)

    def session_end_hooks(self) -> tuple[SessionLifecycleHandler, ...]:
        return tuple(self._session_end_hooks)

    def register_clear_errors_hook(self, handler: SessionLifecycleHandler) -> None:
        """Let a command class react to ``ClearAllErrors()`` (6.3), e.g. resetting a modal flag.

        Mirrors :meth:`register_session_start_hook`.
        """
        self._clear_errors_hooks.append(handler)

    def clear_errors_hooks(self) -> tuple[SessionLifecycleHandler, ...]:
        return tuple(self._clear_errors_hooks)

    def register_home_hook(self, handler: SessionLifecycleHandler) -> None:
        """Let a command class react to ``Home()`` (6.4), e.g. ``Mover``'s implicit ``DisableUser``.

        Mirrors :meth:`register_session_start_hook`: ``Home()``
        (:mod:`pyippdme.server.classes.dme_class`) is mandatory and knows
        nothing about any specific class's state, but the standard says it
        triggers side effects on other classes anyway (6.7.1's "Server
        Remarks" under ``DisableUser()``: implicitly called by any command
        that physically moves the machine, ``Home()`` included).
        """
        self._home_hooks.append(handler)

    def home_hooks(self) -> tuple[SessionLifecycleHandler, ...]:
        return tuple(self._home_hooks)

    def get(self, name: str) -> CommandHandler | None:
        return self._handlers.get(name)

    def arguments(self, name: str) -> tuple[Parameter, ...] | None:
        """Return ``name``'s declared :class:`~pyippdme.protocol.signature.Parameter` tuple.

        ``None`` means no schema is on record at all; ``()`` means one is,
        and it says ``name`` takes no arguments - see :meth:`register`.
        """
        return self._signatures.get(name)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._handlers))


def commands_of(register_fn: Callable[[CommandRegistry], None]) -> tuple[str, ...]:
    """Return the command names ``register_fn`` would register, without a real registry.

    There is no separate, hand-maintained list of "Scanning's commands" (or
    any other class's) anywhere - a ``register_*_class`` function's own body
    *is* that list, so this reads it from the one place it can't drift out
    of sync: by actually calling it against a scratch registry.
    """
    scratch = CommandRegistry()
    register_fn(scratch)
    return scratch.names()


def component_name(register_fn: Callable[[CommandRegistry], None]) -> str:
    """Return the short name a ``register_*_class`` function stands for, e.g. ``"cartcmm"``.

    Read from the module ``register_fn`` is *defined* in
    (``pyippdme.simulation.classes.cartcmm_class`` -> ``"cartcmm"``), not its
    ``__name__`` - every one of these functions is named plainly
    ``register`` in its own module and only gets an alias like
    ``register_cartcmm_class`` when re-exported from
    :mod:`pyippdme.server.classes`/:mod:`pyippdme.simulation.classes`, and
    ``__name__`` doesn't follow that alias. Used wherever a caller needs to
    refer to "which command classes" by name (CLI flags, a machine-class
    string derived from them) without a second, hand-maintained
    name-to-function table that could drift from what it actually names.
    """
    module_name = register_fn.__module__.rsplit(".", 1)[-1]
    return module_name.removesuffix("_class")


def register_subset(
    registry: CommandRegistry,
    register_fn: Callable[[CommandRegistry], None],
    keep: Iterable[str],
) -> None:
    """Register everything ``register_fn`` defines, then drop whatever isn't in ``keep``.

    For picking which commands *within* one command class a particular
    server instance actually supports (e.g. a vision-only ``Scanning``
    backend that does ``ScanOnLine`` but not ``ScanOnCircle``/``ScanOnHelix``)
    - :class:`~pyippdme.server.IppDmeServer`'s own ``command_classes``
    parameter already covers picking whole classes; this is the same idea
    one level finer, without hand-writing a matching
    :meth:`CommandRegistry.unregister` call per command you don't want.
    ``GetSupportedCommands()``/
    ``GetSupportedArguments()`` (6.4.1) reflect the result correctly, since
    they read the registry directly - nothing dropped here is ever
    registered in the first place from a caller's perspective.

    Commands ``register_fn`` would have registered anyway (because ``keep``
    included them, or because they were already present before this call)
    are left untouched either way.
    """
    before = frozenset(registry.names())
    register_fn(registry)
    added = frozenset(registry.names()) - before
    for name in added - frozenset(keep):
        registry.unregister(name)
