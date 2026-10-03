# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""The ``Server`` command class (section 6.3), mandatory for all servers.

Covers session lifecycle, error status/handling, and the property-handling
commands (section 6.3.1.1: ``SetProp``/``GetProp``/``GetPropE``/
``EnumProp``/``EnumAllProp``). These are the only handlers registered for
those five command names - the standard does not prefix commands by class, so
per-property behaviour from other classes (:mod:`pyippdme.simulation.classes.tool_class`'s
``Tool.*`` parameter blocks, :mod:`pyippdme.simulation.classes.part_class`'s
``Part.*`` properties, ...) plugs in via
:meth:`~pyippdme.server.registry.CommandRegistry.register_property_resolver`/
:meth:`~pyippdme.server.registry.CommandRegistry.register_property_children`
instead of a second class overriding these handlers outright.
``SetProp``/``GetProp`` fall back to a simple generic per-connection
property store (:attr:`~pyippdme.server.registry.MachineState.properties`) for
anything no resolver recognizes, rather than validating against a full,
per-property schema; ``EnumProp``/``EnumAllProp`` have no such fallback -
see their own docstrings for the (deliberately bounded) set of references
this simulation can enumerate.

``StopDaemon``/``StopAllDaemons`` (6.3.1) work the same way: this class
owns no daemon of its own (there is nothing here that could be one - see
:mod:`pyippdme.server.registry`'s ``MachineState``), so it dispatches to
whichever registered class's
:meth:`~pyippdme.server.registry.CommandRegistry.register_daemon_stopper`
callback recognizes the target, e.g.
:mod:`pyippdme.simulation.classes.mover_class`'s ``OnMoveReport``/``OnMoveReportE``.
"""

from __future__ import annotations

from pyippdme.protocol.ast import Argument, EventTag, Items, NamedValue, Number, PropertyData
from pyippdme.protocol.commands import CommandName
from pyippdme.protocol.errors import ErrorCode, ErrorSeverity, ServerError, describe
from pyippdme.protocol.signature import DataType, Parameter
from pyippdme.server import builders
from pyippdme.server._util import bad_argument, generic_get_prop, generic_set_prop
from pyippdme.server.context import Ctx
from pyippdme.server.registry import CommandRegistry, HandlerResult, PropertyKind

_STOP_DAEMON_PARAMS = (Parameter("EventTag", DataType.NAME, positional=True),)
_GET_ERROR_INFO_PARAMS = (Parameter("ErrorNumber", DataType.INT, positional=True),)
_PROPERTIES_PARAMS = (Parameter("Properties", DataType.ENUM),)
_REFERENCE_PARAMS = (Parameter("Reference", DataType.NAME),)


async def _start_session(ctx: Ctx, _args: tuple[Argument, ...]) -> HandlerResult:
    if ctx.state.session_active:
        raise ServerError(
            ErrorSeverity.CRITICAL,
            ErrorCode.PROTOCOL_ERROR,
            CommandName.START_SESSION,
            "Protocol Error",
        )
    ctx.state.session_active = True
    ctx.state.active_error = None
    # No state of its own to reset beyond the above - a class with a
    # documented "StartSession() resets this" default (e.g. Alignable_AB's
    # UseSmallestAngletoAlignTool) registers its own hook instead.
    for hook in ctx.registry.session_start_hooks():
        hook(ctx)
    return None


async def _end_session(ctx: Ctx, _args: tuple[Argument, ...]) -> HandlerResult:
    ctx.state.session_active = False
    # 6.3.1: "The method must make sure that all daemons are stopped".
    _run_daemon_stoppers(ctx, None)
    # No resources of its own to release - a class holding session-scoped
    # resources (e.g. rawdata_class's buffered acquisitions, 6.15.1's
    # "EndSession() implicitly deletes all buffered acquisitions") registers
    # its own hook instead.
    for hook in ctx.registry.session_end_hooks():
        hook(ctx)
    return None


def _run_daemon_stoppers(ctx: Ctx, tag: EventTag | None) -> bool:
    """Call every registered daemon stopper; return whether any of them handled ``tag``.

    Not ``any(stopper(...) for stopper in ...)``: that would short-circuit
    and skip calling the remaining stoppers once one returns ``True``, but
    ``StopAllDaemons`` needs every one of them to actually run.
    """
    stopped = False
    for stopper in ctx.registry.daemon_stoppers():
        if stopper(ctx, tag):
            stopped = True
    return stopped


async def _stop_daemon(ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
    if len(args) != 1 or not isinstance(args[0], EventTag):
        raise bad_argument(CommandName.STOP_DAEMON, "Expected StopDaemon(<EventTag>)")
    # No daemon-owning class of its own here (see the module docstring) -
    # dispatch to whichever registered class recognizes this tag, e.g.
    # mover_class's OnMoveReport()/OnMoveReportE().
    if not _run_daemon_stoppers(ctx, args[0]):
        raise ServerError(
            ErrorSeverity.ERROR,
            ErrorCode.DAEMON_DOES_NOT_EXIST,
            CommandName.STOP_DAEMON,
            "Daemon Does Not Exist",
        )
    return None


async def _stop_all_daemons(ctx: Ctx, _args: tuple[Argument, ...]) -> HandlerResult:
    if not _run_daemon_stoppers(ctx, None):
        raise ServerError(
            ErrorSeverity.WARNING,
            ErrorCode.NO_DAEMONS_ACTIVE,
            CommandName.STOP_ALL_DAEMONS,
            "No daemons are active",
        )
    return None


async def _abort_e_unreachable(_ctx: Ctx, _args: tuple[Argument, ...]) -> HandlerResult:
    # pyippdme.server._ServerConnection intercepts "AbortE" before it ever
    # reaches the registry lookup in _execute(), so this handler is never
    # actually called; it exists only so GetSupportedCommands() still lists
    # AbortE as a supported command.
    raise AssertionError("AbortE is handled at the connection level and should never dispatch here")


async def _get_error_info(_ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
    if len(args) != 1 or not isinstance(args[0], Number):
        raise ServerError(
            ErrorSeverity.CRITICAL,
            ErrorCode.BAD_ARGUMENT,
            CommandName.GET_ERROR_INFO,
            "Expected a single error number",
        )
    number = f"{int(args[0].value):04d}"
    entry = describe(number)
    description = entry[1] if entry is not None else "Unknown error"
    return builders.string_value(description)


async def _get_err_status(ctx: Ctx, _args: tuple[Argument, ...]) -> HandlerResult:
    return builders.boolean(ctx.state.active_error is not None)


async def _get_xtd_err_status(ctx: Ctx, _args: tuple[Argument, ...]) -> HandlerResult:
    err = ctx.state.active_error
    error = (int(err.number), int(err.severity)) if err is not None else None
    return builders.get_xtd_err_status(homed=ctx.state.homed, error=error)


async def _clear_all_errors(ctx: Ctx, _args: tuple[Argument, ...]) -> HandlerResult:
    ctx.state.active_error = None
    for hook in ctx.registry.clear_errors_hooks():
        hook(ctx)
    return None


async def _enum_name_spaces(_ctx: Ctx, _args: tuple[Argument, ...]) -> HandlerResult:
    return ()  # No proprietary namespaces are defined by this server.


async def _set_prop(ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
    for arg in args:
        if not isinstance(arg, NamedValue):
            raise ServerError(
                ErrorSeverity.CRITICAL, ErrorCode.BAD_PROPERTY, CommandName.SET_PROP, "Bad property"
            )
        if not any(setter(ctx, arg) for setter in ctx.registry.property_setters()):
            generic_set_prop(ctx, (arg,), cause=CommandName.SET_PROP)
    return None


async def _get_prop(ctx: Ctx, args: tuple[Argument, ...], cause: str) -> HandlerResult:
    results: list[NamedValue] = []
    for arg in args:
        if not isinstance(arg, NamedValue):
            raise ServerError(ErrorSeverity.CRITICAL, ErrorCode.BAD_PROPERTY, cause, "Bad property")
        resolved: NamedValue | None = None
        for getter in ctx.registry.property_getters():
            resolved = getter(ctx, arg)
            if resolved is not None:
                break
        if resolved is None:
            fallback = generic_get_prop(ctx, (arg,), cause=cause)
            assert isinstance(fallback, Items)  # noqa: S101 (narrows our own helper's return type)
            (resolved,) = fallback.values
        results.append(resolved)
    return Items(tuple(results))


async def _get_prop_cmd(ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
    return await _get_prop(ctx, args, CommandName.GET_PROP)


async def _get_prop_e_cmd(ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
    return await _get_prop(ctx, args, CommandName.GET_PROP_E)


def _reference_name(args: tuple[Argument, ...], cause: str) -> str:
    """Parse ``EnumProp``/``EnumAllProp``'s ``Reference`` (e.g. ``Tool.GoToPar()``, ``Part()``)."""
    if len(args) != 1 or not isinstance(args[0], NamedValue) or args[0].args:
        raise bad_argument(cause, "Expected a single Reference, e.g. Tool.GoToPar()")
    return args[0].name


def _children(ctx: Ctx, reference: str, cause: str) -> tuple[tuple[str, str], ...]:
    """Look up ``reference``'s children via every registered resolver, in order.

    Only references a registered :data:`~pyippdme.server.registry.PropertyChildrenHandler`
    recognizes can be enumerated - this simulation does not model every
    built-in class's properties as a literal tree, only the ones that are
    genuinely structured (``Tool``'s parameter blocks, ``Part``'s
    properties; see :mod:`pyippdme.simulation.classes.tool_class`/
    :mod:`pyippdme.simulation.classes.part_class`). Anything else raises ``0505``,
    same as an unresolvable ``SetProp``/``GetProp`` property name.
    """
    for handler in ctx.registry.property_children_handlers():
        found = handler(ctx, reference)
        if found is not None:
            return found
    raise ServerError(
        ErrorSeverity.CRITICAL,
        ErrorCode.ARGUMENT_NOT_RECOGNIZED,
        cause,
        f"Unknown reference {reference}",
    )


async def _enum_prop(ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
    reference = _reference_name(args, CommandName.ENUM_PROP)
    children = _children(ctx, reference, CommandName.ENUM_PROP)
    return [builders.property_entry(name, kind) for name, kind in children]


async def _enum_all_prop(ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
    reference = _reference_name(args, CommandName.ENUM_ALL_PROP)
    results: list[PropertyData] = []

    def walk(ref: str) -> None:
        for name, kind in _children(ctx, ref, CommandName.ENUM_ALL_PROP):
            results.append(builders.property_entry(name, kind))
            if kind == PropertyKind.PROPERTY:
                walk(f"{ref}.{name}")

    walk(reference)
    return results


def register(registry: CommandRegistry) -> None:
    registry.register(CommandName.START_SESSION, _start_session, arguments=())
    registry.register(CommandName.END_SESSION, _end_session, arguments=())
    registry.register(CommandName.STOP_DAEMON, _stop_daemon, arguments=_STOP_DAEMON_PARAMS)
    registry.register(CommandName.STOP_ALL_DAEMONS, _stop_all_daemons, arguments=())
    registry.register(CommandName.ABORT_E, _abort_e_unreachable, arguments=())
    registry.register(CommandName.GET_ERROR_INFO, _get_error_info, arguments=_GET_ERROR_INFO_PARAMS)
    registry.register(CommandName.GET_ERR_STATUS_E, _get_err_status, arguments=())
    registry.register(CommandName.GET_XTD_ERR_STATUS, _get_xtd_err_status, arguments=())
    registry.register(CommandName.CLEAR_ALL_ERRORS, _clear_all_errors, arguments=())
    registry.register(CommandName.ENUM_NAME_SPACES, _enum_name_spaces, arguments=())
    registry.register(CommandName.SET_PROP, _set_prop, arguments=_PROPERTIES_PARAMS)
    registry.register(CommandName.GET_PROP, _get_prop_cmd, arguments=_PROPERTIES_PARAMS)
    registry.register(CommandName.GET_PROP_E, _get_prop_e_cmd, arguments=_PROPERTIES_PARAMS)
    registry.register(CommandName.ENUM_PROP, _enum_prop, arguments=_REFERENCE_PARAMS)
    registry.register(CommandName.ENUM_ALL_PROP, _enum_all_prop, arguments=_REFERENCE_PARAMS)
