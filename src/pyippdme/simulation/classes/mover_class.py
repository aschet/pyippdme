# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""The ``Mover`` command class (6.7.1).

``EnableUser``/``DisableUser``/``IsUserEnabled`` model the "manual jog box"
enable state as plain boolean ``MachineState`` fields. Per 6.7.1's own
"Server Remarks" under ``DisableUser()``, the server calls it implicitly
whenever the client calls a command that physically moves the machine; this
project wires that up for every command on the standard's list (``Home``,
``GoTo``/``GoToOnCircle``/``GoToOnSpiral``/``Step``/``PtMeas``, ``ChangeTool``,
``AlignTool``, ``Tool.A``/``B``/``C`` and every ``ScanOn...`` command) by
setting ``user_enabled = False`` directly in those handlers rather than
routing through this module.

``EnumerateMoverAxes()`` reports only ``X``/``Y``/``Z``: the axes this
project's simulated ``CartCMM`` actually moves (see
:mod:`pyippdme.simulation.classes.cartcmm_class`), not the standard's full axis
vocabulary (``R``/``A``/``B``/``C``) that ``FormTester``'s ``LockAxis``
merely accepts and stores.

The scale-temperature commands are a plain per-axis float store (no real
temperature sensors), keyed by any of ``X``/``Y``/``Z``/``R``/``A``/``B``/
``C`` (matching ``LockAxis``'s axis vocabulary, since scale temperature is a
property of an axis's encoder, not of whether that axis physically moves in
this simulation). ``SetTemperatureCompensationOrigin`` is restricted to
``X``/``Y``/``Z`` per its own definition. Per its "Client Remarks"
("after changing the active coordinate system... the origin has to be set
again, otherwise the zero point... will be used"),
:mod:`pyippdme.simulation.classes.cartcmm_class`'s ``SetCoordSystem`` handler clears it.

``OnMoveReport``/``OnMoveReportE`` (6.10.2, Table 68) start a "daemon" that reports
requested axes each time the machine moves - :func:`report_move` is the
seam other modules call into (:mod:`pyippdme.simulation.classes.cartcmm_class`'s
``GoTo``/``PtMeas``/``SetCoordSystem``,
:mod:`pyippdme.simulation.classes.toolchanger_class`'s ``ChangeTool``, per
6.10.2's own "Server Remarks" list of triggers - not every move-adjacent
command has been wired to it, just those). The real command is time/
distance-triggered ("a report is sent every `Time` seconds" or every `Dis`
millimeters moved); since every move this simulation performs is
instantaneous, there is no interval for that to fire within, so `Time`/
`Dis` are accepted and validated but otherwise unused - one report is sent
per triggering command instead. A requested ``Tool.A``/``Tool.B``/``Tool.C``
that the active tool has no rotation axis for is reported as the standard's
own ``NULL`` ("If the client requests properties or information which the
server cannot deliver, such as non-existing orientation axes of a tool, the
server responds NULL"). Per 5.5.2, the client itself must
send ``OnMoveReport``/``OnMoveReportE`` tagged with the
:class:`~pyippdme.protocol.ast.EventTag` its reports should arrive under
(see :meth:`~pyippdme.client.IppDmeClient.start_daemon`) - a plain numbered
``Tag`` is rejected (``0502``). Only one daemon can be active at a time
(``0515 Daemon exists already`` otherwise, matching 6.10.2's own "Client
Remarks": a client must ``StopDaemon()`` before starting a
differently-configured one anyway).
"""

from __future__ import annotations

from pyippdme.protocol.ast import (
    Argument,
    BasicName,
    DataPayload,
    EventTag,
    Items,
    NamedValue,
    Number,
)
from pyippdme.protocol.commands import CommandName
from pyippdme.protocol.errors import ErrorCode, ErrorSeverity, ServerError
from pyippdme.protocol.signature import DataType, Parameter
from pyippdme.server import builders
from pyippdme.server._util import named_number
from pyippdme.server.motion import TemperatureProvider
from pyippdme.server.registry import CommandRegistry, HandlerResult
from pyippdme.simulation.classes.tool_class import is_alignable_tool, tool_axis_value
from pyippdme.simulation.context import Ctx
from pyippdme.simulation.state import MoveReportDaemon

_TEMPERATURES_PARAMS = (Parameter("Temperatures", DataType.ENUM),)
_AXES_PARAMS = (Parameter("Axes", DataType.ENUM),)
# SetTemperatureCompensationOrigin's own table types this [float], not
# [enum] like the other "one or more axes" commands above, and its Kind is
# N (named): the client supplies a subset of X/Y/Z, each independently
# optional (this handler's own "at least one" check enforces the rest).
_SET_TEMPERATURE_COMPENSATION_ORIGIN_PARAMS = (
    Parameter("X", DataType.FLOAT, mandatory=False),
    Parameter("Y", DataType.FLOAT, mandatory=False),
    Parameter("Z", DataType.FLOAT, mandatory=False),
)

#: Axes this project actually moves; all EnumerateMoverAxes() reports.
_MOVABLE_AXES = ("X", "Y", "Z")
#: Axes a scale temperature may be tracked for (an encoder/scale property,
#: not tied to whether this simulation physically moves that axis).
_TEMPERATURE_AXES = frozenset({"X", "Y", "Z", "R", "A", "B", "C"})
_DEFAULT_SCALE_TEMPERATURE = 20.0
#: Table 68: the shortest interval between two move reports, in seconds.
_MIN_REPORT_INTERVAL = 0.1

_ON_MOVE_REPORT_PARAMS = (
    Parameter("Time", DataType.FLOAT),
    Parameter("Dis", DataType.FLOAT),
    Parameter("Axes", DataType.ENUM),
)


def _axis_not_found(cause: str, name: str) -> ServerError:
    return ServerError(
        ErrorSeverity.FATAL, ErrorCode.AXIS_DOES_NOT_EXIST, cause, f"Axis does not exist: {name}"
    )


async def _enable_user(ctx: Ctx, _args: tuple[Argument, ...]) -> HandlerResult:
    ctx.state.mover.user_enabled = True
    return None


async def _disable_user(ctx: Ctx, _args: tuple[Argument, ...]) -> HandlerResult:
    ctx.state.mover.user_enabled = False
    return None


async def _is_user_enabled(ctx: Ctx, _args: tuple[Argument, ...]) -> HandlerResult:
    return builders.named_boolean(CommandName.IS_USER_ENABLED, ctx.state.mover.user_enabled)


async def _enumerate_mover_axes(_ctx: Ctx, _args: tuple[Argument, ...]) -> HandlerResult:
    return builders.bare_names(*_MOVABLE_AXES)


async def _update_scale_temperatures(ctx: Ctx, _args: tuple[Argument, ...]) -> HandlerResult:
    # Take the temperatures of the scales from the sensors; a model without real sensors has none.
    if isinstance(ctx.motion, TemperatureProvider):
        for axis in ("X", "Y", "Z"):
            ctx.state.mover.scale_temperatures[axis] = ctx.motion.scale_temperature(axis)
    return None


async def _set_scale_temperatures(ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
    if not args:
        raise ServerError(
            ErrorSeverity.CRITICAL,
            ErrorCode.INCORRECT_ARGUMENTS,
            CommandName.SET_SCALE_TEMPERATURES,
            "Incorrect arguments",
        )
    updates: dict[str, float] = {}
    for arg in args:
        if (
            not isinstance(arg, NamedValue)
            or len(arg.args) != 1
            or not isinstance(arg.args[0], Number)
        ):
            raise ServerError(
                ErrorSeverity.CRITICAL,
                ErrorCode.BAD_ARGUMENT,
                CommandName.SET_SCALE_TEMPERATURES,
                "Bad argument",
            )
        if arg.name not in _TEMPERATURE_AXES:
            raise _axis_not_found(CommandName.SET_SCALE_TEMPERATURES, arg.name)
        updates[arg.name] = arg.args[0].value
    ctx.state.mover.scale_temperatures.update(updates)
    return None


async def _get_scale_temperatures(ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
    if not args:
        raise ServerError(
            ErrorSeverity.CRITICAL,
            ErrorCode.INCORRECT_ARGUMENTS,
            CommandName.GET_SCALE_TEMPERATURES,
            "Incorrect arguments",
        )
    results: list[NamedValue] = []
    for arg in args:
        if not isinstance(arg, NamedValue) or arg.args:
            raise ServerError(
                ErrorSeverity.CRITICAL,
                ErrorCode.BAD_ARGUMENT,
                CommandName.GET_SCALE_TEMPERATURES,
                "Bad argument",
            )
        if arg.name not in _TEMPERATURE_AXES:
            raise _axis_not_found(CommandName.GET_SCALE_TEMPERATURES, arg.name)
        temperature = ctx.state.mover.scale_temperatures.get(arg.name, _DEFAULT_SCALE_TEMPERATURE)
        results.append(NamedValue(arg.name, (Number.of(temperature),)))
    return Items(tuple(results))


async def _set_temperature_compensation_origin(
    ctx: Ctx, args: tuple[Argument, ...]
) -> HandlerResult:
    if not args:
        raise ServerError(
            ErrorSeverity.CRITICAL,
            ErrorCode.INCORRECT_ARGUMENTS,
            CommandName.SET_TEMPERATURE_COMPENSATION_ORIGIN,
            "Incorrect arguments",
        )
    updates: dict[str, float] = {}
    for arg in args:
        if (
            not isinstance(arg, NamedValue)
            or len(arg.args) != 1
            or not isinstance(arg.args[0], Number)
        ):
            raise ServerError(
                ErrorSeverity.CRITICAL,
                ErrorCode.BAD_ARGUMENT,
                CommandName.SET_TEMPERATURE_COMPENSATION_ORIGIN,
                "Bad argument",
            )
        if arg.name not in _MOVABLE_AXES:
            raise _axis_not_found(CommandName.SET_TEMPERATURE_COMPENSATION_ORIGIN, arg.name)
        updates[arg.name] = arg.args[0].value
    ctx.state.mover.temperature_compensation_origin.update(updates)
    return None


async def _on_move_report(ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
    if not isinstance(ctx.tag, EventTag):
        # 5.5.2: a daemon-starting command is itself sent tagged
        # with the EventTag its later reports will use (see
        # IppDmeClient.start_daemon()) - a plain numbered Tag has nowhere
        # for those reports to go.
        raise ServerError(
            ErrorSeverity.CRITICAL,
            ErrorCode.INCORRECT_ARGUMENTS,
            CommandName.ON_MOVE_REPORT,
            "Must be sent tagged with an EventTag",
        )
    if ctx.state.mover.report_daemon is not None:
        raise ServerError(
            ErrorSeverity.ERROR,
            ErrorCode.DAEMON_ALREADY_EXISTS,
            CommandName.ON_MOVE_REPORT,
            "Daemon exists already",
        )
    time_value = named_number(args, "Time")
    dis_value = named_number(args, "Dis")
    if time_value is None or dis_value is None:
        raise ServerError(
            ErrorSeverity.CRITICAL,
            ErrorCode.INCORRECT_ARGUMENTS,
            CommandName.ON_MOVE_REPORT,
            "Incorrect arguments",
        )
    if time_value < _MIN_REPORT_INTERVAL:
        # Table 68: "Time must be greater or equal 0.1".
        raise ServerError(
            ErrorSeverity.CRITICAL,
            ErrorCode.INCORRECT_ARGUMENTS,
            CommandName.ON_MOVE_REPORT,
            "Time must be at least 0.1 seconds",
        )
    axes: list[str] = []
    for arg in args:
        if not isinstance(arg, NamedValue) or arg.name in ("Time", "Dis"):
            continue
        if arg.args:
            raise ServerError(
                ErrorSeverity.CRITICAL,
                ErrorCode.BAD_PROPERTY,
                CommandName.ON_MOVE_REPORT,
                "Bad property",
            )
        axes.append(arg.name)
    if not axes:
        raise ServerError(
            ErrorSeverity.CRITICAL,
            ErrorCode.BAD_PROPERTY,
            CommandName.ON_MOVE_REPORT,
            "The enumeration may not be empty",
        )
    ctx.state.mover.report_daemon = MoveReportDaemon(tag=ctx.tag, axes=tuple(axes))
    return None


def _report_axis_value(ctx: Ctx, name: str) -> float | None:
    if name == "R":
        return ctx.state.rotary_table.position
    if name in _MOVABLE_AXES:
        index = _MOVABLE_AXES.index(name)
        return ctx.state.cart_cmm.position[index]
    if name in ("Tool.A", "Tool.B", "Tool.C"):
        # Only an alignable tool has rotation axes, and the catalog's has just A and B.
        if not is_alignable_tool(ctx.state.tool.active_name) or name == "Tool.C":
            return None
        return tool_axis_value(ctx, name, CommandName.ON_MOVE_REPORT)
    return None  # unknown to this simulation: reported as NULL


async def report_move(ctx: Ctx) -> None:
    """Push one ``OnMoveReport``/``OnMoveReportE`` report, if a daemon is currently active.

    See the module docstring for which commands call this, and for the
    "one report per trigger, not per Time/Dis interval" simplification.
    """
    daemon = ctx.state.mover.report_daemon
    if daemon is None or ctx.emit_event is None:
        return
    values = tuple(
        NamedValue(name, (Number.of(value) if value is not None else BasicName("NULL"),))
        for name in daemon.axes
        for value in (_report_axis_value(ctx, name),)
    )
    payload: DataPayload = Items(values)
    await ctx.emit_event(daemon.tag, payload)


def try_stop_daemon(ctx: Ctx, tag: EventTag) -> bool:
    """Stop the active ``OnMoveReport`` daemon if ``tag`` matches it; return whether it did."""
    daemon = ctx.state.mover.report_daemon
    if daemon is not None and daemon.tag == tag:
        ctx.state.mover.report_daemon = None
        return True
    return False


def stop_all_daemons(ctx: Ctx) -> bool:
    """Stop every active daemon (just the one kind, for now); return whether any were active."""
    if ctx.state.mover.report_daemon is not None:
        ctx.state.mover.report_daemon = None
        return True
    return False


def _stop_daemon_hook(ctx: Ctx, tag: EventTag | None) -> bool:
    """Adapt :func:`try_stop_daemon`/:func:`stop_all_daemons` to :data:`DaemonStopHandler`."""
    if tag is None:
        return stop_all_daemons(ctx)
    return try_stop_daemon(ctx, tag)


def _on_home(ctx: Ctx) -> None:
    # Mover 6.7.1's own "Server Remarks" under DisableUser(): implicitly
    # called by Home() (dme_class, mandatory), which has no state of its
    # own to set this on.
    ctx.state.mover.user_enabled = False


def register(registry: CommandRegistry) -> None:
    registry.register(CommandName.ENABLE_USER, _enable_user, arguments=())
    registry.register(CommandName.DISABLE_USER, _disable_user, arguments=())
    registry.register(CommandName.IS_USER_ENABLED, _is_user_enabled, arguments=())
    registry.register(CommandName.ENUMERATE_MOVER_AXES, _enumerate_mover_axes, arguments=())
    registry.register(
        CommandName.UPDATE_SCALE_TEMPERATURES, _update_scale_temperatures, arguments=()
    )
    registry.register(
        CommandName.SET_SCALE_TEMPERATURES, _set_scale_temperatures, arguments=_TEMPERATURES_PARAMS
    )
    registry.register(
        CommandName.GET_SCALE_TEMPERATURES, _get_scale_temperatures, arguments=_AXES_PARAMS
    )
    registry.register(
        CommandName.SET_TEMPERATURE_COMPENSATION_ORIGIN,
        _set_temperature_compensation_origin,
        arguments=_SET_TEMPERATURE_COMPENSATION_ORIGIN_PARAMS,
    )
    registry.register(CommandName.ON_MOVE_REPORT, _on_move_report, arguments=_ON_MOVE_REPORT_PARAMS)
    registry.register(
        CommandName.ON_MOVE_REPORT_E, _on_move_report, arguments=_ON_MOVE_REPORT_PARAMS
    )
    registry.register_daemon_stopper(_stop_daemon_hook)
    registry.register_home_hook(_on_home)
