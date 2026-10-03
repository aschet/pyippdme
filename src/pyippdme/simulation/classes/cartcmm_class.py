# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""A subset of the ``CartCMM``/``Cartesian``/``TouchTrigger`` command classes.

Implements coordinate-system selection (6.5.2: ``SetCoordSystem``,
``GetCoordSystem``), the coordinate-system-transformation command group
(6.5.1/6.5.2: ``SetCsyTransformation``, ``GetCsyTransformation``,
``SaveNamedCsyTransformation``, ``GetNamedCsyTransformation``,
``SaveActiveCoordSystem``, ``LoadCoordSystem``, ``DeleteCoordSystem``,
``EnumCoordSystems``; see :mod:`pyippdme.types.csy` for the transform math and
persistence seam), axis queries (6.10.2: ``Get``), absolute motion (6.8.1:
``GoTo``), and single-point measurement (6.12.1: ``PtMeas``,
``OnPtMeasReport``) against a simulated machine: no real tool, only a
``(x, y, z)`` position that ``GoTo`` sets directly. This is enough to
exercise the protocol end-to-end; a real server would replace these
handlers with ones driving actual hardware. The coordinate-transformation
commands are stored and returned faithfully but - see :mod:`pyippdme.types.csy` -
are *not* applied to convert ``Get``/``GoTo`` coordinates between CSYs.

``PtMeas``'s approach/search/retract behaviour (6.12.1) *is* implemented,
but only takes effect when the server is configured with a
:class:`~pyippdme.server.surface.SampleSurface` - without one, there is
still no real part to touch, so ``PtMeas`` falls back to reporting the
commanded position exactly, as before. With a surface: the probing
direction is the given ``IJK``, or (6.12.1's own fallback) the direction
from the commanded point to wherever the machine was just before this
``PtMeas`` - normalized; a genuinely degenerate case (no ``IJK`` and no
prior motion to infer one from, e.g. the very first ``PtMeas`` of a
session) falls back the same way, since the spec's own fallback rule has
nothing to fall back to there either. The approach position is offset from
the commanded point by ``Part.Approach() + Tool.PtMeasPar.Approach() + Tool.AvrRadius()``
along that direction (the radius is zero unless the tool handler reports one); the search then
runs from there back past the commanded point by ``Tool.PtMeasPar.Search() + Part.Search()``,
and the first surface contact within that range is the measured point - reported instead of
the commanded one - or ``1006 Surface not found`` if the surface never comes into range.
``Tool.PtMeasPar.Retract()`` then moves the machine's *resting* position
(what later ``Get``/``GoTo`` see) away from the contact point, per 6.12.1's
own sign convention, without changing what was already reported as
measured.

Also implements the temperature-sensor commands (``GetTemperatureSensors``,
``ReadTemperatureSensor``, ``ReadAllTemperatures``) against a small fixed
catalog of simulated sensors (:data:`_TEMPERATURE_SENSORS`) rather than
real hardware: one ``"Part"``-kind sensor that reads
:mod:`pyippdme.simulation.classes.part_class`'s ``Part.Temperature`` (so setting that
property is observable here too), one ``"CMM"``-kind ambient sensor fixed
at 20.0, and one ``"Mover"``-kind sensor per axis reading
:mod:`pyippdme.simulation.classes.mover_class`'s per-axis scale temperatures. None of
the simulated sensors is ever "defect or not connected", so the ``-273``
sentinel value never appears and ``ReadAllTemperatures()`` always reports
every sensor.

Also implements ``Step`` (6.8.1: a relative move, like ``GoTo`` but adding
to the current position rather than replacing it) and
``GoToOnCircle``/``GoToOnSpiral`` (6.8.1: an absolute move to a point on a
circle/spiral defined by ``Center``/``IJK`` - validated to actually lie in
that plane for ``GoToOnCircle`` via the same orthogonality check
``Scanning``'s known-contour commands use, see
:func:`pyippdme.server._util.check_orthogonal`; ``GoToOnSpiral`` has no
such check, since a spiral's target legitimately leaves the plane). Like
``GoTo``, both jump straight to the target rather than simulating the
circular/spiral path itself - there is no real motion to interpolate.

``GoTo``/``Step``/``PtMeas`` take everything their tables allow: ``X``/``Y``/``Z``,
the rotary table ``R``, the tool angles ``Tool.A``/``Tool.B``/``Tool.C``,
``Tool.Alignment`` (not for ``Step``), ``Sync`` (not for ``PtMeas``) and, for
``PtMeas``, ``IJK`` and ``AlignPart``. The tool angles and ``Tool.Alignment`` move
the same orientation ``AlignTool()`` sets (see
:mod:`pyippdme.simulation.classes.tool_class`), and ``AlignPart`` is the one of
:mod:`pyippdme.simulation.classes.rotarytable_class`. An argument a command
does not take is rejected with ``0506``, except a proprietary one. ``Get``
answers ``Tool.A``/``Tool.B``/``Tool.C`` (and the ``FoundTool`` ones) and ``IJK``,
the direction of the tool.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from pyippdme.protocol.ast import Argument, BasicName, Items, NamedValue, Number, String
from pyippdme.protocol.commands import CommandName
from pyippdme.protocol.errors import ErrorCode, ErrorSeverity, ServerError
from pyippdme.protocol.parameters import ParameterName
from pyippdme.protocol.signature import DataType, Parameter
from pyippdme.server import builders
from pyippdme.server._util import (
    bad_argument,
    check_orthogonal,
    incorrect_arguments,
    named_number,
    named_vector,
    single_basic_name,
)
from pyippdme.server.motion import MotionError, MotionRequest, ProbeRequest
from pyippdme.server.registry import CommandRegistry, HandlerResult, PropertyKind
from pyippdme.server.surface import SampleSurface
from pyippdme.simulation.classes.mover_class import report_move
from pyippdme.simulation.classes.rotarytable_class import apply_part_alignment
from pyippdme.simulation.classes.tool_class import (
    apply_tool_orientation,
    require_measuring_tool,
    tool_alignment_numbers,
    tool_avr_radius,
    tool_axis_value,
)
from pyippdme.simulation.context import Ctx, csy_context
from pyippdme.types.csy import CSY_CHAIN, LIVE_TRANSFORM_NAMES, CoordinateTransform, CsyContext
from pyippdme.types.vec3 import Vec3, add, norm, normalize, scale, sub

_SET_COORD_SYSTEM_PARAMS = (Parameter("Csy", DataType.NAME, positional=True),)
_AXES_PARAMS = (Parameter("Axes", DataType.ENUM),)
_POSITION_PARAMS = (Parameter("Position", DataType.ENUM),)
# GoTo's Sync isn't read by this handler (rotary-table/multi-arm
# synchronization isn't simulated), but it is still a real, spec-defined
# optional parameter for GetSupportedArguments purposes.
_GO_TO_PARAMS = (
    Parameter("Positions", DataType.ENUM),
    Parameter("Sync", DataType.INT, mandatory=False),
)
_ON_PT_MEAS_REPORT_PARAMS = (Parameter("Parameters", DataType.ENUM),)
_STEP_PARAMS = (
    Parameter("Distances", DataType.ENUM),
    Parameter("Sync", DataType.INT, mandatory=False),
)
_GO_TO_ON_CIRCLE_OR_SPIRAL_PARAMS = (
    Parameter("Center", DataType.FLOAT),
    Parameter("IJK", DataType.FLOAT),
    Parameter("X", DataType.FLOAT),
    Parameter("Y", DataType.FLOAT),
    Parameter("Z", DataType.FLOAT),
)
#: Table 60: "Raised if Tool.Alignment is specified as argument" to Step() -
#: not applicable, since Tool.Alignment always defines an absolute
#: orientation and Step() is a relative move.
_TOOL_ALIGNMENT_PROPERTY = "Tool.Alignment"

_CSY_TRANSFORM_PARAM_NAMES = ("X0", "Y0", "Z0", "Theta", "Psi", "Phi")
# _parse_csy_transform_args reads all 7 as bare, ordered values - Csy/Name
# is not wrapped in its own name either, unlike GoTo-style commands.
_SET_CSY_TRANSFORMATION_PARAMS = (
    Parameter("Csy", DataType.NAME, positional=True),
    *(Parameter(n, DataType.FLOAT, positional=True) for n in _CSY_TRANSFORM_PARAM_NAMES),
)
_SAVE_NAMED_CSY_TRANSFORMATION_PARAMS = (
    Parameter("Name", DataType.STRING, positional=True),
    *(Parameter(n, DataType.FLOAT, positional=True) for n in _CSY_TRANSFORM_PARAM_NAMES),
)
_GET_CSY_TRANSFORMATION_PARAMS = (Parameter("Csy", DataType.NAME, positional=True),)
_NAME_PARAMS = (Parameter("Name", DataType.STRING, positional=True),)

#: The CSY that SaveActiveCoordSystem()/LoadCoordSystem() (6.5.2) save/restore.
_PART_CSY = "PartCsy"
#: ``SetCoordSystem`` (6.5.2) chooses where a client enters the transformation chain of Figure 12.
_VALID_CSY = CSY_CHAIN

_AXES = ("X", "Y", "Z")
_DEFAULT_PT_MEAS_REPORT = _AXES
#: Fields ``OnPtMeasReport()`` (6.12.1's own Table 75) accepts, beyond the
#: axes ``Get()`` already covers. ``IJK`` is the unit probing direction (the
#: given ``IJK``, or 6.12.1's own motion-vector fallback - see :func:`_pt_meas`)
#: and ``IJKAct`` says it is the nominal one (Table 32), since this
#: simulation never adjusts the probing direction once determined.
#: ``Tool.Alignment`` and the tool angles report what ``AlignTool()`` set.
_PT_MEAS_REPORT_EXTRA_NAMES = (
    "R",
    "Q",
    "ER",
    "IJK",
    "IJKAct",
    "Tool.Alignment",
    "FoundTool.Alignment",
    "Tool.A",
    "Tool.B",
    "Tool.C",
    "FoundTool.A",
    "FoundTool.B",
    "FoundTool.C",
)
_PT_MEAS_REPORT_NAMES = (*_AXES, *_PT_MEAS_REPORT_EXTRA_NAMES)
#: ``IJKAct`` value (Table 32) for "IJK is the nominal vector".
_IJK_ACT_NOMINAL = 2
#: A proprietary argument name starts with a two-letter company namespace (6.1).
_PROPRIETARY_ARGUMENT_RE = re.compile(r"^[A-Z]{2}[A-Za-z0-9]+$")

_READ_TEMPERATURE_SENSOR_PARAMS = (Parameter("Name", DataType.STRING, positional=True),)

#: A sensor that could not be read reports this ("defect or not connected", Table 44).
_TEMPERATURE_SENSOR_DISCONNECTED = -273.0
#: Every simulated sensor's own default reading, absent any state to read
#: from (matches ``mover_class``'s ``_DEFAULT_SCALE_TEMPERATURE`` and
#: ``part_class``'s ``Part.Temperature`` default - both 20.0 "room
#: temperature", so a fresh session's sensors agree with each other).
_DEFAULT_TEMPERATURE = 20.0
_PART_TEMPERATURE_PROPERTY = "Part.Temperature"


@dataclass(frozen=True, slots=True)
class _TemperatureSensor:
    """One entry of :data:`_TEMPERATURE_SENSORS`."""

    name: str
    #: One of Table 43's ``TemperatureSensor`` enum values: ``Mover``, ``Part``, ``CMM``.
    kind: str
    #: Whether the controller itself already compensates for this sensor's reading.
    cmm_temp_correction: bool
    #: The axis this sensor measures, for ``Mover``-kind sensors only.
    scale_axis: str | None
    read: Callable[[Ctx], float]


def _axis_temperature_sensor(axis: str) -> _TemperatureSensor:
    def read(ctx: Ctx) -> float:
        return ctx.state.mover.scale_temperatures.get(axis, _DEFAULT_TEMPERATURE)

    return _TemperatureSensor(
        f"{axis}AxisSensor", "Mover", cmm_temp_correction=False, scale_axis=axis, read=read
    )


#: The fixed set of simulated temperature sensors (6.5.2); see the module
#: docstring for the simulation's scope. Read functions are late-bound
#: (called with the current :class:`CommandContext`) rather than snapshotted,
#: so e.g. ``SetProp(Part.Temperature(...))`` is reflected immediately.
_TEMPERATURE_SENSORS = (
    _TemperatureSensor(
        "PartSensor",
        "Part",
        cmm_temp_correction=False,
        scale_axis=None,
        read=lambda ctx: ctx.state.part.properties.get(
            _PART_TEMPERATURE_PROPERTY, _DEFAULT_TEMPERATURE
        ),
    ),
    _TemperatureSensor(
        "CMMSensor",
        "CMM",
        cmm_temp_correction=False,
        scale_axis=None,
        read=lambda _ctx: _DEFAULT_TEMPERATURE,
    ),
    *(_axis_temperature_sensor(axis) for axis in _AXES),
)


def _axis_index(name: str) -> int | None:
    return _AXES.index(name) if name in _AXES else None


def _keep_position(ctx: Ctx, before: CsyContext) -> None:
    """Re-express the stored position after the coordinate system or its placement changed.

    The machine does not move when a client activates another coordinate system or moves one;
    only the numbers that describe where it stands change (6.5.2).
    """
    after = csy_context(ctx)
    position = ctx.state.cart_cmm.position
    ctx.state.cart_cmm.position = after.to_client(before.to_machine(position))


async def _set_coord_system(ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
    name = single_basic_name(args, CommandName.SET_COORD_SYSTEM)
    if name not in _VALID_CSY:
        raise ServerError(
            ErrorSeverity.CRITICAL,
            ErrorCode.ARGUMENT_NOT_RECOGNIZED,
            CommandName.SET_COORD_SYSTEM,
            f"Unknown coordinate system {name}",
        )
    before = csy_context(ctx)
    ctx.state.cart_cmm.active_csy = name
    _keep_position(ctx, before)
    # Mover 6.7.1 SetTemperatureCompensationOrigin's Client Remarks: "After
    # changing the active coordinate system... the origin has to be set
    # again, otherwise the zero point of the active coordinate system will
    # be used" - i.e. a coordinate system change clears any explicit origin.
    ctx.state.mover.temperature_compensation_origin.clear()
    # 6.10.2's own "Server Remarks" under OnMoveReport(): a report is
    # also due after a "virtual movement of the tool", which explicitly
    # includes SetCoordSystem().
    await report_move(ctx)
    return None


async def _get_coord_system(ctx: Ctx, _args: tuple[Argument, ...]) -> HandlerResult:
    return builders.name_value(ctx.state.cart_cmm.active_csy)


_TOOL_AXES = ("Tool.A", "Tool.B", "Tool.C", "FoundTool.A", "FoundTool.B", "FoundTool.C")


async def _get(ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
    if not args:
        raise ServerError(
            ErrorSeverity.CRITICAL,
            ErrorCode.INCORRECT_ARGUMENTS,
            CommandName.GET,
            "Incorrect arguments",
        )
    results: list[NamedValue] = []
    for arg in args:
        if not isinstance(arg, NamedValue) or arg.args:
            raise ServerError(
                ErrorSeverity.CRITICAL, ErrorCode.BAD_ARGUMENT, CommandName.GET, "Bad argument"
            )
        if arg.name == "R":
            # RotaryTable 6.23.1: "This method can only be invoked as an
            # argument of a Get or OnReport command."
            results.append(
                NamedValue(ParameterName.R, (Number.of(ctx.state.rotary_table.position),))
            )
        elif arg.name in _TOOL_AXES:
            # Alignable_AB/Alignable_ABC 6.20.2/6.21.2: "The Property needs to be
            # prepended by Tool or FoundTool".
            value = tool_axis_value(ctx, arg.name, CommandName.GET)
            results.append(NamedValue(arg.name, (Number.of(value),)))
        elif arg.name == ParameterName.IJK:
            # Table 30: "Current direction vector of the active tool".
            numbers = tool_alignment_numbers(ctx, CommandName.TOOL, CommandName.GET)
            results.append(NamedValue(arg.name, tuple(Number.of(c) for c in numbers[:3])))
        else:
            index = _axis_index(arg.name)
            if index is None:
                raise ServerError(
                    ErrorSeverity.CRITICAL,
                    ErrorCode.ARGUMENT_NOT_SUPPORTED,
                    CommandName.GET,
                    f"Argument {arg.name} not supported",
                )
            results.append(NamedValue(arg.name, (Number.of(ctx.state.cart_cmm.position[index]),)))
    return Items(tuple(results))


@dataclass(frozen=True, slots=True)
class _Motion:
    """The parsed arguments of ``GoTo``/``Step``/``PtMeas`` (Tables 59, 60, 76)."""

    x: float | None = None
    y: float | None = None
    z: float | None = None
    r: float | None = None
    a: float | None = None
    b: float | None = None
    c: float | None = None
    alignment: tuple[float, ...] | None = None
    align_part: tuple[float, ...] | None = None
    ijk: Vec3 | None = None


def _parse_motion(
    args: tuple[Argument, ...],
    cause: str,
    *,
    allow_alignment: bool = True,
    allow_sync: bool = True,
    allow_pt_meas: bool = False,
) -> _Motion:
    """Parse the argument list of a move; raise ``0506`` for anything the command doesn't take."""
    singles: dict[str, float] = {}
    vectors: dict[str, tuple[float, ...]] = {}
    for arg in args:
        if not isinstance(arg, NamedValue):
            raise bad_argument(cause, "Expected named arguments")
        name = arg.name
        numbers = tuple(a.value for a in arg.args if isinstance(a, Number))
        if len(numbers) != len(arg.args):
            raise bad_argument(cause, f"{name} takes numbers only")
        if name in ("X", "Y", "Z", "R", "Tool.A", "Tool.B", "Tool.C"):
            if len(numbers) != 1:
                raise bad_argument(cause, f"Expected {name}(<number>)")
            singles[name] = numbers[0]
        elif name == _TOOL_ALIGNMENT_PROPERTY and allow_alignment:
            vectors[name] = numbers
        elif name == "Sync" and allow_sync:
            if len(numbers) != 1 or numbers[0] not in (0.0, 1.0):
                raise bad_argument(cause, "Expected Sync(0) or Sync(1)")
        elif allow_pt_meas and name in ("IJK", "AlignPart", "LMN"):
            vectors[name] = numbers
        elif _PROPRIETARY_ARGUMENT_RE.match(name) is None:
            raise ServerError(
                ErrorSeverity.CRITICAL,
                ErrorCode.ARGUMENT_NOT_SUPPORTED,
                cause,
                f"{name} is not supported"
                + (
                    " (Tool.Alignment always defines an absolute orientation, Table 60)"
                    if name == _TOOL_ALIGNMENT_PROPERTY
                    else ""
                ),
            )
    ijk = vectors.get("IJK")
    if ijk is not None and len(ijk) != 3:
        raise bad_argument(cause, "Expected IJK(<i>,<j>,<k>)")
    return _Motion(
        singles.get("X"),
        singles.get("Y"),
        singles.get("Z"),
        singles.get("R"),
        singles.get("Tool.A"),
        singles.get("Tool.B"),
        singles.get("Tool.C"),
        vectors.get(_TOOL_ALIGNMENT_PROPERTY),
        vectors.get("AlignPart"),
        (ijk[0], ijk[1], ijk[2]) if ijk is not None else None,
    )


def _targets(ctx: Ctx, motion: _Motion, *, relative: bool) -> tuple[Vec3, float]:
    """Where a parsed move ends: the linear position and the rotary table angle."""
    locked = ctx.state.form_tester.locked_axes

    def moved(current: float, axis: str, value: float | None) -> float:
        if value is None or axis in locked:
            return current
        return current + value if relative else value

    x, y, z = ctx.state.cart_cmm.position
    target = (moved(x, "X", motion.x), moved(y, "Y", motion.y), moved(z, "Z", motion.z))
    rotary = ctx.state.rotary_table.position
    rotary_target = rotary
    if motion.r is not None and "R" not in locked:
        rotary_target = rotary + motion.r if relative else motion.r
    return target, rotary_target


async def _move(
    ctx: Ctx,
    motion: _Motion,
    cause: str,
    *,
    relative: bool,
    end_offset: Vec3 = (0.0, 0.0, 0.0),
) -> None:
    """Carry out a parsed move: position, rotary table and tool orientation.

    ``end_offset`` shifts the end of the linear move (``PtMeas`` stops at the approach position).
    """
    locked = ctx.state.form_tester.locked_axes
    # Locked axes (FormTester's LockAxis, 6.6.1) are silently ignored, even
    # when a value is given for them - "without causing an error".
    apply_tool_orientation(
        ctx,
        cause,
        a=None if "A" in locked else motion.a,
        b=None if "B" in locked else motion.b,
        c=None if "C" in locked else motion.c,
        alignment=motion.alignment,
        relative=relative,
    )

    x, y, z = ctx.state.cart_cmm.position
    target, rotary_target = _targets(ctx, motion, relative=relative)
    target = add(target, end_offset)
    rotary = ctx.state.rotary_table
    if ctx.motion is not None:
        request = MotionRequest(
            cause=cause,
            start=(x, y, z),
            end=target,
            rotary_start=rotary.position,
            rotary_end=rotary_target,
            homed=ctx.state.homed,
            tool_name=ctx.state.tool.active_name,
            cancel=ctx.cancel,
        )
        try:
            target = await ctx.motion.travel(request)
        except MotionError as error:
            ctx.state.cart_cmm.position = error.stopped_at
            raise
    ctx.state.cart_cmm.position = target
    # RotaryTable 6.23.1 R(r): "can only be invoked as an argument of a
    # GoTo, PtMeas or ScanOnCurve command"; the shortest-distance-move and
    # 180-degree-ambiguity rules it describes are a real motion controller's
    # concern, not modeled by this simulation.
    rotary.position = rotary_target
    # Mover 6.7.1: GoTo(), Step() and PtMeas() implicitly execute DisableUser().
    ctx.state.mover.user_enabled = False
    # Mover 6.10.2's OnMoveReport() daemon, if any (see mover_class's module docstring).
    await report_move(ctx)


async def _go_to(ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
    await _move(ctx, _parse_motion(args, CommandName.GO_TO), CommandName.GO_TO, relative=False)
    return None


async def _on_pt_meas_report(ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
    names: list[str] = []
    for arg in args:
        if not isinstance(arg, NamedValue) or arg.args:
            raise ServerError(
                ErrorSeverity.CRITICAL,
                ErrorCode.BAD_PROPERTY,
                CommandName.ON_PT_MEAS_REPORT,
                "Bad property",
            )
        if arg.name not in _PT_MEAS_REPORT_NAMES:
            # Same convention as Scanning's OnScanReport (see its module
            # docstring): a field this simulation cannot genuinely compute
            # is rejected up front rather than silently reported as 0.
            raise ServerError(
                ErrorSeverity.CRITICAL,
                ErrorCode.BAD_PROPERTY,
                CommandName.ON_PT_MEAS_REPORT,
                f"{arg.name} is not allowed in a report",
            )
        names.append(arg.name)
    if not names:
        raise ServerError(
            ErrorSeverity.CRITICAL,
            ErrorCode.BAD_PROPERTY,
            CommandName.ON_PT_MEAS_REPORT,
            "The enumeration may not be empty",
        )
    ctx.state.cart_cmm.pt_meas_report = tuple(names)
    return None


async def _step(ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
    # Table 60: Tool.Alignment "always defines an absolute orientation" and so
    # is "Argument not supported" for a relative move (raised by _parse_motion).
    motion = _parse_motion(args, CommandName.STEP, allow_alignment=False)
    await _move(ctx, motion, CommandName.STEP, relative=True)
    return None


async def _go_to_on_circle_or_spiral(
    ctx: Ctx, args: tuple[Argument, ...], *, cause: str, check_plane: bool
) -> HandlerResult:
    center = named_vector(args, "Center")
    normal = named_vector(args, "IJK")
    x = named_number(args, "X")
    y = named_number(args, "Y")
    z = named_number(args, "Z")
    if center is None or normal is None or x is None or y is None or z is None:
        raise bad_argument(cause, "Expected Center(x,y,z), IJK(i,j,k), X(x), Y(y), Z(z)")
    target = (x, y, z)
    if check_plane:
        start = ctx.state.cart_cmm.position
        check_plane_text = (
            "IJK must be orthogonal to the center-to-{}-position vector (not on the plane)"
        )
        check_orthogonal(cause, normal, sub(target, center), check_plane_text.format("target"))
        check_orthogonal(cause, normal, sub(start, center), check_plane_text.format("start"))
    ctx.state.cart_cmm.position = target
    # Mover 6.7.1: GoToOnCircle()/GoToOnSpiral() implicitly execute
    # DisableUser(), like every other implemented motion command.
    ctx.state.mover.user_enabled = False
    return None


async def _go_to_on_circle(ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
    return await _go_to_on_circle_or_spiral(
        ctx, args, cause=CommandName.GO_TO_ON_CIRCLE, check_plane=True
    )


async def _go_to_on_spiral(ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
    # Unlike GoToOnCircle, Table 62 defines no "target off-plane" error - a
    # spiral's target legitimately advances along the normal each turn.
    return await _go_to_on_circle_or_spiral(
        ctx, args, cause=CommandName.GO_TO_ON_SPIRAL, check_plane=False
    )


def _parse_csy_transform_args(
    args: tuple[Argument, ...], cause: str
) -> tuple[str, CoordinateTransform]:
    """Parse ``(Csy|Name, X0, Y0, Z0, Theta, Psi, Phi)``, as shared by several commands.

    The leading argument is a bare name for ``Csy [name]`` commands and a
    quoted string for ``Name [string]`` ones (``SaveNamedCsyTransformation``
    and friends); both are accepted here and the caller picks the right
    parameter schema/error semantics for ``GetSupportedArguments``.
    """
    if len(args) != 7 or not isinstance(args[0], String | BasicName):
        raise bad_argument(cause)
    numbers = args[1:]
    if not all(isinstance(a, Number) for a in numbers):
        raise bad_argument(cause)
    x0, y0, z0, theta, psi, phi = (a.value for a in numbers if isinstance(a, Number))
    return args[0].value, CoordinateTransform(x0, y0, z0, theta, psi, phi)


def _require_theta_in_range(transform: CoordinateTransform, cause: str) -> None:
    if not transform.is_theta_in_range:
        raise ServerError(
            ErrorSeverity.CRITICAL, ErrorCode.THETA_OUT_OF_RANGE, cause, "Theta out of range"
        )


async def _set_csy_transformation(ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
    csy, transform = _parse_csy_transform_args(args, CommandName.SET_CSY_TRANSFORMATION)
    if csy not in LIVE_TRANSFORM_NAMES:
        raise ServerError(
            ErrorSeverity.CRITICAL,
            ErrorCode.ARGUMENT_NOT_RECOGNIZED,
            CommandName.SET_CSY_TRANSFORMATION,
            f"Unknown coordinate system {csy}",
        )
    _require_theta_in_range(transform, CommandName.SET_CSY_TRANSFORMATION)
    before = csy_context(ctx)
    ctx.state.cart_cmm.csy_transformations[csy] = transform
    _keep_position(ctx, before)
    return None


async def _get_csy_transformation(ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
    csy = single_basic_name(args, CommandName.GET_CSY_TRANSFORMATION)
    transform = ctx.state.cart_cmm.csy_transformations.get(csy)
    if transform is None:
        raise ServerError(
            ErrorSeverity.CRITICAL,
            ErrorCode.COORDINATE_SYSTEM_NOT_FOUND,
            CommandName.GET_CSY_TRANSFORMATION,
            "Coordinate system not found",
        )
    return builders.csy_transformation(transform)


async def _save_named_csy_transformation(ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
    name, transform = _parse_csy_transform_args(args, CommandName.SAVE_NAMED_CSY_TRANSFORMATION)
    _require_theta_in_range(transform, CommandName.SAVE_NAMED_CSY_TRANSFORMATION)
    await ctx.csy_store.save(name, transform)
    return None


async def _get_named_csy_transformation(ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
    if len(args) != 1 or not isinstance(args[0], String):
        raise bad_argument(CommandName.GET_NAMED_CSY_TRANSFORMATION)
    transform = await ctx.csy_store.load(args[0].value)
    if transform is None:
        raise ServerError(
            ErrorSeverity.CRITICAL,
            ErrorCode.COORDINATE_SYSTEM_NOT_FOUND,
            CommandName.GET_NAMED_CSY_TRANSFORMATION,
            "Coordinate system not found",
        )
    return builders.csy_transformation(transform)


async def _save_active_coord_system(ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
    if len(args) != 1 or not isinstance(args[0], String):
        raise bad_argument(CommandName.SAVE_ACTIVE_COORD_SYSTEM)
    transform = ctx.state.cart_cmm.csy_transformations.get(
        _PART_CSY, CoordinateTransform.identity()
    )
    await ctx.csy_store.save(args[0].value, transform)
    return None


async def _load_coord_system(ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
    if len(args) != 1 or not isinstance(args[0], String):
        raise bad_argument(CommandName.LOAD_COORD_SYSTEM)
    transform = await ctx.csy_store.load(args[0].value)
    if transform is None:
        raise ServerError(
            ErrorSeverity.CRITICAL,
            ErrorCode.COORDINATE_SYSTEM_NOT_FOUND,
            CommandName.LOAD_COORD_SYSTEM,
            "Coordinate system not found",
        )
    before = csy_context(ctx)
    ctx.state.cart_cmm.csy_transformations[_PART_CSY] = transform
    _keep_position(ctx, before)
    return None


async def _delete_coord_system(ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
    if len(args) != 1 or not isinstance(args[0], String):
        raise bad_argument(CommandName.DELETE_COORD_SYSTEM)
    existed = await ctx.csy_store.delete(args[0].value)
    if not existed:
        raise ServerError(
            ErrorSeverity.CRITICAL,
            ErrorCode.COORDINATE_SYSTEM_NOT_FOUND,
            CommandName.DELETE_COORD_SYSTEM,
            "Coordinate system not found",
        )
    return None


async def _enum_coord_systems(ctx: Ctx, _args: tuple[Argument, ...]) -> HandlerResult:
    return builders.string_list(await ctx.csy_store.names())


async def _pt_meas(ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
    require_measuring_tool(ctx, CommandName.PT_MEAS)
    motion = _parse_motion(args, CommandName.PT_MEAS, allow_pt_meas=True)
    if motion.ijk is not None and motion.x is None and motion.y is None and motion.z is None:
        # 6.12.1: "The argument IJK(..) without any linear axis coordinates is currently
        # not allowed."
        raise incorrect_arguments(CommandName.PT_MEAS, "IJK needs at least one of X, Y and Z")
    if motion.align_part is not None:
        apply_part_alignment(ctx, motion.align_part, CommandName.PT_MEAS)
    previous_position = ctx.state.cart_cmm.position
    if ctx.motion is not None:
        nominal_position, _ = _targets(ctx, motion, relative=False)
        direction = (
            motion.ijk if motion.ijk is not None else sub(nominal_position, previous_position)
        )
        try:
            unit_direction = normalize(direction)
        except ValueError:
            unit_direction = None
        if unit_direction is not None:
            return await _probe_with_motion(ctx, motion, nominal_position, unit_direction)
    await _move(ctx, motion, CommandName.PT_MEAS, relative=False)
    nominal_position = ctx.state.cart_cmm.position

    ijk = motion.ijk
    direction = ijk if ijk is not None else sub(nominal_position, previous_position)
    try:
        unit_direction = normalize(direction)
    except ValueError:
        # Degenerate (6.12.1's own fallback has nothing to fall back to
        # either, e.g. the very first PtMeas of a session with no IJK and
        # no prior motion) - report "no direction known" rather than raise,
        # since IJK is an optional report field, not a mandatory one.
        unit_direction = (0.0, 0.0, 0.0)

    report_position = nominal_position
    if ctx.sample_surface is not None:
        probed = _probe_surface(ctx, ctx.sample_surface, nominal_position, direction)
        if probed is not None:
            report_position, final_position = probed
            ctx.state.cart_cmm.position = final_position

    return Items(pt_meas_fields(ctx, report_position, unit_direction, CommandName.PT_MEAS))


async def _probe_with_motion(ctx: Ctx, motion: _Motion, nominal: Vec3, unit: Vec3) -> HandlerResult:
    """``PtMeas`` with a motion model: go to the approach position, then run the probing cycle.

    The machine moves to ``nominal + unit * approach`` like any other move; the motion model
    then searches for the surface (6.12.1), which can take time, find the part early or late, or
    fail with ``1006``.
    """
    parameters = ctx.state.tool.parameters.pt_meas_par
    approach = (
        ctx.state.part.properties.get("Part.Approach", 0.0)
        + parameters["Approach"].value
        + tool_avr_radius(ctx)
    )
    await _move(ctx, motion, CommandName.PT_MEAS, relative=False, end_offset=scale(unit, approach))
    assert ctx.motion is not None  # noqa: S101 (checked by the caller)
    result = await ctx.motion.probe(
        ProbeRequest(
            cause=CommandName.PT_MEAS,
            nominal=nominal,
            direction=unit,
            approach=approach,
            search=parameters["Search"].value + ctx.state.part.properties.get("Part.Search", 0.0),
            retract=parameters["Retract"].value,
            speed=parameters["Speed"].value,
            accel=parameters["Accel"].value,
            tool_name=ctx.state.tool.active_name,
            cancel=ctx.cancel,
        )
    )
    ctx.state.cart_cmm.position = result.rest
    return Items(pt_meas_fields(ctx, result.contact, unit, CommandName.PT_MEAS))


def pt_meas_fields(
    ctx: Ctx, position: Vec3, unit_direction: Vec3, cause: str
) -> tuple[NamedValue, ...]:
    """Build the fields ``OnPtMeasReport`` asked for, as ``PtMeas`` and the point events send."""
    results: list[NamedValue] = []
    for name in ctx.state.cart_cmm.pt_meas_report or _DEFAULT_PT_MEAS_REPORT:
        if name == ParameterName.IJK:
            results.append(NamedValue(name, tuple(Number.of(c) for c in unit_direction)))
        elif name == ParameterName.IJK_ACT:
            # Table 32: 1 = measured normal, 2 = nominal vector, 3 = tool alignment. The
            # direction reported is the nominal probing direction.
            results.append(NamedValue(name, (Number.of(_IJK_ACT_NOMINAL),)))
        elif name.endswith(".Alignment"):
            numbers = tool_alignment_numbers(ctx, name.partition(".")[0], cause)
            results.append(NamedValue(name, tuple(Number.of(c) for c in numbers)))
        elif name in _TOOL_AXES:
            value = tool_axis_value(ctx, name, cause)
            results.append(NamedValue(name, (Number.of(value),)))
        else:
            if name == "R":
                value = ctx.state.rotary_table.position
            elif name in ("Q", "ER"):
                value = 0.0  # no real contact/geometry assessment - see _PT_MEAS_REPORT_NAMES
            else:
                index = _axis_index(name)
                value = position[index] if index is not None else 0.0
            results.append(NamedValue(name, (Number.of(value),)))
    return tuple(results)


def _probe_surface(
    ctx: Ctx, surface: SampleSurface, nominal_position: Vec3, direction: Vec3
) -> tuple[Vec3, Vec3] | None:
    """Run 6.12.1's approach/search/retract algorithm; return ``(measured, resting)`` positions.

    Returns ``None`` - falling back to the legacy exact-nominal-position
    report - if ``direction`` is degenerate (the zero vector): there is no
    probing direction to search along, and this is the only case 6.12.1's
    own IJK fallback rule ("direction from the nominal point to the last
    position") can't resolve either (e.g. the very first ``PtMeas`` of a
    session, with no ``IJK`` given and no prior motion).
    """
    try:
        unit = normalize(direction)
    except ValueError:
        return None

    approach_distance = (
        ctx.state.part.properties.get("Part.Approach", 0.0)
        + ctx.state.tool.parameters.pt_meas_par["Approach"].value
        + tool_avr_radius(ctx)
    )
    search_distance = ctx.state.tool.parameters.pt_meas_par["Search"].value + (
        ctx.state.part.properties.get("Part.Search", 0.0)
    )
    retract_distance = ctx.state.tool.parameters.pt_meas_par["Retract"].value

    approach_position = add(nominal_position, scale(unit, approach_distance))
    search_direction = scale(unit, -1.0)
    max_search_length = approach_distance + search_distance

    contact = surface.intersect(approach_position, search_direction)
    if contact is None or norm(sub(contact, approach_position)) > max_search_length + 1e-6:
        raise ServerError(
            ErrorSeverity.ERROR,
            ErrorCode.SURFACE_NOT_FOUND,
            CommandName.PT_MEAS,
            "Surface not found",
        )

    if retract_distance >= 0:
        resting_position = add(contact, scale(unit, retract_distance))
    else:
        # 6.12.1's own remark on Tool.Retract(): negative means "move back
        # to the approach position" rather than shifting by that distance.
        resting_position = approach_position
    return contact, resting_position


async def _pt_meas_self_center(ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
    """``PtMeasSelfCenter(X(),Y(),Z(),IJK())`` (Table 94): a self-centering ``PtMeas`` variant.

    No real free-axis centering ("deepest point in a cone or inside
    sphere") is simulated - there is no part geometry to search a cone
    against, only a single probing line (see module docstring). This
    degenerates to exactly :func:`_pt_meas`'s own straight-line probe
    along the given ``IJK``, which is what a cone search collapses to once
    there is nothing but that one line to find contact on. Table 94 says
    the returned data is "as defined by OnScanReport", but this is
    structurally a single-point measurement, not a scan stream - treated
    as another instance of the standard's own internal inconsistency (see
    :mod:`pyippdme.protocol.ast`'s module docstring for others), so this
    reports via ``OnPtMeasReport``'s own configured fields instead, like
    ``PtMeas()`` itself.
    """
    return await _pt_meas(ctx, args)


async def _pt_meas_self_center_locked(ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
    """``PtMeasSelfCenterLocked(X(),Y(),Z(),IJK(),LMN())``: self-centering within a plane.

    ``LMN`` (the constraint plane's normal) must be orthogonal to ``IJK``
    (6.14.1's one explicit constraint); beyond that check, it plays no
    further role - like :func:`_pt_meas_self_center`, there is no real
    "search within a plane" to simulate, only the same single-line probe
    ``PtMeas()`` already does.
    """
    ijk = named_vector(args, "IJK")
    lmn = named_vector(args, "LMN")
    if ijk is not None and lmn is not None:
        check_orthogonal(
            CommandName.PT_MEAS_SELF_CENTER_LOCKED, ijk, lmn, "IJK must be orthogonal to LMN"
        )
    return await _pt_meas(ctx, args)


async def _get_temperature_sensors(_ctx: Ctx, _args: tuple[Argument, ...]) -> HandlerResult:
    return [
        builders.temperature_sensor(
            s.name, s.kind, cmm_temp_correction=s.cmm_temp_correction, scale_axis=s.scale_axis
        )
        for s in _TEMPERATURE_SENSORS
    ]


async def _read_temperature_sensor(ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
    if len(args) != 1 or not isinstance(args[0], String):
        raise bad_argument(CommandName.READ_TEMPERATURE_SENSOR)
    name = args[0].value
    sensor = next((s for s in _TEMPERATURE_SENSORS if s.name == name), None)
    value = _TEMPERATURE_SENSOR_DISCONNECTED if sensor is None else sensor.read(ctx)
    return builders.number(value)


async def _read_all_temperatures(ctx: Ctx, _args: tuple[Argument, ...]) -> HandlerResult:
    return [builders.temperature_reading(s.name, s.read(ctx)) for s in _TEMPERATURE_SENSORS]


def _cart_cmm_children(_ctx: Ctx, reference: str) -> tuple[tuple[str, str], ...] | None:
    """Answer ``EnumProp`` for ``CartCMM`` (6.5.3) and the classes without properties."""
    if reference == "CartCMM":
        return tuple(
            (name, PropertyKind.PROPERTY) for name in ("ToolChanger", "Tool", "RotaryTable", "Part")
        )
    if reference in ("ToolChanger", "RotaryTable"):
        return ()  # 6.22.2 and 6.23.2: these classes have no properties
    return None


#: Where ``Home()`` leaves the machine: "The home position for a given machine is fixed" (6.4.1).
HOME_POSITION: Vec3 = (0.0, 0.0, 0.0)


def _move_home(ctx: Ctx) -> None:
    ctx.state.cart_cmm.position = ctx.state.home_position or HOME_POSITION


def _reset_report_on_start_session(ctx: Ctx) -> None:
    # 6.3.1 StartSession(): "The default arguments for OnPtMeasReport are set to (X(),Y(),Z())".
    ctx.state.cart_cmm.pt_meas_report = ()


def register(registry: CommandRegistry) -> None:
    registry.register_session_start_hook(_reset_report_on_start_session)
    registry.register_property_children(_cart_cmm_children)
    registry.register_home_hook(_move_home)
    registry.register(
        CommandName.SET_COORD_SYSTEM, _set_coord_system, arguments=_SET_COORD_SYSTEM_PARAMS
    )
    registry.register(CommandName.GET_COORD_SYSTEM, _get_coord_system, arguments=())
    registry.register(
        CommandName.SET_CSY_TRANSFORMATION,
        _set_csy_transformation,
        arguments=_SET_CSY_TRANSFORMATION_PARAMS,
    )
    registry.register(
        CommandName.GET_CSY_TRANSFORMATION,
        _get_csy_transformation,
        arguments=_GET_CSY_TRANSFORMATION_PARAMS,
    )
    registry.register(
        CommandName.SAVE_NAMED_CSY_TRANSFORMATION,
        _save_named_csy_transformation,
        arguments=_SAVE_NAMED_CSY_TRANSFORMATION_PARAMS,
    )
    registry.register(
        CommandName.GET_NAMED_CSY_TRANSFORMATION,
        _get_named_csy_transformation,
        arguments=_NAME_PARAMS,
    )
    registry.register(
        CommandName.SAVE_ACTIVE_COORD_SYSTEM, _save_active_coord_system, arguments=_NAME_PARAMS
    )
    registry.register(CommandName.LOAD_COORD_SYSTEM, _load_coord_system, arguments=_NAME_PARAMS)
    registry.register(CommandName.DELETE_COORD_SYSTEM, _delete_coord_system, arguments=_NAME_PARAMS)
    registry.register(CommandName.ENUM_COORD_SYSTEMS, _enum_coord_systems, arguments=())
    registry.register(CommandName.GET, _get, arguments=_AXES_PARAMS)
    registry.register(CommandName.GO_TO, _go_to, arguments=_GO_TO_PARAMS)
    registry.register(
        CommandName.ON_PT_MEAS_REPORT, _on_pt_meas_report, arguments=_ON_PT_MEAS_REPORT_PARAMS
    )
    registry.register(CommandName.PT_MEAS, _pt_meas, arguments=_AXES_PARAMS)
    registry.register(
        CommandName.PT_MEAS_SELF_CENTER, _pt_meas_self_center, arguments=_POSITION_PARAMS
    )
    registry.register(
        CommandName.PT_MEAS_SELF_CENTER_LOCKED,
        _pt_meas_self_center_locked,
        arguments=_POSITION_PARAMS,
    )
    registry.register(CommandName.STEP, _step, arguments=_STEP_PARAMS)
    registry.register(
        CommandName.GO_TO_ON_CIRCLE,
        _go_to_on_circle,
        arguments=_GO_TO_ON_CIRCLE_OR_SPIRAL_PARAMS,
    )
    registry.register(
        CommandName.GO_TO_ON_SPIRAL,
        _go_to_on_spiral,
        arguments=_GO_TO_ON_CIRCLE_OR_SPIRAL_PARAMS,
    )
    registry.register(CommandName.GET_TEMPERATURE_SENSORS, _get_temperature_sensors, arguments=())
    registry.register(
        CommandName.READ_TEMPERATURE_SENSOR,
        _read_temperature_sensor,
        arguments=_READ_TEMPERATURE_SENSOR_PARAMS,
    )
    registry.register(CommandName.READ_ALL_TEMPERATURES, _read_all_temperatures, arguments=())
