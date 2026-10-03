# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""A subset of the ``Scanning`` command class (6.13.2.1-6.13.2.2: scans and hints).

``ScanOnCircle``/``ScanOnLine`` are the sharpest contrast in this project's
command set to the ``NamedValue``-wrapped style used everywhere else
(``GoTo(X(10), Y(20))``): their arguments are a long, fixed-order list of
*bare* numbers (``ScanOnLine(Sx, Sy, Sz, Ex, Ey, Ez, i, j, k, StepW, RT)``),
and their results are a variable-length stream of scanned points, each
itself a bare (unnamed) list of values in whatever order ``OnScanReport(...)``
last requested - the standard is explicit that "the report returns the
values of the enumerated parameters only and omits the name and braces,
e.g., it returns 500 only, instead of X(500)" (6.13.2), which is why the
data below is built from :class:`~pyippdme.protocol.ast.NumericData`
rather than the :class:`~pyippdme.protocol.ast.Items` used by ``Get``/
``PtMeas``. ``OnScanReport(...)`` only accepts field names this
simulation can genuinely report (see :data:`_SCAN_REPORT_NAMES`) -
requesting anything else is rejected up front, rather than silently
reporting 0 for a field that looks like real measured data. ``IJK``
reports the scan's own orientation argument - a constant
scanning-plane/cylinder-axis normal for ``ScanOnCircle``/``ScanOnLine``/
``ScanOnHelix``/the unknown-contour scans below, or (genuinely per-point,
since the client supplies it directly) each point's own ``IJK`` for
``ScanOnCurve`` - not a measured contact-surface normal, which no backend
here produces; ``IJKAct`` is the single number 2 that says so (Table 32).

Argument validation (degenerate geometry, orthogonality, StepW) happens
here, synchronously, before any points are produced. The points themselves
come from ``ctx.backend`` (see :mod:`pyippdme.server.backend`) as an async stream,
which is what lets ``AbortE()`` interrupt a scan mid-flight (5.7)
and lets a real backend transmit points as they are physically acquired
rather than all at once at the end.

As with :mod:`pyippdme.simulation.classes.cartcmm_class`, there is no real tool: no
approach/search/contact simulation, no rotary-table motion (the optional
``RT`` argument is accepted and ignored), and the ``*Hint``/``*Density``
commands are accepted and ignored (beyond validating the arguments they do
define, e.g. ``ScanUnknownDensity``'s ``Angle``/``AngleBaseLength``
co-occurrence rule), which the standard explicitly permits ("Hints are
optional; the DME must be able to execute without the interpretation of a
given hint").

``ScanOnHelix`` (Table 86) is ``ScanOnCircle`` with an added ``pitch``
argument: a constant rise along the scanning plane's normal per full turn,
computed by :meth:`~pyippdme.server.backend.MachineBackend.scan_helix`. Its ``sfa``
argument (surface angle) is validated per the one constraint Table 86
states outright (not 90 or 270 degrees); no simulated surface exists for it
to otherwise act on, matching how ``sfa`` is already accepted and unused by
``ScanOnCircle``.

All five unknown-contour scans (6.13.2.2: ``ScanInPlaneEndIsSphere``/
``EndIsPlane``/``EndIsCyl``, ``ScanInCylEndIsSphere``/``EndIsPlane``) are
implemented, deliberately simplified. "Unknown contour" scanning is
inherently adaptive - the server is meant to follow a real, physically
probed surface it does not know the shape of in advance - which this
project's simulation (no part geometry at all, see
:mod:`pyippdme.simulation.classes.cartcmm_class`) cannot do faithfully, and
attempting a genuine contour-tracing algorithm here would trade one
incomplete feature for another rather than giving users something they
can actually run their own client commands against today. Each command
instead runs a straight-line scan (reusing the same
``ctx.backend.scan_line`` geometry as ``ScanOnLine``) aimed directly at
the stop criterion's own reference point/center - the sphere's ``Ex, Ey,
Ez``, the plane's ``Px, Py, Pz``, or the stop cylinder's axis base point
``Cx, Cy, Cz`` - with that stop criterion then applied as a real,
per-point cutoff on top of the resulting stream (sphere/cylinder:
distance to center/axis within ``Dia``/``d``; plane: each point past the
plane, by sign of its distance to ``Pi,Pj,Pk``, counts once - the plane
line is extended a little past its own point first, since unlike a
sphere or cylinder a plane has no interior to linger in). ``Dx, Dy, Dz``
(the "direction point") is validated per each command's one explicit
constraint (non-coincident with the start point) but does not otherwise
steer the (straight-line) path, since the path already aims at a fixed
target derived from the stop criterion; the spec's own "start checking
the stop criterion only after moving further than dist(start, direction
point)" rule is likewise not modeled, since it exists for genuine
adaptive tracing this simplification does not attempt. The two
scanning-cylinder commands have no scanning-plane normal argument to
validate/report an orientation from, unlike the three scanning-plane
ones - the cylinder axis direction (``Ci,Cj,Ck``) is reused for
``IJK`` instead, the closest available orientation reference.

``ScanOnCurveHint``/``ScanOnCurveDensity`` (6.13.2, both hints) are
accepted and ignored like the other ``*Hint`` commands. ``ScanUnknownDensity``
(Table 88, the equivalent density hint for the unknown-contour scans below)
enforces its one real constraint - ``Angle``/``AngleBaseLength`` must be
given together or not at all - then is likewise accepted and ignored;
Table 88 marks ``Dis`` mandatory, but that reads as a table artifact given
the command's own "the server will use default values" remark and no
positional form to supply it through, so all three are treated as optional
(see :data:`_SCAN_UNKNOWN_DENSITY_PARAMS`).

``ScanOnCurve`` (Table 85) sends its nominal points as a single flat,
positional ``Data(...)`` list rather than nested groups - confirmed against
a real captured wire example (the NIST/I++ DME reference test suite's
``ScanOnCurve.prg``, e.g. ``ScanOnCurve(Closed(0),
Format(X(),Y(),Z(),IJK(),tag), Data(0.,0.,0., 0.,0.,1., 555, 100.,0.,0.,
0.,0.,1., 666, ...))``, since that reference project pre-dates this one and
is not treated as a source of truth for anything beyond wire syntax like
this). This implementation supports that ``Format`` plus the optional columns
Table 85 lists after ``tag``: the nominal tool directions ``pi,pj,pk`` and
``si,sj,sk`` (see ``AlignTool``) and the rotary-table angle ``R()``; the
directions are accepted and not acted on, and ``R`` moves the rotary table
when ``RT`` is 1. Like ``ScanOnLine``/``ScanOnCircle``, each nominal point is
echoed back as the "measured" one (no real part geometry), and ``Closed``
does not change the stream.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass

from pyippdme.protocol.ast import Argument, BasicName, NamedValue, Number, NumericData
from pyippdme.protocol.commands import CommandName
from pyippdme.protocol.errors import ErrorCode, ErrorSeverity, ServerError
from pyippdme.protocol.parameters import ParameterName
from pyippdme.protocol.signature import (
    DataType,
    Parameter,
    argument_count_bounds,
    positional_float_parameters,
)
from pyippdme.server._util import (
    bad_argument,
    check_orthogonal,
    incorrect_arguments,
    positional_numbers,
)
from pyippdme.server.backend import MachineBackend
from pyippdme.server.registry import CommandRegistry, HandlerResult
from pyippdme.simulation.classes.tool_class import (
    require_measuring_tool,
    tool_alignment_numbers,
    tool_axis_value,
)
from pyippdme.simulation.context import Ctx
from pyippdme.types.vec3 import Vec3, add, cross, dot, norm, normalize, scale, sub


def _require_backend(ctx: Ctx, cause: str) -> MachineBackend:
    """Narrow ``ctx.backend`` to non-``None``, or fail clearly if this server has none.

    Every scan command here needs a real :class:`~pyippdme.server.backend.MachineBackend`
    to actually produce points from - unlike the rest of this simulation,
    there is no fake fallback path, since "scan geometry" *is* what a
    backend is. A server registering this class without one is
    misconfigured; report that plainly rather than an opaque ``AttributeError``.
    """
    require_measuring_tool(ctx, cause)
    if ctx.backend is None:
        raise ServerError(
            ErrorSeverity.CRITICAL,
            ErrorCode.ERROR_PROCESSING_METHOD,
            cause,
            "No backend configured for scanning",
        )
    return ctx.backend


_DEFAULT_SCAN_REPORT = (
    ParameterName.X,
    ParameterName.Y,
    ParameterName.Z,
    ParameterName.Q,
)

_ON_SCAN_REPORT_PARAMS = (Parameter("Parameters", DataType.ENUM),)
_SCAN_ON_CIRCLE_HINT_PARAMS = positional_float_parameters("Displacement", "Form")
_SCAN_ON_LINE_HINT_PARAMS = positional_float_parameters("Angle", "Form")
_SCAN_ON_CIRCLE_PARAMS = (
    *positional_float_parameters(
        "Cx", "Cy", "Cz", "Sx", "Sy", "Sz", "i", "j", "k", "delta", "sfa", "StepW"
    ),
    Parameter(ParameterName.RT, DataType.BOOL, mandatory=False, positional=True),
)
_SCAN_ON_LINE_PARAMS = (
    *positional_float_parameters("Sx", "Sy", "Sz", "Ex", "Ey", "Ez", "i", "j", "k", "StepW"),
    Parameter(ParameterName.RT, DataType.BOOL, mandatory=False, positional=True),
)
_SCAN_ON_HELIX_PARAMS = (
    *positional_float_parameters(
        "Cx", "Cy", "Cz", "Sx", "Sy", "Sz", "i", "j", "k", "delta", "sfa", "StepW", "pitch"
    ),
    Parameter(ParameterName.RT, DataType.BOOL, mandatory=False, positional=True),
)
#: Table 86: "Surface angle of the circle in degree. 90 and 270 not allowed."
_SCAN_ON_HELIX_FORBIDDEN_SFA = (90.0, 270.0)
_SCAN_UNKNOWN_HINT_PARAMS = positional_float_parameters("MinRadiusOfCurvature")
#: Table 88 marks ``Dis`` mandatory, but General Remarks says "If these
#: values are not set by the client, the server will use default values" -
#: with no other way to supply a mandatory value (there is no positional
#: form), and its sibling ``ScanOnCurveDensity`` already treated the same
#: way, this is read as a table artifact: all three are optional.
_SCAN_UNKNOWN_DENSITY_NAMES = (
    ParameterName.DIS,
    ParameterName.ANGLE,
    ParameterName.ANGLE_BASE_LENGTH,
)
_SCAN_UNKNOWN_DENSITY_PARAMS = tuple(
    Parameter(name, DataType.FLOAT, mandatory=False) for name in _SCAN_UNKNOWN_DENSITY_NAMES
)
_SCAN_ON_CURVE_HINT_PARAMS = positional_float_parameters("Deviation", "MinRadiusOfCurvature")
_SCAN_ON_CURVE_DENSITY_NAMES = (
    ParameterName.DIS,
    ParameterName.ANGLE,
    ParameterName.ANGLE_BASE_LENGTH,
    ParameterName.AT_NOMINALS,
)
_SCAN_ON_CURVE_DENSITY_PARAMS = (
    Parameter(ParameterName.DIS, DataType.FLOAT, mandatory=False),
    Parameter(ParameterName.ANGLE, DataType.FLOAT, mandatory=False),
    Parameter(ParameterName.ANGLE_BASE_LENGTH, DataType.FLOAT, mandatory=False),
    Parameter(ParameterName.AT_NOMINALS, DataType.BOOL, mandatory=False),
)
#: The only supported ``Format`` (Table 85's three mandatory columns:
#: position, orientation, contact tag); see the module docstring.
_SCAN_ON_CURVE_FORMAT_NAMES = (
    ParameterName.X,
    ParameterName.Y,
    ParameterName.Z,
    ParameterName.IJK,
    ParameterName.TAG,
)
#: Numbers per point when ``Format`` is :data:`_SCAN_ON_CURVE_FORMAT_NAMES`:
#: Px, Py, Pz, i, j, k, tag.
_SCAN_ON_CURVE_POINT_SIZE = 7
_SCAN_ON_CURVE_PARAMS = (
    Parameter(ParameterName.CLOSED, DataType.BOOL),
    Parameter(ParameterName.FORMAT, DataType.ENUM),
    Parameter(ParameterName.RT, DataType.BOOL, mandatory=False),
    Parameter(ParameterName.DATA, DataType.ENUM),
)
_SCAN_IN_PLANE_END_IS_SPHERE_PARAMS = positional_float_parameters(
    "Sx",
    "Sy",
    "Sz",
    "Si",
    "Sj",
    "Sk",
    "Ni",
    "Nj",
    "Nk",
    "Dx",
    "Dy",
    "Dz",
    "StepW",
    "Ex",
    "Ey",
    "Ez",
    "Dia",
    "n",
    "Ei",
    "Ej",
    "Ek",
)
_SCAN_IN_PLANE_END_IS_PLANE_PARAMS = positional_float_parameters(
    "Sx",
    "Sy",
    "Sz",
    "Si",
    "Sj",
    "Sk",
    "Ni",
    "Nj",
    "Nk",
    "Dx",
    "Dy",
    "Dz",
    "StepW",
    "Px",
    "Py",
    "Pz",
    "Pi",
    "Pj",
    "Pk",
    "n",
    "Ei",
    "Ej",
    "Ek",
)
_SCAN_IN_PLANE_END_IS_CYL_PARAMS = positional_float_parameters(
    "Sx",
    "Sy",
    "Sz",
    "Si",
    "Sj",
    "Sk",
    "Ni",
    "Nj",
    "Nk",
    "Dx",
    "Dy",
    "Dz",
    "StepW",
    "Cx",
    "Cy",
    "Cz",
    "Ci",
    "Cj",
    "Ck",
    "d",
    "n",
    "Ei",
    "Ej",
    "Ek",
)
_SCAN_IN_CYL_END_IS_SPHERE_PARAMS = positional_float_parameters(
    "Cx",
    "Cy",
    "Cz",
    "Ci",
    "Cj",
    "Ck",
    "Sx",
    "Sy",
    "Sz",
    "Si",
    "Sj",
    "Sk",
    "Dx",
    "Dy",
    "Dz",
    "StepW",
    "Ex",
    "Ey",
    "Ez",
    "Dia",
    "n",
    "Ei",
    "Ej",
    "Ek",
)
_SCAN_IN_CYL_END_IS_PLANE_PARAMS = positional_float_parameters(
    "Cx",
    "Cy",
    "Cz",
    "Ci",
    "Cj",
    "Ck",
    "Sx",
    "Sy",
    "Sz",
    "Si",
    "Sj",
    "Sk",
    "Dx",
    "Dy",
    "Dz",
    "StepW",
    "Px",
    "Py",
    "Pz",
    "Pi",
    "Pj",
    "Pk",
    "n",
    "Ei",
    "Ej",
    "Ek",
)


#: Fields this simulation can actually put a real value behind (the scanned
#: point's own position). ``Q`` (a per-point measurement-quality flag) is
#: also accepted - it is part of :data:`_DEFAULT_SCAN_REPORT` - but always
#: reports 0 ("no defect"), the same "no real contact/quality assessment"
#: simplification the module docstring already states for this simulation;
#: ``ER`` (effective tool radius) is accepted too, always ``0.0`` for the
#: same reason. Anything else this simulation genuinely can't compute (e.g.
#: ``I``/``J``/``K`` surface-normal direction, which would need per-point
#: orientation this simulation's backends don't produce) is rejected up
#: front rather than silently reported as 0.
_AXIS_REPORT_NAMES = (ParameterName.X, ParameterName.Y, ParameterName.Z)
#: ``ER`` (6.10.2 Table 66, "effective tool radius during a measurement")
#: joins ``Q`` as another always-``0.0`` field: no real geometry means no
#: real effective radius either. ``IJK`` is the scan orientation - the
#: command's own ``i,j,k``/per-point ``Si,Sj,Sk`` argument, whichever this
#: scan kind was actually given (see each handler's own call to
#: :func:`_report_values`) - and ``IJKAct`` says it is the nominal vector
#: (Table 32). ``R`` is the rotary table position, and ``Tool.Alignment``
#: and the tool angles report what ``AlignTool()`` set.
_SCAN_REPORT_NAMES = (
    *_AXIS_REPORT_NAMES,
    ParameterName.R,
    ParameterName.Q,
    ParameterName.ER,
    ParameterName.IJK,
    ParameterName.IJK_ACT,
    "Tool.A",
    "Tool.B",
    "Tool.C",
    "FoundTool.A",
    "FoundTool.B",
    "FoundTool.C",
    "Tool.Alignment",
    "FoundTool.Alignment",
)
#: ``IJKAct`` value (Table 32) for "IJK is the nominal vector".
_IJK_ACT_NOMINAL = 2.0


def _report_values(ctx: Ctx, point: Vec3, ijk: Vec3, cause: str) -> tuple[Number, ...]:
    """Build one scanned point's bare values, in the order ``OnScanReport`` asked for."""
    axis: dict[str, int] = {ParameterName.X: 0, ParameterName.Y: 1, ParameterName.Z: 2}
    values: list[Number] = []
    for n in ctx.state.scanning.report or _DEFAULT_SCAN_REPORT:
        if n in axis:
            values.append(Number.of(point[axis[n]]))
        elif n == ParameterName.IJK:
            values.extend(Number.of(c) for c in ijk)
        elif n == ParameterName.IJK_ACT:
            values.append(Number.of(_IJK_ACT_NOMINAL))
        elif n == ParameterName.R:
            values.append(Number.of(ctx.state.rotary_table.position))
        elif n.endswith(".Alignment"):
            numbers = tool_alignment_numbers(ctx, n.partition(".")[0], cause)
            values.extend(Number.of(c) for c in numbers)
        elif n.startswith(("Tool.", "FoundTool.")):
            values.append(Number.of(tool_axis_value(ctx, n, cause)))
        else:
            values.append(Number.of(0.0))  # Q/ER: see _SCAN_REPORT_NAMES
    return tuple(values)


async def _on_scan_report(ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
    names: list[str] = []
    for arg in args:
        if not isinstance(arg, NamedValue) or arg.args:
            raise ServerError(
                ErrorSeverity.CRITICAL,
                ErrorCode.BAD_PROPERTY,
                CommandName.ON_SCAN_REPORT,
                "Bad property",
            )
        if arg.name not in _SCAN_REPORT_NAMES:
            raise ServerError(
                ErrorSeverity.CRITICAL,
                ErrorCode.BAD_PROPERTY,
                CommandName.ON_SCAN_REPORT,
                f"{arg.name} is not allowed in a report",
            )
        names.append(arg.name)
    if not names:
        raise ServerError(
            ErrorSeverity.CRITICAL,
            ErrorCode.BAD_PROPERTY,
            CommandName.ON_SCAN_REPORT,
            "The enumeration may not be empty",
        )
    ctx.state.scanning.report = tuple(names)
    return None


async def _scan_on_circle_hint(_ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
    # (Displacement, Form); accepted and ignored (see module docstring).
    positional_numbers(
        args, CommandName.SCAN_ON_CIRCLE_HINT, *argument_count_bounds(_SCAN_ON_CIRCLE_HINT_PARAMS)
    )
    return None


async def _scan_on_line_hint(_ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
    # (Angle, Form); accepted and ignored (see module docstring).
    positional_numbers(
        args, CommandName.SCAN_ON_LINE_HINT, *argument_count_bounds(_SCAN_ON_LINE_HINT_PARAMS)
    )
    return None


async def _scan_on_curve_hint(_ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
    _deviation, min_radius = positional_numbers(
        args, CommandName.SCAN_ON_CURVE_HINT, *argument_count_bounds(_SCAN_ON_CURVE_HINT_PARAMS)
    )
    if min_radius <= 0:
        raise bad_argument(
            CommandName.SCAN_ON_CURVE_HINT, "MinRadiusOfCurvature must be greater than zero"
        )
    # Accepted and ignored, like the other *Hint commands (see module docstring).
    return None


async def _scan_on_curve_density(_ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
    for arg in args:
        if not isinstance(arg, NamedValue) or arg.name not in _SCAN_ON_CURVE_DENSITY_NAMES:
            raise ServerError(
                ErrorSeverity.CRITICAL,
                ErrorCode.BAD_PROPERTY,
                CommandName.SCAN_ON_CURVE_DENSITY,
                f"Unsupported parameter {arg.name if isinstance(arg, NamedValue) else arg!r}",
            )
    # A hint command ("If these values are not set by the client, the
    # server will use default values", 6.13.2): accepted and ignored.
    return None


@dataclass(frozen=True, slots=True)
class _CurveFormat:
    """Which optional columns a ``ScanOnCurve`` ``Format`` asks for (Table 85)."""

    primary: bool = False
    secondary: bool = False
    rotary: bool = False

    @property
    def point_size(self) -> int:
        return _SCAN_ON_CURVE_POINT_SIZE + 3 * self.primary + 3 * self.secondary + self.rotary


@dataclass(frozen=True, slots=True)
class _CurvePoint:
    position: Vec3
    ijk: Vec3
    rotary: float | None


def _validate_scan_on_curve_format(args: tuple[Argument, ...]) -> _CurveFormat:
    """Check ``Format(X(),Y(),Z(),IJK(),tag[,pi,pj,pk[,si,sj,sk]][,R()])`` (Table 85)."""
    mandatory = _SCAN_ON_CURVE_FORMAT_NAMES
    if len(args) < len(mandatory):
        raise bad_argument(
            CommandName.SCAN_ON_CURVE, "Format must start with X(),Y(),Z(),IJK(),tag"
        )
    for arg, expected in zip(args, mandatory[:-1], strict=False):
        if not isinstance(arg, NamedValue) or arg.args or arg.name != expected:
            raise bad_argument(CommandName.SCAN_ON_CURVE, f"Expected {expected}() in Format")
    tag_item = args[len(mandatory) - 1]
    if not isinstance(tag_item, BasicName) or tag_item.value.lower() != ParameterName.TAG:
        raise bad_argument(CommandName.SCAN_ON_CURVE, "Expected tag after IJK() in Format")
    rest = [
        item.value if isinstance(item, BasicName) else f"{item.name}()"
        for item in args[len(mandatory) :]
        if isinstance(item, BasicName | NamedValue)
    ]
    if len(rest) != len(args) - len(mandatory):
        raise bad_argument(CommandName.SCAN_ON_CURVE, "Unexpected item in Format")
    primary = rest[:3] == ["pi", "pj", "pk"]
    rest = rest[3:] if primary else rest
    secondary = primary and rest[:3] == ["si", "sj", "sk"]
    rest = rest[3:] if secondary else rest
    rotary = rest == ["R()"]
    if rest and not rotary:
        raise bad_argument(
            CommandName.SCAN_ON_CURVE,
            "Format items after tag must be pi,pj,pk then si,sj,sk then R()",
        )
    return _CurveFormat(primary, secondary, rotary)


def _scan_on_curve_points(
    data_args: tuple[Argument, ...], curve_format: _CurveFormat
) -> list[_CurvePoint]:
    """Parse ``Data``'s flat number list into one :class:`_CurvePoint` per nominal point."""
    size = curve_format.point_size
    if not data_args or len(data_args) % size != 0:
        raise bad_argument(
            CommandName.SCAN_ON_CURVE,
            f"Data must hold a multiple of {size} numbers per point",
        )
    if not all(isinstance(a, Number) for a in data_args):
        raise bad_argument(CommandName.SCAN_ON_CURVE, "Data must be a flat list of numbers")
    numbers = tuple(a.value for a in data_args if isinstance(a, Number))
    return [
        _CurvePoint(
            (numbers[i], numbers[i + 1], numbers[i + 2]),
            (numbers[i + 3], numbers[i + 4], numbers[i + 5]),
            numbers[i + size - 1] if curve_format.rotary else None,
        )
        for i in range(0, len(numbers), size)
    ]


async def _scan_on_curve(ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
    by_name: dict[str, NamedValue] = {}
    for arg in args:
        if not isinstance(arg, NamedValue):
            raise bad_argument(CommandName.SCAN_ON_CURVE, "Expected named arguments")
        if arg.name not in ("Closed", "Format", "RT", "Data"):
            raise ServerError(
                ErrorSeverity.CRITICAL,
                ErrorCode.ARGUMENT_NOT_SUPPORTED,
                CommandName.SCAN_ON_CURVE,
                f"Argument {arg.name} not supported",
            )
        by_name[arg.name] = arg

    missing = {"Closed", "Format", "Data"} - by_name.keys()
    if missing:
        raise incorrect_arguments(
            CommandName.SCAN_ON_CURVE, f"Missing mandatory argument(s): {missing}"
        )

    closed_args = by_name["Closed"].args
    if len(closed_args) != 1 or not isinstance(closed_args[0], Number):
        raise bad_argument(CommandName.SCAN_ON_CURVE, "Expected Closed(<bool>)")
    require_measuring_tool(ctx, CommandName.SCAN_ON_CURVE)
    curve_format = _validate_scan_on_curve_format(by_name["Format"].args)
    points = _scan_on_curve_points(by_name["Data"].args, curve_format)
    rotary_table = False
    if "RT" in by_name:
        rt_args = by_name["RT"].args
        if len(rt_args) != 1 or not isinstance(rt_args[0], Number):
            raise bad_argument(CommandName.SCAN_ON_CURVE, "Expected RT(<bool>)")
        rotary_table = rt_args[0].value != 0
    # RT (rotary-table motion during scanning) is accepted but ignored, like
    # every other ScanOn... command's RT (see module docstring); Closed does
    # not change the emitted stream (no real path planning is simulated).

    ctx.state.mover.user_enabled = False

    async def _stream() -> AsyncIterator[NumericData]:
        for nominal in points:
            if ctx.cancel.is_set():
                return
            ctx.state.cart_cmm.position = nominal.position
            if rotary_table and nominal.rotary is not None:
                ctx.state.rotary_table.position = nominal.rotary
            yield NumericData(
                _report_values(ctx, nominal.position, nominal.ijk, CommandName.SCAN_ON_CURVE)
            )

    return _stream()


async def _scan_on_circle(ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
    values = positional_numbers(
        args, CommandName.SCAN_ON_CIRCLE, *argument_count_bounds(_SCAN_ON_CIRCLE_PARAMS)
    )
    cx, cy, cz, sx, sy, sz, i, j, k, delta, _sfa, step_w = values[:12]
    # RT (values[12], if present) is accepted but ignored: rotary-table motion is not simulated.
    center: Vec3 = (cx, cy, cz)
    start: Vec3 = (sx, sy, sz)
    normal: Vec3 = (i, j, k)
    check_orthogonal(
        CommandName.SCAN_ON_CIRCLE,
        normal,
        sub(start, center),
        "i,j,k must be orthogonal to the center-to-start vector",
    )
    if step_w <= 0:
        raise bad_argument(CommandName.SCAN_ON_CIRCLE, "StepW must be positive")
    unit_normal = normalize(normal)
    backend = _require_backend(ctx, CommandName.SCAN_ON_CIRCLE)

    # Mover 6.7.1: all ScanOn... commands implicitly execute DisableUser().
    ctx.state.mover.user_enabled = False

    async def _stream() -> AsyncIterator[NumericData]:
        async for point in backend.scan_circle(center, start, normal, delta, step_w, ctx.cancel):
            ctx.state.cart_cmm.position = point
            yield NumericData(_report_values(ctx, point, unit_normal, CommandName.SCAN_ON_CIRCLE))

    return _stream()


async def _scan_on_helix(ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
    values = positional_numbers(
        args, CommandName.SCAN_ON_HELIX, *argument_count_bounds(_SCAN_ON_HELIX_PARAMS)
    )
    cx, cy, cz, sx, sy, sz, i, j, k, delta, sfa, step_w, pitch = values[:13]
    # RT (values[13], if present) is accepted but ignored: rotary-table motion is not simulated.
    center: Vec3 = (cx, cy, cz)
    start: Vec3 = (sx, sy, sz)
    normal: Vec3 = (i, j, k)
    check_orthogonal(
        CommandName.SCAN_ON_HELIX,
        normal,
        sub(start, center),
        "i,j,k must be orthogonal to the center-to-start vector",
    )
    if step_w <= 0:
        raise bad_argument(CommandName.SCAN_ON_HELIX, "StepW must be positive")
    if sfa in _SCAN_ON_HELIX_FORBIDDEN_SFA:
        raise bad_argument(CommandName.SCAN_ON_HELIX, "sfa may not be 90 or 270 (Table 86)")
    unit_normal = normalize(normal)
    backend = _require_backend(ctx, CommandName.SCAN_ON_HELIX)

    # Mover 6.7.1: all ScanOn... commands implicitly execute DisableUser().
    ctx.state.mover.user_enabled = False

    async def _stream() -> AsyncIterator[NumericData]:
        async for point in backend.scan_helix(
            center, start, normal, delta, step_w, pitch, ctx.cancel
        ):
            ctx.state.cart_cmm.position = point
            yield NumericData(_report_values(ctx, point, unit_normal, CommandName.SCAN_ON_HELIX))

    return _stream()


async def _scan_on_line(ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
    values = positional_numbers(
        args, CommandName.SCAN_ON_LINE, *argument_count_bounds(_SCAN_ON_LINE_PARAMS)
    )
    sx, sy, sz, ex, ey, ez, i, j, k, step_w = values[:10]
    # RT (values[10], if present) is accepted but ignored: rotary-table motion is not simulated.
    start: Vec3 = (sx, sy, sz)
    end: Vec3 = (ex, ey, ez)
    normal: Vec3 = (i, j, k)
    check_orthogonal(
        CommandName.SCAN_ON_LINE,
        normal,
        sub(end, start),
        "i,j,k must be orthogonal to the start-to-end vector",
    )
    if step_w <= 0:
        raise bad_argument(CommandName.SCAN_ON_LINE, "StepW must be positive")
    unit_normal = normalize(normal)
    backend = _require_backend(ctx, CommandName.SCAN_ON_LINE)

    # Mover 6.7.1: all ScanOn... commands implicitly execute DisableUser().
    ctx.state.mover.user_enabled = False

    async def _stream() -> AsyncIterator[NumericData]:
        async for point in backend.scan_line(start, end, normal, step_w, ctx.cancel):
            ctx.state.cart_cmm.position = point
            yield NumericData(_report_values(ctx, point, unit_normal, CommandName.SCAN_ON_LINE))

    return _stream()


async def _scan_unknown_hint(_ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
    (min_radius,) = positional_numbers(
        args, CommandName.SCAN_UNKNOWN_HINT, *argument_count_bounds(_SCAN_UNKNOWN_HINT_PARAMS)
    )
    if min_radius <= 0:
        raise bad_argument(
            CommandName.SCAN_UNKNOWN_HINT, "MinRadiusOfCurvature must be greater than zero"
        )
    # Accepted and ignored, like ScanOnLineHint/ScanOnCircleHint (see module docstring).
    return None


async def _scan_unknown_density(_ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
    seen: set[str] = set()
    for arg in args:
        if not isinstance(arg, NamedValue) or arg.name not in _SCAN_UNKNOWN_DENSITY_NAMES:
            raise ServerError(
                ErrorSeverity.CRITICAL,
                ErrorCode.BAD_PROPERTY,
                CommandName.SCAN_UNKNOWN_DENSITY,
                f"Unsupported parameter {arg.name if isinstance(arg, NamedValue) else arg!r}",
            )
        seen.add(arg.name)
    if ("Angle" in seen) != ("AngleBaseLength" in seen):
        raise incorrect_arguments(
            CommandName.SCAN_UNKNOWN_DENSITY,
            "Angle and AngleBaseLength must be given together or not at all (Table 88)",
        )
    # A hint command ("the server will use default values", 6.13.2.2): accepted and ignored.
    return None


async def _scan_in_plane_end_is_sphere(ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
    bounds = argument_count_bounds(_SCAN_IN_PLANE_END_IS_SPHERE_PARAMS)
    values = positional_numbers(args, CommandName.SCAN_IN_PLANE_END_IS_SPHERE, *bounds)
    (
        sx,
        sy,
        sz,
        _si,
        _sj,
        _sk,
        ni,
        nj,
        nk,
        dx,
        dy,
        dz,
        step_w,
        ex,
        ey,
        ez,
        dia,
        n_raw,
        _ei,
        _ej,
        _ek,
    ) = values
    start: Vec3 = (sx, sy, sz)
    end: Vec3 = (ex, ey, ez)
    normal: Vec3 = (ni, nj, nk)
    direction_point: Vec3 = (dx, dy, dz)
    check_orthogonal(
        CommandName.SCAN_IN_PLANE_END_IS_SPHERE,
        normal,
        sub(end, start),
        "Ni,Nj,Nk must be orthogonal to the start-to-end vector",
    )
    if step_w <= 0:
        raise bad_argument(CommandName.SCAN_IN_PLANE_END_IS_SPHERE, "StepW must be positive")
    if dia <= 0:
        raise bad_argument(CommandName.SCAN_IN_PLANE_END_IS_SPHERE, "Dia must be positive")
    n = int(n_raw)
    if n < 1:
        raise bad_argument(CommandName.SCAN_IN_PLANE_END_IS_SPHERE, "n must be at least 1")
    if norm(sub(direction_point, start)) == 0.0:
        raise bad_argument(
            CommandName.SCAN_IN_PLANE_END_IS_SPHERE,
            "The direction point (Dx,Dy,Dz) may not coincide with the start point",
        )
    unit_normal = normalize(normal)
    backend = _require_backend(ctx, CommandName.SCAN_IN_PLANE_END_IS_SPHERE)

    ctx.state.mover.user_enabled = False

    async def _stream() -> AsyncIterator[NumericData]:
        sphere_entries = 0
        async for point in backend.scan_line(start, end, normal, step_w, ctx.cancel):
            ctx.state.cart_cmm.position = point
            yield NumericData(
                _report_values(ctx, point, unit_normal, CommandName.SCAN_IN_PLANE_END_IS_SPHERE)
            )
            if norm(sub(point, end)) <= dia:
                sphere_entries += 1
                if sphere_entries >= n:
                    return

    return _stream()


def _radial_distance(point: Vec3, axis_point: Vec3, axis_unit: Vec3) -> float:
    """Perpendicular distance from ``point`` to the infinite line through ``axis_point``."""
    return norm(cross(sub(point, axis_point), axis_unit))


def _plane_side(point: Vec3, plane_point: Vec3, plane_normal_unit: Vec3) -> float:
    """Signed distance from ``point`` to the plane through ``plane_point``."""
    return dot(sub(point, plane_point), plane_normal_unit)


async def _scan_in_plane_end_is_plane(ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
    bounds = argument_count_bounds(_SCAN_IN_PLANE_END_IS_PLANE_PARAMS)
    values = positional_numbers(args, CommandName.SCAN_IN_PLANE_END_IS_PLANE, *bounds)
    (
        sx,
        sy,
        sz,
        _si,
        _sj,
        _sk,
        ni,
        nj,
        nk,
        dx,
        dy,
        dz,
        step_w,
        px,
        py,
        pz,
        pi,
        pj,
        pk,
        n_raw,
        _ei,
        _ej,
        _ek,
    ) = values
    start: Vec3 = (sx, sy, sz)
    plane_point: Vec3 = (px, py, pz)
    normal: Vec3 = (ni, nj, nk)
    direction_point: Vec3 = (dx, dy, dz)
    check_orthogonal(
        CommandName.SCAN_IN_PLANE_END_IS_PLANE,
        normal,
        sub(plane_point, start),
        "Ni,Nj,Nk must be orthogonal to the start-to-stop-plane vector",
    )
    if step_w <= 0:
        raise bad_argument(CommandName.SCAN_IN_PLANE_END_IS_PLANE, "StepW must be positive")
    n = int(n_raw)
    if n < 1:
        raise bad_argument(CommandName.SCAN_IN_PLANE_END_IS_PLANE, "n must be at least 1")
    if norm(sub(direction_point, start)) == 0.0:
        raise bad_argument(
            CommandName.SCAN_IN_PLANE_END_IS_PLANE,
            "The direction point (Dx,Dy,Dz) may not coincide with the start point",
        )
    try:
        plane_normal_unit = normalize((pi, pj, pk))
        path_unit = normalize(sub(plane_point, start))
    except ValueError as exc:
        raise bad_argument(
            CommandName.SCAN_IN_PLANE_END_IS_PLANE,
            "Pi,Pj,Pk must be non-zero, and P may not coincide with the start point",
        ) from exc
    unit_normal = normalize(normal)
    # Simplified straight-line scan (see module docstring): aims directly at
    # the stop plane's own point P, then a little past it - unlike the
    # sphere/cylinder stop criteria, P sits exactly on the boundary rather
    # than inside a stop volume, so a scan that stopped right at P could
    # never satisfy n > 1 (nothing beyond the plane to count). The spec's
    # own "start checking only after moving past dist(start, direction
    # point)" gate is not modeled - it exists for genuine adaptive contour
    # tracing, which this simplification does not attempt.
    end = add(plane_point, scale(path_unit, step_w * n))
    start_side = _plane_side(start, plane_point, plane_normal_unit)
    backend = _require_backend(ctx, CommandName.SCAN_IN_PLANE_END_IS_PLANE)

    ctx.state.mover.user_enabled = False

    async def _stream() -> AsyncIterator[NumericData]:
        crossings = 0
        async for point in backend.scan_line(start, end, normal, step_w, ctx.cancel):
            ctx.state.cart_cmm.position = point
            yield NumericData(
                _report_values(ctx, point, unit_normal, CommandName.SCAN_IN_PLANE_END_IS_PLANE)
            )
            side = _plane_side(point, plane_point, plane_normal_unit)
            if side * start_side <= 0:
                crossings += 1
                if crossings >= n:
                    return

    return _stream()


async def _scan_in_plane_end_is_cyl(ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
    bounds = argument_count_bounds(_SCAN_IN_PLANE_END_IS_CYL_PARAMS)
    values = positional_numbers(args, CommandName.SCAN_IN_PLANE_END_IS_CYL, *bounds)
    (
        sx,
        sy,
        sz,
        _si,
        _sj,
        _sk,
        ni,
        nj,
        nk,
        dx,
        dy,
        dz,
        step_w,
        cx,
        cy,
        cz,
        ci,
        cj,
        ck,
        d,
        n_raw,
        _ei,
        _ej,
        _ek,
    ) = values
    start: Vec3 = (sx, sy, sz)
    axis_point: Vec3 = (cx, cy, cz)
    normal: Vec3 = (ni, nj, nk)
    direction_point: Vec3 = (dx, dy, dz)
    check_orthogonal(
        CommandName.SCAN_IN_PLANE_END_IS_CYL,
        normal,
        sub(axis_point, start),
        "Ni,Nj,Nk must be orthogonal to the start-to-stop-cylinder vector",
    )
    if step_w <= 0:
        raise bad_argument(CommandName.SCAN_IN_PLANE_END_IS_CYL, "StepW must be positive")
    if d <= 0:
        raise bad_argument(CommandName.SCAN_IN_PLANE_END_IS_CYL, "d must be positive")
    n = int(n_raw)
    if n < 1:
        raise bad_argument(CommandName.SCAN_IN_PLANE_END_IS_CYL, "n must be at least 1")
    if norm(sub(direction_point, start)) == 0.0:
        raise bad_argument(
            CommandName.SCAN_IN_PLANE_END_IS_CYL,
            "The direction point (Dx,Dy,Dz) may not coincide with the start point",
        )
    try:
        axis_unit = normalize((ci, cj, ck))
    except ValueError as exc:
        raise bad_argument(
            CommandName.SCAN_IN_PLANE_END_IS_CYL, "Ci,Cj,Ck must be non-zero"
        ) from exc
    unit_normal = normalize(normal)
    end = axis_point  # simplified straight-line scan (see module docstring)
    backend = _require_backend(ctx, CommandName.SCAN_IN_PLANE_END_IS_CYL)

    ctx.state.mover.user_enabled = False

    async def _stream() -> AsyncIterator[NumericData]:
        entries = 0
        async for point in backend.scan_line(start, end, normal, step_w, ctx.cancel):
            ctx.state.cart_cmm.position = point
            yield NumericData(
                _report_values(ctx, point, unit_normal, CommandName.SCAN_IN_PLANE_END_IS_CYL)
            )
            if _radial_distance(point, axis_point, axis_unit) <= d:
                entries += 1
                if entries >= n:
                    return

    return _stream()


async def _scan_in_cyl_end_is_sphere(ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
    bounds = argument_count_bounds(_SCAN_IN_CYL_END_IS_SPHERE_PARAMS)
    values = positional_numbers(args, CommandName.SCAN_IN_CYL_END_IS_SPHERE, *bounds)
    (
        cx,
        cy,
        cz,
        ci,
        cj,
        ck,
        sx,
        sy,
        sz,
        _si,
        _sj,
        _sk,
        dx,
        dy,
        dz,
        step_w,
        ex,
        ey,
        ez,
        dia,
        n_raw,
        _ei,
        _ej,
        _ek,
    ) = values
    start: Vec3 = (sx, sy, sz)
    end: Vec3 = (ex, ey, ez)
    axis_point: Vec3 = (cx, cy, cz)
    direction_point: Vec3 = (dx, dy, dz)
    if step_w <= 0:
        raise bad_argument(CommandName.SCAN_IN_CYL_END_IS_SPHERE, "StepW must be positive")
    if dia <= 0:
        raise bad_argument(CommandName.SCAN_IN_CYL_END_IS_SPHERE, "Dia must be positive")
    n = int(n_raw)
    if n < 1:
        raise bad_argument(CommandName.SCAN_IN_CYL_END_IS_SPHERE, "n must be at least 1")
    if norm(sub(direction_point, start)) == 0.0:
        raise bad_argument(
            CommandName.SCAN_IN_CYL_END_IS_SPHERE,
            "The direction point (Dx,Dy,Dz) may not coincide with the start point",
        )
    try:
        axis_unit = normalize((ci, cj, ck))
    except ValueError as exc:
        raise bad_argument(
            CommandName.SCAN_IN_CYL_END_IS_SPHERE, "Ci,Cj,Ck must be non-zero"
        ) from exc
    if _radial_distance(start, axis_point, axis_unit) == 0.0:
        raise bad_argument(
            CommandName.SCAN_IN_CYL_END_IS_SPHERE,
            "The start point may not lie on the cylinder axis",
        )
    backend = _require_backend(ctx, CommandName.SCAN_IN_CYL_END_IS_SPHERE)

    ctx.state.mover.user_enabled = False

    async def _stream() -> AsyncIterator[NumericData]:
        sphere_entries = 0
        # No scanning-plane normal exists for this surface kind (see module
        # docstring) - the cylinder axis is the closest available
        # orientation reference, reused for IJK too.
        async for point in backend.scan_line(start, end, axis_unit, step_w, ctx.cancel):
            ctx.state.cart_cmm.position = point
            yield NumericData(
                _report_values(ctx, point, axis_unit, CommandName.SCAN_IN_CYL_END_IS_SPHERE)
            )
            if norm(sub(point, end)) <= dia:
                sphere_entries += 1
                if sphere_entries >= n:
                    return

    return _stream()


async def _scan_in_cyl_end_is_plane(ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
    bounds = argument_count_bounds(_SCAN_IN_CYL_END_IS_PLANE_PARAMS)
    values = positional_numbers(args, CommandName.SCAN_IN_CYL_END_IS_PLANE, *bounds)
    (
        cx,
        cy,
        cz,
        ci,
        cj,
        ck,
        sx,
        sy,
        sz,
        _si,
        _sj,
        _sk,
        dx,
        dy,
        dz,
        step_w,
        px,
        py,
        pz,
        pi,
        pj,
        pk,
        n_raw,
        _ei,
        _ej,
        _ek,
    ) = values
    start: Vec3 = (sx, sy, sz)
    plane_point: Vec3 = (px, py, pz)
    axis_point: Vec3 = (cx, cy, cz)
    direction_point: Vec3 = (dx, dy, dz)
    if step_w <= 0:
        raise bad_argument(CommandName.SCAN_IN_CYL_END_IS_PLANE, "StepW must be positive")
    n = int(n_raw)
    if n < 1:
        raise bad_argument(CommandName.SCAN_IN_CYL_END_IS_PLANE, "n must be at least 1")
    if norm(sub(direction_point, start)) == 0.0:
        raise bad_argument(
            CommandName.SCAN_IN_CYL_END_IS_PLANE,
            "The direction point (Dx,Dy,Dz) may not coincide with the start point",
        )
    try:
        axis_unit = normalize((ci, cj, ck))
        plane_normal_unit = normalize((pi, pj, pk))
        path_unit = normalize(sub(plane_point, start))
    except ValueError as exc:
        raise bad_argument(
            CommandName.SCAN_IN_CYL_END_IS_PLANE,
            "Ci,Cj,Ck and Pi,Pj,Pk must be non-zero, and P may not coincide with the start point",
        ) from exc
    if _radial_distance(start, axis_point, axis_unit) == 0.0:
        raise bad_argument(
            CommandName.SCAN_IN_CYL_END_IS_PLANE,
            "The start point may not lie on the cylinder axis",
        )
    end = add(plane_point, scale(path_unit, step_w * n))
    start_side = _plane_side(start, plane_point, plane_normal_unit)
    backend = _require_backend(ctx, CommandName.SCAN_IN_CYL_END_IS_PLANE)

    ctx.state.mover.user_enabled = False

    async def _stream() -> AsyncIterator[NumericData]:
        crossings = 0
        async for point in backend.scan_line(start, end, axis_unit, step_w, ctx.cancel):
            ctx.state.cart_cmm.position = point
            yield NumericData(
                _report_values(ctx, point, axis_unit, CommandName.SCAN_IN_CYL_END_IS_PLANE)
            )
            side = _plane_side(point, plane_point, plane_normal_unit)
            if side * start_side <= 0:
                crossings += 1
                if crossings >= n:
                    return

    return _stream()


def _reset_report_on_start_session(ctx: Ctx) -> None:
    # 6.3.1 StartSession(): "those of OnScanReport are set to (X(),Y(),Z(),Q())".
    ctx.state.scanning.report = ()


def register(registry: CommandRegistry) -> None:
    registry.register_session_start_hook(_reset_report_on_start_session)
    registry.register(CommandName.ON_SCAN_REPORT, _on_scan_report, arguments=_ON_SCAN_REPORT_PARAMS)
    registry.register(
        CommandName.SCAN_ON_CIRCLE_HINT, _scan_on_circle_hint, arguments=_SCAN_ON_CIRCLE_HINT_PARAMS
    )
    registry.register(
        CommandName.SCAN_ON_LINE_HINT, _scan_on_line_hint, arguments=_SCAN_ON_LINE_HINT_PARAMS
    )
    registry.register(
        CommandName.SCAN_ON_CURVE_HINT, _scan_on_curve_hint, arguments=_SCAN_ON_CURVE_HINT_PARAMS
    )
    registry.register(
        CommandName.SCAN_ON_CURVE_DENSITY,
        _scan_on_curve_density,
        arguments=_SCAN_ON_CURVE_DENSITY_PARAMS,
    )
    registry.register(CommandName.SCAN_ON_CURVE, _scan_on_curve, arguments=_SCAN_ON_CURVE_PARAMS)
    registry.register(CommandName.SCAN_ON_CIRCLE, _scan_on_circle, arguments=_SCAN_ON_CIRCLE_PARAMS)
    registry.register(CommandName.SCAN_ON_LINE, _scan_on_line, arguments=_SCAN_ON_LINE_PARAMS)
    registry.register(CommandName.SCAN_ON_HELIX, _scan_on_helix, arguments=_SCAN_ON_HELIX_PARAMS)
    registry.register(
        CommandName.SCAN_UNKNOWN_HINT, _scan_unknown_hint, arguments=_SCAN_UNKNOWN_HINT_PARAMS
    )
    registry.register(
        CommandName.SCAN_UNKNOWN_DENSITY,
        _scan_unknown_density,
        arguments=_SCAN_UNKNOWN_DENSITY_PARAMS,
    )
    registry.register(
        CommandName.SCAN_IN_PLANE_END_IS_SPHERE,
        _scan_in_plane_end_is_sphere,
        arguments=_SCAN_IN_PLANE_END_IS_SPHERE_PARAMS,
    )
    registry.register(
        CommandName.SCAN_IN_PLANE_END_IS_PLANE,
        _scan_in_plane_end_is_plane,
        arguments=_SCAN_IN_PLANE_END_IS_PLANE_PARAMS,
    )
    registry.register(
        CommandName.SCAN_IN_PLANE_END_IS_CYL,
        _scan_in_plane_end_is_cyl,
        arguments=_SCAN_IN_PLANE_END_IS_CYL_PARAMS,
    )
    registry.register(
        CommandName.SCAN_IN_CYL_END_IS_SPHERE,
        _scan_in_cyl_end_is_sphere,
        arguments=_SCAN_IN_CYL_END_IS_SPHERE_PARAMS,
    )
    registry.register(
        CommandName.SCAN_IN_CYL_END_IS_PLANE,
        _scan_in_cyl_end_is_plane,
        arguments=_SCAN_IN_CYL_END_IS_PLANE_PARAMS,
    )
