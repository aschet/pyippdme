# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Typed helpers that build the argument tuple for ``NamedValue``-shaped commands.

:class:`~pyippdme.protocol.commands.CommandName` and
:class:`~pyippdme.protocol.parameters.ParameterName` remove the risk of
*mistyping* a command or argument name; they don't remove the remaining
risk of getting a command's argument *shape* wrong - how many
arguments, in what order, wrapped how (a bare ``NamedValue``, a nested one,
a fixed-order number list, ...). A caller using the low-level
:class:`~pyippdme.client.IppDmeClient` API still has to know all of that by
hand, e.g. ``ScanOnCurve``'s three-argument, fixed-``Format``, flattened-
``Data`` shape (see :mod:`pyippdme.simulation.classes.scanning_class`'s module
docstring).

Each function here builds exactly the ``tuple[Argument, ...]`` its command
needs, so it plugs directly into :meth:`~pyippdme.client.IppDmeClient.call`/
:meth:`~pyippdme.client.IppDmeClient.send` without changing that API at all::

    await client.call(CommandName.GO_TO, *go_to(x=10, y=20))

:class:`~pyippdme.client.model.IppDmeMachine` (:mod:`pyippdme.client.model`) uses these
same functions internally for every command they cover, instead of
duplicating the argument-construction logic - so there is exactly one
place that knows each command's wire shape, not two.

These functions only *build* arguments; they never validate values (e.g.
``go_to()`` with every axis ``None`` is accepted here and left for the
server's own error response) - matching this project's existing split
between construction and validation (validation happens server-side, see
:mod:`pyippdme.server.classes`).

:func:`bare_names`/:func:`named_numbers` are the generic building blocks
the per-command functions below are written in terms of; they are public
so a caller building a proprietary/vendor-extension command that isn't
covered by a dedicated function here can still reach for the same
recurring shapes (a bare name per argument, a plain named number)
instead of constructing :class:`~pyippdme.protocol.ast.NamedValue`
directly. Mirrors :mod:`pyippdme.server.builders`'s own generic helpers
on the response side.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from pyippdme.protocol.ast import Argument, BasicName, NamedValue, Number, String
from pyippdme.protocol.parameters import ParameterName
from pyippdme.types.csy import CoordinateTransform
from pyippdme.types.vec3 import Vec3


def bare_names(*names: str) -> tuple[Argument, ...]:
    """Build one empty-argument ``NamedValue`` per name, e.g. ``Get(X(), Y())``."""
    return tuple(NamedValue(n, ()) for n in names)


def named_numbers(*named: tuple[str, float | None]) -> tuple[Argument, ...]:
    """Build one ``NamedValue(name, (Number,))`` per non-``None`` pair, in the order given."""
    return tuple(NamedValue(n, (Number.of(v),)) for n, v in named if v is not None)


@dataclass(frozen=True, slots=True)
class ToolAlignment:
    """A tool orientation: the primary direction and, optionally, the secondary one (6.20)."""

    primary: Vec3
    secondary: Vec3 | None = None


@dataclass(frozen=True, slots=True)
class PartAlignment:
    """The rotary-table orientation ``AlignPart`` asks for (6.23.1).

    ``second_part_vector``, ``second_machine_vector`` and ``beta`` are for a
    second rotary table, orthogonal to the first.
    """

    part_vector: Vec3
    machine_vector: Vec3
    alpha: float
    second_part_vector: Vec3 | None = None
    second_machine_vector: Vec3 | None = None
    beta: float | None = None

    def __post_init__(self) -> None:
        second = (self.second_part_vector, self.second_machine_vector, self.beta)
        if any(item is None for item in second) and any(item is not None for item in second):
            raise ValueError(
                "second_part_vector, second_machine_vector and beta must be given together"
            )


def _motion(
    x: float | None,
    y: float | None,
    z: float | None,
    ijk: Vec3 | None,
    r: float | None,
    a: float | None,
    b: float | None,
    c: float | None,
    alignment: ToolAlignment | None,
    align_part: PartAlignment | None,
    sync: bool | None,
) -> tuple[Argument, ...]:
    arguments = list(
        named_numbers(
            (ParameterName.X, x),
            (ParameterName.Y, y),
            (ParameterName.Z, z),
        )
    )
    if ijk is not None:
        arguments.append(vector(ParameterName.IJK, ijk))
    arguments.extend(
        named_numbers(("R", r), ("Tool.A", a), ("Tool.B", b), ("Tool.C", c)),
    )
    if alignment is not None:
        arguments.append(
            NamedValue("Tool.Alignment", _flat(alignment.primary, alignment.secondary))
        )
    if align_part is not None:
        arguments.append(NamedValue("AlignPart", _align_part_numbers(align_part)))
    if sync is not None:
        arguments.append(NamedValue("Sync", (Number.of(1 if sync else 0),)))
    return tuple(arguments)


def go_to(
    x: float | None = None,
    y: float | None = None,
    z: float | None = None,
    *,
    r: float | None = None,
    a: float | None = None,
    b: float | None = None,
    c: float | None = None,
    alignment: ToolAlignment | None = None,
    sync: bool | None = None,
) -> tuple[Argument, ...]:
    """Build ``GoTo``'s argument list, e.g. ``go_to(x=10, y=20)`` -> ``(X(10), Y(20))``."""
    return _motion(x, y, z, None, r, a, b, c, alignment, None, sync)


def step(
    x: float | None = None,
    y: float | None = None,
    z: float | None = None,
    *,
    r: float | None = None,
    a: float | None = None,
    b: float | None = None,
    c: float | None = None,
    sync: bool | None = None,
) -> tuple[Argument, ...]:
    """Build ``Step``'s argument list (6.8.1's relative move)."""
    return _motion(x, y, z, None, r, a, b, c, None, None, sync)


def pt_meas(
    x: float | None = None,
    y: float | None = None,
    z: float | None = None,
    *,
    ijk: Vec3 | None = None,
    r: float | None = None,
    a: float | None = None,
    b: float | None = None,
    c: float | None = None,
    alignment: ToolAlignment | None = None,
    align_part: PartAlignment | None = None,
) -> tuple[Argument, ...]:
    """Build ``PtMeas``'s argument list."""
    return _motion(x, y, z, ijk, r, a, b, c, alignment, align_part, None)


def get(*axes: str) -> tuple[Argument, ...]:
    """Build ``Get``'s argument list, e.g. ``get("X", "Y", "R")``."""
    return bare_names(*axes)


def on_pt_meas_report(*axes: str) -> tuple[Argument, ...]:
    """Build ``OnPtMeasReport``'s argument list."""
    return bare_names(*axes)


def get_scale_temperatures(*axes: str) -> tuple[Argument, ...]:
    """Build ``GetScaleTemperatures``' argument list."""
    return bare_names(*axes)


def lock_axis(*axes: str) -> tuple[Argument, ...]:
    """Build ``LockAxis``'s argument list."""
    return bare_names(*axes)


def lock_position(*positions: str) -> tuple[Argument, ...]:
    """Build ``LockPosition``'s argument list."""
    return bare_names(*positions)


def enum_prop(reference: str) -> tuple[Argument, ...]:
    """Build ``EnumProp``'s argument list."""
    return bare_names(reference)


def enum_all_prop(reference: str) -> tuple[Argument, ...]:
    """Build ``EnumAllProp``'s argument list."""
    return bare_names(reference)


def get_prop(*names: str) -> tuple[Argument, ...]:
    """Build ``GetProp``'s argument list, e.g. ``get_prop("Tool.GoToPar.Speed")``."""
    return bare_names(*names)


def set_prop(name: str, value: float | str | Sequence[float]) -> tuple[Argument, ...]:
    """Build ``SetProp``'s argument for one property: a number, a string or several numbers."""
    if isinstance(value, str):
        return (NamedValue(name, (String(value),)),)
    if isinstance(value, int | float):
        return (NamedValue(name, (Number.of(value),)),)
    return (NamedValue(name, tuple(Number.of(v) for v in value)),)


def set_scale_temperatures(**temperatures: float) -> tuple[Argument, ...]:
    """Build ``SetScaleTemperatures``' argument list, e.g. ``set_scale_temperatures(X=21.0)``."""
    return tuple(NamedValue(axis, (Number.of(v),)) for axis, v in temperatures.items())


def _center_ijk_target(center: Vec3, ijk: Vec3, target: Vec3) -> tuple[Argument, ...]:
    return (
        NamedValue(ParameterName.CENTER, tuple(Number.of(v) for v in center)),
        NamedValue(ParameterName.IJK, tuple(Number.of(v) for v in ijk)),
        NamedValue(ParameterName.X, (Number.of(target[0]),)),
        NamedValue(ParameterName.Y, (Number.of(target[1]),)),
        NamedValue(ParameterName.Z, (Number.of(target[2]),)),
    )


def go_to_on_circle(center: Vec3, ijk: Vec3, target: Vec3) -> tuple[Argument, ...]:
    """Build ``GoToOnCircle``'s argument list."""
    return _center_ijk_target(center, ijk, target)


def go_to_on_spiral(center: Vec3, ijk: Vec3, target: Vec3) -> tuple[Argument, ...]:
    """Build ``GoToOnSpiral``'s argument list (the same shape as ``GoToOnCircle``)."""
    return _center_ijk_target(center, ijk, target)


def on_scan_report(*fields: str) -> tuple[Argument, ...]:
    """Build ``OnScanReport``'s argument list, e.g. ``on_scan_report("X", "Y", "Z")``."""
    return bare_names(*fields)


def set_csy_transformation(transform: CoordinateTransform) -> tuple[Argument, ...]:
    """Build the ``X0, Y0, Z0, Theta, Psi, Phi`` group shared by the csy-transform commands.

    This is a fixed-order *positional* number list, not ``NamedValue``-
    wrapped (unlike every other builder here) - the standard sends it that way
    on the request side (only the ``GetCsyTransformation`` response wraps
    the same six values as ``NamedValue``s). The caller still passes the
    leading csy name (``BasicName``/``String``) itself, since it isn't part
    of this group.
    """
    return tuple(
        Number.of(v)
        for v in (
            transform.x0,
            transform.y0,
            transform.z0,
            transform.theta,
            transform.psi,
            transform.phi,
        )
    )


@dataclass(frozen=True, slots=True)
class CurvePoint:
    """One nominal point of a ``ScanOnCurve`` (Table 85).

    ``tag`` is ``+1`` (on the part surface) or ``-1`` (no contact expected).
    ``primary`` and ``secondary`` are the nominal tool directions at the point
    (see ``AlignTool``; ``secondary`` needs ``primary``) and ``rotary`` is the
    rotary-table angle. Every point of one scan must give the same optional
    columns.
    """

    position: Vec3
    normal: Vec3
    tag: int
    primary: Vec3 | None = None
    secondary: Vec3 | None = None
    rotary: float | None = None

    def __post_init__(self) -> None:
        if self.secondary is not None and self.primary is None:
            raise ValueError("secondary needs primary")


def scan_on_curve(
    points: Sequence[CurvePoint],
    *,
    closed: bool = False,
    rotary_table: bool | None = None,
) -> tuple[Argument, ...]:
    """Build ``ScanOnCurve``'s argument list from a sequence of nominal points."""
    columns = {
        (p.primary is not None, p.secondary is not None, p.rotary is not None) for p in points
    }
    if len(columns) > 1:
        raise ValueError("Every curve point must give the same optional columns")
    with_primary, with_secondary, with_rotary = columns.pop() if columns else (False,) * 3
    format_items: list[Argument] = [
        NamedValue(ParameterName.X, ()),
        NamedValue(ParameterName.Y, ()),
        NamedValue(ParameterName.Z, ()),
        NamedValue(ParameterName.IJK, ()),
        BasicName(ParameterName.TAG),
    ]
    if with_primary:
        format_items.extend(BasicName(name) for name in ("pi", "pj", "pk"))
    if with_secondary:
        format_items.extend(BasicName(name) for name in ("si", "sj", "sk"))
    if with_rotary:
        format_items.append(NamedValue("R", ()))
    data_numbers = [
        v
        for p in points
        for v in (
            *p.position,
            *p.normal,
            float(p.tag),
            *(p.primary or ()),
            *(p.secondary or ()),
            *(() if p.rotary is None else (p.rotary,)),
        )
    ]
    arguments: list[Argument] = [
        NamedValue(ParameterName.CLOSED, (Number.of(1 if closed else 0),)),
        NamedValue(ParameterName.FORMAT, tuple(format_items)),
    ]
    if rotary_table is not None:
        arguments.append(NamedValue("RT", (Number.of(1 if rotary_table else 0),)))
    arguments.append(NamedValue(ParameterName.DATA, tuple(Number.of(v) for v in data_numbers)))
    return tuple(arguments)


def _positional(*values: float | bool | None) -> tuple[Argument, ...]:
    """Build bare numbers in order; ``None`` entries (trailing optional arguments) are left out."""
    return tuple(
        Number.of(1 if v is True else 0 if v is False else v) for v in values if v is not None
    )


def _flat(*parts: float | bool | Vec3 | None) -> tuple[Argument, ...]:
    flat: list[float | bool | None] = []
    for part in parts:
        if isinstance(part, tuple):
            flat.extend(part)
        else:
            flat.append(part)
    return _positional(*flat)


def vector(name: str, value: Vec3) -> NamedValue:
    """Build a three-number named argument such as ``IJK(0,0,1)``."""
    return NamedValue(name, tuple(Number.of(v) for v in value))


def pt_meas_self_center(
    x: float | None = None,
    y: float | None = None,
    z: float | None = None,
    ijk: Vec3 | None = None,
    lmn: Vec3 | None = None,
) -> tuple[Argument, ...]:
    """Build ``PtMeasSelfCenter``'s (or, with ``lmn``, ``PtMeasSelfCenterLocked``'s) arguments."""
    arguments = list(
        named_numbers((ParameterName.X, x), (ParameterName.Y, y), (ParameterName.Z, z))
    )
    if ijk is not None:
        arguments.append(vector("IJK", ijk))
    if lmn is not None:
        arguments.append(vector("LMN", lmn))
    return tuple(arguments)


def on_move_report(time: float, dis: float, axes: Sequence[str]) -> tuple[Argument, ...]:
    """Build ``OnMoveReport``'s argument list, e.g. ``Time(0.5), Dis(1), X(), Y()``."""
    return (*named_numbers(("Time", time), ("Dis", dis)), *bare_names(*axes))


def align_tool(
    primary: Vec3, alpha: float, secondary: Vec3 | None = None, beta: float | None = None
) -> tuple[Argument, ...]:
    """Build ``AlignTool``'s arguments: ``i1,j1,k1,alpha`` or all of ``i1..k2,alpha,beta``."""
    if (secondary is None) != (beta is None):
        raise ValueError("secondary and beta must be given together")
    return _flat(primary, secondary, alpha, beta)


def alignment_axis(namespace: str, axis: str) -> tuple[Argument, ...]:
    """Build ``CalcToolAlignment``'s argument, e.g. ``Tool.A()``."""
    return (NamedValue(f"{namespace}.{axis}", ()),)


def alignment_vectors(
    namespace: str, primary: Vec3, secondary: Vec3 | None
) -> tuple[Argument, ...]:
    """Build ``CalcToolAngles``' argument, e.g. ``Tool.Alignment(i1,j1,k1[,i2,j2,k2])``."""
    return (NamedValue(f"{namespace}.Alignment", _flat(primary, secondary)),)


def raw_data_bin_setup(data_format: str, port: int, live_mode: bool) -> tuple[Argument, ...]:
    """Build ``RawDataBinSetup``'s arguments."""
    return (BasicName(data_format), Number.of(port), BasicName("On" if live_mode else "Off"))


def scan_on_circle(
    center: Vec3,
    start: Vec3,
    normal: Vec3,
    delta: float,
    surface_angle: float,
    step_width: float,
    rotary_table: bool | None = None,
    *,
    pitch: float | None = None,
) -> tuple[Argument, ...]:
    """Build ``ScanOnCircle``'s (or, with ``pitch``, ``ScanOnHelix``'s) positional arguments."""
    return _flat(center, start, normal, delta, surface_angle, step_width, pitch, rotary_table)


def density(
    dis: float | None = None,
    angle: float | None = None,
    angle_base_length: float | None = None,
    at_nominals: bool | None = None,
) -> tuple[Argument, ...]:
    """Build ``ScanOnCurveDensity``'s/``ScanUnknownDensity``'s optional named arguments."""
    arguments = list(
        named_numbers(("Dis", dis), ("Angle", angle), ("AngleBaseLength", angle_base_length))
    )
    if at_nominals is not None:
        arguments.append(NamedValue("AtNominals", (Number.of(1 if at_nominals else 0),)))
    return tuple(arguments)


def positional(*parts: float | bool | Vec3 | None) -> tuple[Argument, ...]:
    """Build bare positional numbers, flattening vectors, for the fixed-order scan commands."""
    return _flat(*parts)


def _align_part_numbers(alignment: PartAlignment) -> tuple[Argument, ...]:
    return _flat(
        alignment.part_vector,
        alignment.machine_vector,
        alignment.second_part_vector,
        alignment.second_machine_vector,
        alignment.alpha,
        alignment.beta,
    )


def align_part(alignment: PartAlignment) -> tuple[Argument, ...]:
    """Build ``AlignPart``'s positional arguments."""
    return _align_part_numbers(alignment)


@dataclass(frozen=True, slots=True)
class AcquisitionPoint:
    """One position of a ``DataAcquire`` path (Table 95).

    ``primary`` is the vector anti-parallel to the tool's main axis and
    ``secondary`` the one describing the orientation within the working plane
    (see ``AlignTool``).
    """

    position: Vec3
    primary: Vec3
    secondary: Vec3


def data_acquire(
    acq_name: str,
    acquisition_type: str,
    settings_name: str,
    points: Sequence[AcquisitionPoint] = (),
    *,
    step_width: float | None = None,
    rotary_table: bool | None = None,
) -> tuple[Argument, ...]:
    """Build ``DataAcquire``'s positional arguments."""
    numbers = [v for p in points for v in (*p.position, *p.primary, *p.secondary)]
    tail = _positional(step_width, rotary_table)
    if rotary_table is not None and step_width is None:
        raise ValueError("rotary_table needs step_width")
    return (
        String(acq_name),
        BasicName(acquisition_type),
        String(settings_name),
        Number.of(len(points)),
        *(Number.of(v) for v in numbers),
        *tail,
    )
