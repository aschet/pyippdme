# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for the Scanning class subset (geometry, validation, OnScanReport)."""

from __future__ import annotations

import math

import pytest

from pyippdme import IppDmeClient
from pyippdme.exceptions import IppDmeServerError
from pyippdme.protocol.ast import BasicName, Items, NamedValue, Number, NumericData
from pyippdme.protocol.commands import CommandName
from pyippdme.protocol.parameters import ParameterName
from pyippdme.protocol.parser import parse_method


def _numeric(data: object) -> NumericData:
    assert isinstance(data, NumericData)
    return data


async def test_scan_on_line_point_count_and_endpoints(started_client: IppDmeClient) -> None:
    args = [Number.of(v) for v in (0, 0, 0, 10, 0, 0, 0, 0, 1, 2.5)]  # StepW=2.5 over length 10
    data = await started_client.call(CommandName.SCAN_ON_LINE, *args)
    assert len(data) == 5  # 10 / 2.5 + 1
    first = _numeric(data[0]).values
    last = _numeric(data[-1]).values
    assert [n.value for n in first[:3]] == pytest.approx([0.0, 0.0, 0.0])
    assert [n.value for n in last[:3]] == pytest.approx([10.0, 0.0, 0.0])


async def test_scan_on_line_default_report_is_x_y_z_q(started_client: IppDmeClient) -> None:
    args = [Number.of(v) for v in (0, 0, 0, 10, 0, 0, 0, 0, 1, 10)]
    data = await started_client.call(CommandName.SCAN_ON_LINE, *args)
    assert len(_numeric(data[0]).values) == 4  # X, Y, Z, Q


async def test_on_scan_report_reconfigures_shape(started_client: IppDmeClient) -> None:
    await started_client.call(
        CommandName.ON_SCAN_REPORT, NamedValue(ParameterName.X, ()), NamedValue(ParameterName.Y, ())
    )
    args = [Number.of(v) for v in (0, 0, 0, 10, 0, 0, 0, 0, 1, 10)]
    data = await started_client.call(CommandName.SCAN_ON_LINE, *args)
    values = _numeric(data[0]).values
    assert len(values) == 2
    assert [v.value for v in values] == pytest.approx([0.0, 0.0])


async def test_on_scan_report_empty_enumeration_errors(started_client: IppDmeClient) -> None:
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(CommandName.ON_SCAN_REPORT)
    assert excinfo.value.error.number == "0510"


async def test_on_scan_report_accepts_q_as_the_default_shape_includes_it(
    started_client: IppDmeClient,
) -> None:
    await started_client.call(CommandName.ON_SCAN_REPORT, NamedValue(ParameterName.Q, ()))
    args = [Number.of(v) for v in (0, 0, 0, 10, 0, 0, 0, 0, 1, 10)]
    data = await started_client.call(CommandName.SCAN_ON_LINE, *args)
    values = _numeric(data[0]).values
    assert [v.value for v in values] == pytest.approx([0.0])


async def test_on_scan_report_rejects_a_field_it_cannot_compute(
    started_client: IppDmeClient,
) -> None:
    """I/J/K (surface-normal direction) look like real fields but aren't in Table 78.

    Silently reporting 0 for them would be indistinguishable from a real
    (but coincidentally zero) measurement - see the module docstring.
    """
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(CommandName.ON_SCAN_REPORT, NamedValue("I", ()))
    assert excinfo.value.error.number == "0510"


async def test_on_scan_report_rejects_an_unknown_name(started_client: IppDmeClient) -> None:
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(CommandName.ON_SCAN_REPORT, NamedValue("Bogus", ()))
    assert excinfo.value.error.number == "0510"


async def test_scan_on_circle_quarter_arc(started_client: IppDmeClient) -> None:
    # Center (0,0,0), start (5,0,0), normal +Z, delta=90deg, StepW=30deg -> 4 points at 0/30/60/90.
    args = [Number.of(v) for v in (0, 0, 0, 5, 0, 0, 0, 0, 1, 90, 0, 30)]
    data = await started_client.call(CommandName.SCAN_ON_CIRCLE, *args)
    assert len(data) == 4
    last = [v.value for v in _numeric(data[-1]).values[:3]]
    assert last == pytest.approx([0.0, 5.0, 0.0], abs=1e-9)


async def test_scan_on_circle_negative_delta_is_clockwise(started_client: IppDmeClient) -> None:
    args = [Number.of(v) for v in (0, 0, 0, 5, 0, 0, 0, 0, 1, -90, 0, 30)]
    data = await started_client.call(CommandName.SCAN_ON_CIRCLE, *args)
    last = [v.value for v in _numeric(data[-1]).values[:3]]
    assert last == pytest.approx([0.0, -5.0, 0.0], abs=1e-9)


async def test_scan_on_circle_updates_position(started_client: IppDmeClient) -> None:
    args = [Number.of(v) for v in (0, 0, 0, 5, 0, 0, 0, 0, 1, 90, 0, 30)]
    await started_client.call(CommandName.SCAN_ON_CIRCLE, *args)
    (data,) = await started_client.call(
        CommandName.GET, NamedValue(ParameterName.X, ()), NamedValue(ParameterName.Y, ())
    )
    assert isinstance(data, Items)
    values = {}
    for nv in data.values:
        arg = nv.args[0]
        assert isinstance(arg, Number)
        values[nv.name] = arg.value
    assert values["X"] == pytest.approx(0.0, abs=1e-9)
    assert values["Y"] == pytest.approx(5.0, abs=1e-9)


async def test_scan_on_circle_zero_radius_errors(started_client: IppDmeClient) -> None:
    args = [Number.of(v) for v in (0, 0, 0, 0, 0, 0, 0, 0, 1, 90, 0, 30)]
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(CommandName.SCAN_ON_CIRCLE, *args)
    assert excinfo.value.error.number == "0502"


async def test_scan_on_line_zero_length_errors(started_client: IppDmeClient) -> None:
    args = [Number.of(v) for v in (0, 0, 0, 0, 0, 0, 0, 0, 1, 1)]
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(CommandName.SCAN_ON_LINE, *args)
    assert excinfo.value.error.number == "0502"


async def test_scan_on_line_non_orthogonal_normal_errors(started_client: IppDmeClient) -> None:
    # Direction is +X; a normal of +X (parallel, not orthogonal) must be rejected.
    args = [Number.of(v) for v in (0, 0, 0, 10, 0, 0, 1, 0, 0, 1)]
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(CommandName.SCAN_ON_LINE, *args)
    assert excinfo.value.error.number == "0502"


async def test_scan_on_circle_non_orthogonal_normal_errors(started_client: IppDmeClient) -> None:
    args = [
        Number.of(v) for v in (0, 0, 0, 5, 0, 0, 1, 0, 0, 90, 0, 30)
    ]  # normal parallel to radial
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(CommandName.SCAN_ON_CIRCLE, *args)
    assert excinfo.value.error.number == "0502"


async def test_scan_on_helix_rises_along_normal_each_turn(started_client: IppDmeClient) -> None:
    # Center (0,0,0), start (5,0,0), normal +Z, one full turn (delta=360), pitch=10 -> ends up
    # back over the start XY position but 10 units higher along Z.
    args = [Number.of(v) for v in (0, 0, 0, 5, 0, 0, 0, 0, 1, 360, 0, 90, 10)]
    data = await started_client.call(CommandName.SCAN_ON_HELIX, *args)
    first = [v.value for v in _numeric(data[0]).values[:3]]
    last = [v.value for v in _numeric(data[-1]).values[:3]]
    assert first == pytest.approx([5.0, 0.0, 0.0], abs=1e-9)
    assert last == pytest.approx([5.0, 0.0, 10.0], abs=1e-9)


async def test_scan_on_helix_rejects_forbidden_sfa(started_client: IppDmeClient) -> None:
    for sfa in (90, 270):
        args = [Number.of(v) for v in (0, 0, 0, 5, 0, 0, 0, 0, 1, 360, sfa, 90, 10)]
        with pytest.raises(IppDmeServerError) as excinfo:
            await started_client.call(CommandName.SCAN_ON_HELIX, *args)
        assert excinfo.value.error.number == "0509"
        await started_client.call(CommandName.CLEAR_ALL_ERRORS)


async def test_scan_on_helix_non_orthogonal_normal_errors(started_client: IppDmeClient) -> None:
    args = [
        Number.of(v) for v in (0, 0, 0, 5, 0, 0, 1, 0, 0, 360, 0, 90, 10)
    ]  # normal parallel to radial
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(CommandName.SCAN_ON_HELIX, *args)
    assert excinfo.value.error.number == "0502"


async def test_scan_on_helix_non_positive_step_w_errors(started_client: IppDmeClient) -> None:
    args = [Number.of(v) for v in (0, 0, 0, 5, 0, 0, 0, 0, 1, 360, 0, 0, 10)]
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(CommandName.SCAN_ON_HELIX, *args)
    assert excinfo.value.error.number == "0509"


async def test_scan_on_line_non_positive_step_w_errors(started_client: IppDmeClient) -> None:
    args = [Number.of(v) for v in (0, 0, 0, 10, 0, 0, 0, 0, 1, 0)]
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(CommandName.SCAN_ON_LINE, *args)
    assert excinfo.value.error.number == "0509"


async def test_scan_on_line_wrong_argument_count_errors(started_client: IppDmeClient) -> None:
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(CommandName.SCAN_ON_LINE, *(Number.of(v) for v in (0, 0, 0)))
    assert excinfo.value.error.number == "0509"


async def test_scan_on_line_named_argument_rejected(started_client: IppDmeClient) -> None:
    # ScanOnLine's arguments are positional/unnamed; a NamedValue is not a Number.
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(
            CommandName.SCAN_ON_LINE,
            NamedValue(ParameterName.X, (Number.of(0),)),
            *(Number.of(v) for v in (0, 0, 10, 0, 0, 0, 0, 1, 1)),
        )
    assert excinfo.value.error.number == "0509"


async def test_scan_hints_are_accepted_and_have_no_observable_effect(
    started_client: IppDmeClient,
) -> None:
    await started_client.call(CommandName.SCAN_ON_CIRCLE_HINT, Number.of(0.1), Number.of(0.05))
    await started_client.call(CommandName.SCAN_ON_LINE_HINT, Number.of(1.0), Number.of(0.02))


async def test_scan_on_circle_bare_name_argument_rejected(started_client: IppDmeClient) -> None:
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(
            CommandName.SCAN_ON_CIRCLE_HINT, BasicName("NotANumber"), Number.of(0.05)
        )
    assert excinfo.value.error.number == "0509"


def _sphere_args(
    start: tuple[float, float, float] = (0, 0, 0),
    end: tuple[float, float, float] = (10, 0, 0),
    step_w: float = 2.0,
    dia: float = 1.0,
    n: int = 1,
) -> list[Number]:
    values = (
        *start,
        1,
        0,
        0,  # Si,Sj,Sk (unused by this simplified simulation)
        0,
        0,
        1,  # Ni,Nj,Nk: orthogonal to the default X-axis start-to-end vector
        start[0] + 0.1,
        start[1],
        start[2],  # Dx,Dy,Dz: near, but not coincident with, the start point
        step_w,
        *end,
        dia,
        n,
        1,
        0,
        0,  # Ei,Ej,Ek (unused)
    )
    return [Number.of(v) for v in values]


async def test_scan_in_plane_end_is_sphere_reaches_the_end_point(
    started_client: IppDmeClient,
) -> None:
    data = await started_client.call(CommandName.SCAN_IN_PLANE_END_IS_SPHERE, *_sphere_args())
    last = _numeric(data[-1]).values
    assert [n.value for n in last[:3]] == pytest.approx([10.0, 0.0, 0.0])


async def test_scan_in_plane_end_is_sphere_stops_after_nth_entry(
    started_client: IppDmeClient,
) -> None:
    # A generous Dia (3.0) with StepW=2.0 means the point at x=8 (distance 2
    # from the end point (10,0,0)) already satisfies the stop criterion; with
    # n=1 the scan must stop there, one point short of the full 6-point line.
    data = await started_client.call(
        CommandName.SCAN_IN_PLANE_END_IS_SPHERE, *_sphere_args(step_w=2.0, dia=3.0, n=1)
    )
    xs = [_numeric(d).values[0].value for d in data]
    assert xs[-1] == pytest.approx(8.0)
    assert len(data) == 5  # x = 0, 2, 4, 6, 8 - stopped before reaching x=10


async def test_scan_in_plane_end_is_sphere_rejects_zero_step(
    started_client: IppDmeClient,
) -> None:
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(
            CommandName.SCAN_IN_PLANE_END_IS_SPHERE, *_sphere_args(step_w=0.0)
        )
    assert excinfo.value.error.number == "0509"


async def test_scan_in_plane_end_is_sphere_rejects_non_positive_dia(
    started_client: IppDmeClient,
) -> None:
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(CommandName.SCAN_IN_PLANE_END_IS_SPHERE, *_sphere_args(dia=0.0))
    assert excinfo.value.error.number == "0509"


async def test_scan_in_plane_end_is_sphere_rejects_n_less_than_one(
    started_client: IppDmeClient,
) -> None:
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(CommandName.SCAN_IN_PLANE_END_IS_SPHERE, *_sphere_args(n=0))
    assert excinfo.value.error.number == "0509"


async def test_scan_unknown_hint_accepted(started_client: IppDmeClient) -> None:
    await started_client.call(CommandName.SCAN_UNKNOWN_HINT, Number.of(2.5))


async def test_scan_unknown_hint_rejects_non_positive(started_client: IppDmeClient) -> None:
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(CommandName.SCAN_UNKNOWN_HINT, Number.of(0.0))
    assert excinfo.value.error.number == "0509"


async def test_scan_on_curve_hint_accepted(started_client: IppDmeClient) -> None:
    await started_client.call(CommandName.SCAN_ON_CURVE_HINT, Number.of(0.01), Number.of(2.5))


async def test_scan_on_curve_hint_rejects_non_positive_min_radius(
    started_client: IppDmeClient,
) -> None:
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(CommandName.SCAN_ON_CURVE_HINT, Number.of(0.01), Number.of(0.0))
    assert excinfo.value.error.number == "0509"


async def test_scan_on_curve_density_accepts_any_subset_of_named_args(
    started_client: IppDmeClient,
) -> None:
    await started_client.call(CommandName.SCAN_ON_CURVE_DENSITY)
    await started_client.call(
        CommandName.SCAN_ON_CURVE_DENSITY, NamedValue(ParameterName.DIS, (Number.of(0.5),))
    )
    await started_client.call(
        CommandName.SCAN_ON_CURVE_DENSITY,
        NamedValue(ParameterName.ANGLE, (Number.of(10.0),)),
        NamedValue(ParameterName.AT_NOMINALS, (Number.of(1),)),
    )


async def test_scan_on_curve_density_rejects_unknown_name(started_client: IppDmeClient) -> None:
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(
            CommandName.SCAN_ON_CURVE_DENSITY, NamedValue("Bogus", (Number.of(1),))
        )
    assert excinfo.value.error.number == "0510"


async def test_scan_unknown_density_accepts_any_subset_of_named_args(
    started_client: IppDmeClient,
) -> None:
    await started_client.call(CommandName.SCAN_UNKNOWN_DENSITY)
    await started_client.call(
        CommandName.SCAN_UNKNOWN_DENSITY, NamedValue(ParameterName.DIS, (Number.of(0.5),))
    )
    await started_client.call(
        CommandName.SCAN_UNKNOWN_DENSITY,
        NamedValue(ParameterName.DIS, (Number.of(0.5),)),
        NamedValue(ParameterName.ANGLE, (Number.of(10.0),)),
        NamedValue(ParameterName.ANGLE_BASE_LENGTH, (Number.of(1.0),)),
    )


async def test_scan_unknown_density_rejects_unknown_name(started_client: IppDmeClient) -> None:
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(
            CommandName.SCAN_UNKNOWN_DENSITY, NamedValue("Bogus", (Number.of(1),))
        )
    assert excinfo.value.error.number == "0510"


async def test_scan_unknown_density_requires_angle_and_base_length_together(
    started_client: IppDmeClient,
) -> None:
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(
            CommandName.SCAN_UNKNOWN_DENSITY, NamedValue(ParameterName.ANGLE, (Number.of(10.0),))
        )
    assert excinfo.value.error.number == "0502"
    await started_client.call(CommandName.CLEAR_ALL_ERRORS)

    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(
            CommandName.SCAN_UNKNOWN_DENSITY,
            NamedValue(ParameterName.ANGLE_BASE_LENGTH, (Number.of(1.0),)),
        )
    assert excinfo.value.error.number == "0502"


_SCAN_ON_CURVE_FORMAT = (
    NamedValue(ParameterName.X, ()),
    NamedValue(ParameterName.Y, ()),
    NamedValue(ParameterName.Z, ()),
    NamedValue(ParameterName.IJK, ()),
    BasicName(ParameterName.TAG),
)


def _curve_data(*points: tuple[float, float, float, float, float, float, float]) -> NamedValue:
    numbers = [Number.of(v) for point in points for v in point]
    return NamedValue(ParameterName.DATA, tuple(numbers))


async def test_scan_on_curve_streams_nominal_points(started_client: IppDmeClient) -> None:
    # The exact shape of the NIST/I++ DME reference test suite's
    # ScanOnCurve.prg (see the module docstring): two points, tags +1/+1.
    data = await started_client.call(
        CommandName.SCAN_ON_CURVE,
        NamedValue(ParameterName.CLOSED, (Number.of(0),)),
        NamedValue(ParameterName.FORMAT, _SCAN_ON_CURVE_FORMAT),
        _curve_data((0, 0, 0, 0, 0, 1, 555), (100, 0, 0, 0, 0, 1, 666)),
    )
    assert len(data) == 2
    assert [n.value for n in _numeric(data[0]).values[:3]] == pytest.approx([0.0, 0.0, 0.0])
    assert [n.value for n in _numeric(data[1]).values[:3]] == pytest.approx([100.0, 0.0, 0.0])


async def test_scan_on_curve_parses_real_nist_wire_example(started_client: IppDmeClient) -> None:
    line = (
        "ScanOnCurve(Closed(0),Format(X(),Y(),Z(),IJK(),tag),"
        "Data(0.,0.,0.,0.,0.,1.,555,100.,0.,0.,0.,0.,1.,666))"
    )
    method = parse_method(line)
    data = await started_client.call(method.name, *method.args)
    assert len(data) == 2


async def test_scan_on_curve_rejects_bad_data_length(started_client: IppDmeClient) -> None:
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(
            CommandName.SCAN_ON_CURVE,
            NamedValue(ParameterName.CLOSED, (Number.of(0),)),
            NamedValue(ParameterName.FORMAT, _SCAN_ON_CURVE_FORMAT),
            NamedValue(ParameterName.DATA, (Number.of(0), Number.of(0), Number.of(0))),
        )
    assert excinfo.value.error.number == "0509"


async def test_scan_on_curve_rejects_unsupported_format(started_client: IppDmeClient) -> None:
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(
            CommandName.SCAN_ON_CURVE,
            NamedValue(ParameterName.CLOSED, (Number.of(0),)),
            NamedValue(
                ParameterName.FORMAT,
                (NamedValue(ParameterName.X, ()), NamedValue(ParameterName.Y, ())),
            ),
            _curve_data((0, 0, 0, 0, 0, 1, 555)),
        )
    assert excinfo.value.error.number == "0509"


async def test_scan_on_curve_rejects_missing_mandatory_argument(
    started_client: IppDmeClient,
) -> None:
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(
            CommandName.SCAN_ON_CURVE,
            NamedValue(ParameterName.CLOSED, (Number.of(0),)),
            _curve_data((0, 0, 0, 0, 0, 1, 555)),
        )
    assert excinfo.value.error.number == "0502"


async def test_scan_on_curve_rejects_unsupported_argument(started_client: IppDmeClient) -> None:
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(
            CommandName.SCAN_ON_CURVE,
            NamedValue(ParameterName.CLOSED, (Number.of(0),)),
            NamedValue(ParameterName.FORMAT, _SCAN_ON_CURVE_FORMAT),
            _curve_data((0, 0, 0, 0, 0, 1, 555)),
            NamedValue("Bogus", ()),
        )
    assert excinfo.value.error.number == "0506"


async def test_scan_on_curve_honors_on_scan_report(started_client: IppDmeClient) -> None:
    await started_client.call(
        CommandName.ON_SCAN_REPORT, NamedValue(ParameterName.X, ()), NamedValue(ParameterName.Y, ())
    )
    data = await started_client.call(
        CommandName.SCAN_ON_CURVE,
        NamedValue(ParameterName.CLOSED, (Number.of(0),)),
        NamedValue(ParameterName.FORMAT, _SCAN_ON_CURVE_FORMAT),
        _curve_data((1, 2, 3, 0, 0, 1, 555)),
    )
    (values,) = data
    assert [n.value for n in _numeric(values).values] == pytest.approx([1.0, 2.0])


def test_arc_length_matches_hand_computed_geometry() -> None:
    # Sanity-check the test's own expectations against basic trig, independent
    # of the implementation, so a future regression can't silently "fix" both.
    radius = 5.0
    theta = math.radians(60)
    assert (radius * math.cos(theta), radius * math.sin(theta)) == pytest.approx(
        (2.5, 4.330127018922193)
    )


async def test_scan_on_line_reports_ijk_as_its_own_orientation(
    started_client: IppDmeClient,
) -> None:
    await started_client.call(
        CommandName.ON_SCAN_REPORT,
        NamedValue(ParameterName.X, ()),
        NamedValue(ParameterName.IJK, ()),
    )
    data = await started_client.call(
        CommandName.SCAN_ON_LINE, *(Number.of(v) for v in (0, 0, 0, 10, 0, 0, 0, 0, 1, 5))
    )
    values = _numeric(data[0]).values
    assert [n.value for n in values] == pytest.approx([0.0, 0.0, 0.0, 1.0])


async def test_scan_on_curve_reports_each_points_own_ijk(started_client: IppDmeClient) -> None:
    await started_client.call(CommandName.ON_SCAN_REPORT, NamedValue(ParameterName.IJK, ()))
    data = await started_client.call(
        CommandName.SCAN_ON_CURVE,
        NamedValue(ParameterName.CLOSED, (Number.of(0),)),
        NamedValue(ParameterName.FORMAT, _SCAN_ON_CURVE_FORMAT),
        _curve_data((1, 2, 3, 0.1, 0.2, 0.3, 555), (4, 5, 6, 0.4, 0.5, 0.6, 666)),
    )
    assert [n.value for n in _numeric(data[0]).values] == pytest.approx([0.1, 0.2, 0.3])
    assert [n.value for n in _numeric(data[1]).values] == pytest.approx([0.4, 0.5, 0.6])


async def test_scan_in_plane_end_is_plane_stops_at_the_stop_plane(
    started_client: IppDmeClient,
) -> None:
    args = (
        0,
        0,
        0,  # Sx,Sy,Sz
        1,
        0,
        0,  # Si,Sj,Sk (unused)
        0,
        0,
        1,  # Ni,Nj,Nk: orthogonal to the start-to-P vector (10,0,0)
        0.1,
        0,
        0,  # Dx,Dy,Dz
        2.0,  # StepW
        10,
        0,
        0,  # Px,Py,Pz
        1,
        0,
        0,  # Pi,Pj,Pk
        1,  # n
        1,
        0,
        0,  # Ei,Ej,Ek (unused)
    )
    data = await started_client.call(
        CommandName.SCAN_IN_PLANE_END_IS_PLANE, *(Number.of(v) for v in args)
    )
    xs = [_numeric(d).values[0].value for d in data]
    assert xs == pytest.approx([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])


async def test_scan_in_plane_end_is_plane_rejects_zero_step(
    started_client: IppDmeClient,
) -> None:
    args = (0, 0, 0, 1, 0, 0, 0, 0, 1, 0.1, 0, 0, 0.0, 10, 0, 0, 1, 0, 0, 1, 1, 0, 0)
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(
            CommandName.SCAN_IN_PLANE_END_IS_PLANE, *(Number.of(v) for v in args)
        )
    assert excinfo.value.error.number == "0509"


async def test_scan_in_plane_end_is_cyl_stops_within_the_stop_cylinder(
    started_client: IppDmeClient,
) -> None:
    args = (
        0,
        0,
        0,  # Sx,Sy,Sz
        1,
        0,
        0,  # Si,Sj,Sk (unused)
        0,
        0,
        1,  # Ni,Nj,Nk: orthogonal to the start-to-C vector (10,0,0)
        0.1,
        0,
        0,  # Dx,Dy,Dz
        2.0,  # StepW
        10,
        0,
        0,  # Cx,Cy,Cz
        0,
        1,
        0,  # Ci,Cj,Ck: cylinder axis along Y
        1.0,  # d
        1,  # n
        1,
        0,
        0,  # Ei,Ej,Ek (unused)
    )
    data = await started_client.call(
        CommandName.SCAN_IN_PLANE_END_IS_CYL, *(Number.of(v) for v in args)
    )
    xs = [_numeric(d).values[0].value for d in data]
    assert xs == pytest.approx([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])


async def test_scan_in_plane_end_is_cyl_rejects_non_positive_d(
    started_client: IppDmeClient,
) -> None:
    args = (0, 0, 0, 1, 0, 0, 0, 0, 1, 0.1, 0, 0, 2.0, 10, 0, 0, 0, 1, 0, 0.0, 1, 1, 0, 0)
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(
            CommandName.SCAN_IN_PLANE_END_IS_CYL, *(Number.of(v) for v in args)
        )
    assert excinfo.value.error.number == "0509"


async def test_scan_in_cyl_end_is_sphere_reaches_the_stop_sphere(
    started_client: IppDmeClient,
) -> None:
    args = (
        0,
        0,
        0,  # Cx,Cy,Cz
        0,
        0,
        1,  # Ci,Cj,Ck: cylinder axis along Z
        1,
        0,
        0,  # Sx,Sy,Sz: radial distance 1 from the axis
        0,
        0,
        1,  # Si,Sj,Sk (unused)
        1.1,
        0,
        0,  # Dx,Dy,Dz
        2.0,  # StepW
        11,
        0,
        0,  # Ex,Ey,Ez
        1.0,  # Dia
        1,  # n
        1,
        0,
        0,  # Ei,Ej,Ek (unused)
    )
    data = await started_client.call(
        CommandName.SCAN_IN_CYL_END_IS_SPHERE, *(Number.of(v) for v in args)
    )
    xs = [_numeric(d).values[0].value for d in data]
    assert xs == pytest.approx([1.0, 3.0, 5.0, 7.0, 9.0, 11.0])


async def test_scan_in_cyl_end_is_sphere_rejects_start_on_axis(
    started_client: IppDmeClient,
) -> None:
    args = (0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1.1, 0, 0, 2.0, 11, 0, 0, 1.0, 1, 1, 0, 0)
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(
            CommandName.SCAN_IN_CYL_END_IS_SPHERE, *(Number.of(v) for v in args)
        )
    assert excinfo.value.error.number == "0509"


async def test_scan_in_cyl_end_is_plane_stops_at_the_stop_plane(
    started_client: IppDmeClient,
) -> None:
    args = (
        0,
        0,
        0,  # Cx,Cy,Cz
        0,
        0,
        1,  # Ci,Cj,Ck: cylinder axis along Z
        1,
        0,
        0,  # Sx,Sy,Sz: radial distance 1 from the axis
        0,
        0,
        1,  # Si,Sj,Sk (unused)
        1.1,
        0,
        0,  # Dx,Dy,Dz
        2.0,  # StepW
        11,
        0,
        0,  # Px,Py,Pz
        1,
        0,
        0,  # Pi,Pj,Pk
        1,  # n
        1,
        0,
        0,  # Ei,Ej,Ek (unused)
    )
    data = await started_client.call(
        CommandName.SCAN_IN_CYL_END_IS_PLANE, *(Number.of(v) for v in args)
    )
    xs = [_numeric(d).values[0].value for d in data]
    assert xs == pytest.approx([1.0, 3.0, 5.0, 7.0, 9.0, 11.0])


async def test_scan_in_cyl_end_is_plane_rejects_direction_point_coincident_with_start(
    started_client: IppDmeClient,
) -> None:
    args = (0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 2.0, 11, 0, 0, 1, 0, 0, 1, 1, 0, 0)
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(
            CommandName.SCAN_IN_CYL_END_IS_PLANE, *(Number.of(v) for v in args)
        )
    assert excinfo.value.error.number == "0509"
