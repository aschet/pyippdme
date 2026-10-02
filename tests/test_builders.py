# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for pyippdme.client.builders: typed Argument-tuple construction for NamedValue commands."""

from __future__ import annotations

import pytest

from pyippdme.client import IppDmeClient, builders
from pyippdme.client.builders import CurvePoint
from pyippdme.protocol.ast import BasicName, NamedValue, Number
from pyippdme.protocol.commands import CommandName
from pyippdme.protocol.parameters import ParameterName
from pyippdme.types.csy import CoordinateTransform


def test_bare_names_builds_one_empty_named_value_per_name() -> None:
    assert builders.bare_names("X", "Y") == (NamedValue("X", ()), NamedValue("Y", ()))


def test_named_numbers_omits_none_pairs() -> None:
    assert builders.named_numbers(("X", 10), ("Y", None), ("Z", 5)) == (
        NamedValue("X", (Number.of(10),)),
        NamedValue("Z", (Number.of(5),)),
    )


def test_go_to_omits_unset_axes() -> None:
    args = builders.go_to(x=10, z=5)
    assert args == (
        NamedValue(ParameterName.X, (Number.of(10),)),
        NamedValue(ParameterName.Z, (Number.of(5),)),
    )


def test_go_to_with_nothing_set_is_empty() -> None:
    assert builders.go_to() == ()


def test_step_and_pt_meas_share_go_to_s_shape() -> None:
    assert builders.step(y=1) == (NamedValue(ParameterName.Y, (Number.of(1),)),)
    assert builders.pt_meas(x=2) == (NamedValue(ParameterName.X, (Number.of(2),)),)


def test_get_builds_bare_named_values() -> None:
    assert builders.get("X", "Y", "R") == (
        NamedValue("X", ()),
        NamedValue("Y", ()),
        NamedValue("R", ()),
    )


@pytest.mark.parametrize(
    "func",
    [builders.on_pt_meas_report, builders.get_scale_temperatures, builders.lock_axis],
)
def test_bare_axis_builders_match_get_s_shape(func: object) -> None:
    assert func("X", "Y") == (NamedValue("X", ()), NamedValue("Y", ()))  # type: ignore[operator]


def test_lock_position_builds_bare_named_values() -> None:
    assert builders.lock_position("Center") == (NamedValue("Center", ()),)


def test_enum_prop_and_enum_all_prop_wrap_a_single_reference() -> None:
    assert builders.enum_prop("Tool.GoToPar") == (NamedValue("Tool.GoToPar", ()),)
    assert builders.enum_all_prop("Part") == (NamedValue("Part", ()),)


def test_get_prop_supports_dotted_property_paths() -> None:
    assert builders.get_prop("Tool.GoToPar.Speed", "Part.Temperature") == (
        NamedValue("Tool.GoToPar.Speed", ()),
        NamedValue("Part.Temperature", ()),
    )


def test_set_prop_wraps_a_single_numeric_value() -> None:
    assert builders.set_prop("Tool.GoToPar.Speed", 50.0) == (
        NamedValue("Tool.GoToPar.Speed", (Number.of(50.0),)),
    )


def test_set_scale_temperatures_builds_one_named_value_per_axis() -> None:
    args = builders.set_scale_temperatures(X=21.0, Y=22.0)
    assert args == (
        NamedValue("X", (Number.of(21.0),)),
        NamedValue("Y", (Number.of(22.0),)),
    )


def test_go_to_on_circle_and_spiral_share_the_same_shape() -> None:
    expected = (
        NamedValue(ParameterName.CENTER, (Number.of(0), Number.of(0), Number.of(0))),
        NamedValue(ParameterName.IJK, (Number.of(0), Number.of(0), Number.of(1))),
        NamedValue(ParameterName.X, (Number.of(5),)),
        NamedValue(ParameterName.Y, (Number.of(0),)),
        NamedValue(ParameterName.Z, (Number.of(0),)),
    )
    assert builders.go_to_on_circle((0, 0, 0), (0, 0, 1), (5, 0, 0)) == expected
    assert builders.go_to_on_spiral((0, 0, 0), (0, 0, 1), (5, 0, 0)) == expected


def test_on_scan_report_builds_bare_named_values() -> None:
    assert builders.on_scan_report("X", "Y") == (NamedValue("X", ()), NamedValue("Y", ()))


def test_set_csy_transformation_is_a_fixed_order_positional_number_list() -> None:
    transform = CoordinateTransform(1, 2, 3, 45, 90, 180)
    args = builders.set_csy_transformation(transform)
    assert args == tuple(Number.of(v) for v in (1, 2, 3, 45, 90, 180))


def test_scan_on_curve_builds_closed_format_and_flattened_data() -> None:
    points = [
        CurvePoint((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 1),
        CurvePoint((10.0, 0.0, 0.0), (0.0, 0.0, 1.0), -1),
    ]
    closed_arg, format_arg, data_arg = builders.scan_on_curve(points, closed=True)

    assert closed_arg == NamedValue(ParameterName.CLOSED, (Number.of(1),))
    assert isinstance(format_arg, NamedValue)
    assert format_arg.name == ParameterName.FORMAT
    assert format_arg.args == (
        NamedValue(ParameterName.X, ()),
        NamedValue(ParameterName.Y, ()),
        NamedValue(ParameterName.Z, ()),
        NamedValue(ParameterName.IJK, ()),
        BasicName(ParameterName.TAG),
    )
    assert isinstance(data_arg, NamedValue)
    assert data_arg.name == ParameterName.DATA
    numbers = [n for n in data_arg.args if isinstance(n, Number)]
    assert len(numbers) == len(data_arg.args)
    assert [n.value for n in numbers] == [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        10.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        -1.0,
    ]


async def test_go_to_builder_round_trips_through_a_real_server(
    started_client: IppDmeClient,
) -> None:
    """One end-to-end check that a builder's output is actually valid wire syntax."""
    await started_client.call(CommandName.GO_TO, *builders.go_to(x=10, y=20))
    (data,) = await started_client.call(CommandName.GET, *builders.get("X", "Y", "Z"))
    values = {nv.name: nv.args[0].value for nv in data.values}  # type: ignore[union-attr]
    assert values == {"X": 10.0, "Y": 20.0, "Z": 0.0}
