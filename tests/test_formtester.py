# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for the FormTester class subset (alignment status commands, axis/position locking)."""

from __future__ import annotations

import pytest

from pyippdme import IppDmeClient
from pyippdme.exceptions import IppDmeServerError
from pyippdme.protocol.ast import BasicName, DataPayload, Items, NamedValue, Number, NumericData
from pyippdme.protocol.commands import CommandName
from pyippdme.protocol.parameters import ParameterName


def _int_status(data: object) -> int:
    assert isinstance(data, NumericData)
    (value,) = data.values
    return int(value.value)


def _items(data: DataPayload) -> Items:
    assert isinstance(data, Items)
    return data


def _num(nv: NamedValue) -> float:
    arg = nv.args[0]
    assert isinstance(arg, Number)
    return arg.value


async def test_center_part_within_tolerance_reports_1(started_client: IppDmeClient) -> None:
    # distance = hypot(3, 4) = 5
    (data,) = await started_client.call(
        CommandName.CENTER_PART, *(Number.of(v) for v in (3, 4, 0, 10))
    )
    assert _int_status(data) == 1


async def test_center_part_outside_tolerance_reports_0(started_client: IppDmeClient) -> None:
    (data,) = await started_client.call(
        CommandName.CENTER_PART, *(Number.of(v) for v in (3, 4, 0, 1))
    )
    assert _int_status(data) == 0


async def test_tilt_part_aligned_with_axis_reports_1(started_client: IppDmeClient) -> None:
    (data,) = await started_client.call(
        CommandName.TILT_PART, *(Number.of(v) for v in (0, 0, 1, 0.01))
    )
    assert _int_status(data) == 1


async def test_tilt_part_perpendicular_to_axis_reports_0(started_client: IppDmeClient) -> None:
    (data,) = await started_client.call(
        CommandName.TILT_PART, *(Number.of(v) for v in (1, 0, 0, 0.01))
    )
    assert _int_status(data) == 0


async def test_tilt_part_treats_axis_as_undirected_line(started_client: IppDmeClient) -> None:
    # -Z is just as aligned with the rotation axis as +Z (a line, not a ray).
    (data,) = await started_client.call(
        CommandName.TILT_PART, *(Number.of(v) for v in (0, 0, -1, 0.01))
    )
    assert _int_status(data) == 1


async def test_tilt_part_zero_vector_errors(started_client: IppDmeClient) -> None:
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(CommandName.TILT_PART, *(Number.of(v) for v in (0, 0, 0, 0.01)))
    assert excinfo.value.error.number == "0509"


async def test_tilt_center_part_both_within_tolerance_reports_1(
    started_client: IppDmeClient,
) -> None:
    args = (0, 0, 0, 0, 0, 10, 1)
    (data,) = await started_client.call(CommandName.TILT_CENTER_PART, *(Number.of(v) for v in args))
    assert _int_status(data) == 1


async def test_tilt_center_part_one_outside_tolerance_reports_0(
    started_client: IppDmeClient,
) -> None:
    args = (5, 5, 0, 0, 0, 10, 1)
    (data,) = await started_client.call(CommandName.TILT_CENTER_PART, *(Number.of(v) for v in args))
    assert _int_status(data) == 0


async def test_center_part_wrong_argument_count_errors(started_client: IppDmeClient) -> None:
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(CommandName.CENTER_PART, *(Number.of(v) for v in (0, 0, 0)))
    assert excinfo.value.error.number == "0509"


async def test_lock_axis_prevents_motion_on_locked_axis_only(started_client: IppDmeClient) -> None:
    await started_client.call(CommandName.LOCK_AXIS, NamedValue(ParameterName.X, ()))
    await started_client.call(
        CommandName.GO_TO,
        NamedValue(ParameterName.X, (Number.of(50),)),
        NamedValue(ParameterName.Y, (Number.of(7),)),
    )
    (data,) = await started_client.call(
        CommandName.GET, NamedValue(ParameterName.X, ()), NamedValue(ParameterName.Y, ())
    )
    values = {nv.name: _num(nv) for nv in _items(data).values}
    assert values == {"X": 0.0, "Y": 7.0}


async def test_lock_axis_locking_all_enforced_axes_errors(started_client: IppDmeClient) -> None:
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(
            CommandName.LOCK_AXIS,
            NamedValue(ParameterName.X, ()),
            NamedValue(ParameterName.Y, ()),
            NamedValue(ParameterName.Z, ()),
        )
    assert excinfo.value.error.number == "1011"


async def test_lock_axis_unknown_axis_errors(started_client: IppDmeClient) -> None:
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(CommandName.LOCK_AXIS, NamedValue(ParameterName.Q, ()))
    assert excinfo.value.error.number == "0505"


async def test_lock_axis_with_no_arguments_unlocks_everything(started_client: IppDmeClient) -> None:
    await started_client.call(CommandName.LOCK_AXIS, NamedValue(ParameterName.X, ()))
    await started_client.call(CommandName.LOCK_AXIS)  # per 6.6.1: no axes -> all axes movable
    await started_client.call(CommandName.GO_TO, NamedValue(ParameterName.X, (Number.of(42),)))
    (data,) = await started_client.call(CommandName.GET, NamedValue(ParameterName.X, ()))
    (value,) = _items(data).values
    assert _num(value) == 42.0


async def test_lock_axis_rejects_non_named_argument(started_client: IppDmeClient) -> None:
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(CommandName.LOCK_AXIS, BasicName(ParameterName.X))
    assert excinfo.value.error.number == "0510"


async def test_lock_axis_accepts_unenforced_rotary_axes(started_client: IppDmeClient) -> None:
    # R/A/B/C are valid per the standard but not enforced by this simulated machine.
    await started_client.call(
        CommandName.LOCK_AXIS, NamedValue(ParameterName.R, ()), NamedValue("A", ())
    )


async def test_lock_position_clears_locked_cartesian_axes(started_client: IppDmeClient) -> None:
    await started_client.call(CommandName.LOCK_AXIS, NamedValue(ParameterName.X, ()))
    await started_client.call(CommandName.LOCK_POSITION, NamedValue("XFR", ()))
    await started_client.call(CommandName.GO_TO, NamedValue(ParameterName.X, (Number.of(99),)))
    (data,) = await started_client.call(CommandName.GET, NamedValue(ParameterName.X, ()))
    (value,) = _items(data).values
    assert _num(value) == 99.0


async def test_lock_position_bad_combination_errors(started_client: IppDmeClient) -> None:
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(
            CommandName.LOCK_POSITION, NamedValue("XFR", ()), NamedValue("RFR", ())
        )
    assert excinfo.value.error.number == "1012"


async def test_lock_position_unknown_name_errors(started_client: IppDmeClient) -> None:
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(CommandName.LOCK_POSITION, NamedValue("QFR", ()))
    assert excinfo.value.error.number == "0505"


async def test_lock_position_valid_non_conflicting_combination_succeeds(
    started_client: IppDmeClient,
) -> None:
    await started_client.call(
        CommandName.LOCK_POSITION, NamedValue("XFR", ()), NamedValue("YFR", ())
    )
    await started_client.call(
        CommandName.LOCK_POSITION, NamedValue("RFR", ()), NamedValue("PFR", ())
    )
