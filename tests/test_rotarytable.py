# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for the RotaryTable class subset: R axis, EnableRotaryTableVarCsy, AlignPart."""

from __future__ import annotations

import math

import pytest

from pyippdme import IppDmeClient
from pyippdme.exceptions import IppDmeServerError
from pyippdme.protocol.ast import DataPayload, Items, NamedValue, Number
from pyippdme.protocol.commands import CommandName
from pyippdme.protocol.parameters import ParameterName


def _items(data: DataPayload) -> Items:
    assert isinstance(data, Items)
    return data


def _num(nv: NamedValue) -> float:
    arg = nv.args[0]
    assert isinstance(arg, Number)
    return arg.value


async def _r_value(client: IppDmeClient) -> float:
    (data,) = await client.call(CommandName.GET, NamedValue(ParameterName.R, ()))
    (value,) = _items(data).values
    return _num(value)


async def test_r_axis_defaults_to_zero(started_client: IppDmeClient) -> None:
    assert await _r_value(started_client) == 0.0


async def test_go_to_sets_r_axis(started_client: IppDmeClient) -> None:
    await started_client.call(CommandName.GO_TO, NamedValue(ParameterName.R, (Number.of(45.0),)))
    assert await _r_value(started_client) == 45.0


async def test_go_to_does_not_disturb_r_when_omitted(started_client: IppDmeClient) -> None:
    await started_client.call(CommandName.GO_TO, NamedValue(ParameterName.R, (Number.of(30.0),)))
    await started_client.call(CommandName.GO_TO, NamedValue(ParameterName.X, (Number.of(1.0),)))
    assert await _r_value(started_client) == 30.0


async def test_enable_rotary_table_var_csy(started_client: IppDmeClient) -> None:
    result = await started_client.call(CommandName.ENABLE_ROTARY_TABLE_VAR_CSY, Number.of(1))
    assert result == ()


async def test_align_part_rotates_table_to_align_vectors(started_client: IppDmeClient) -> None:
    # Part vector along +X, machine vector along +Y: the table must turn 90 degrees.
    args = [Number.of(v) for v in (1, 0, 0, 0, 1, 0, 0)]
    (data,) = await started_client.call(CommandName.ALIGN_PART, *args)
    assert data.to_wire().count(",") == 5  # 6 comma-separated numbers

    assert await _r_value(started_client) == pytest.approx(90.0)


async def test_align_part_rejects_zero_vector(started_client: IppDmeClient) -> None:
    args = [Number.of(v) for v in (0, 0, 0, 0, 1, 0, 0)]
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(CommandName.ALIGN_PART, *args)
    assert excinfo.value.error.number == "1010"


async def test_align_part_aligns_a_second_rotary_table(started_client: IppDmeClient) -> None:
    args = [Number.of(v) for v in (1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0)]
    (data,) = await started_client.call(CommandName.ALIGN_PART, *args)
    numbers = [float(n) for n in data.to_wire().split(",")]
    # Returns the projected, normalized vectors, "same number as set" (6.23.1).
    assert numbers == pytest.approx([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1])


async def test_align_part_second_table_needs_all_its_values(started_client: IppDmeClient) -> None:
    args = [Number.of(v) for v in (1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0)]  # beta missing
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(CommandName.ALIGN_PART, *args)
    assert excinfo.value.error.number == "0509"


async def test_align_part_second_table_reports_part_not_aligned(
    started_client: IppDmeClient,
) -> None:
    # A negative allowed error angle can never be met.
    args = [Number.of(v) for v in (1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, -1)]
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(CommandName.ALIGN_PART, *args)
    assert excinfo.value.error.number == "2506"


def test_math_sanity_for_align_angle() -> None:
    # 90 degrees expressed via atan2 difference, as a cross-check of the
    # handler's own math (not calling the handler itself).
    part_angle = math.degrees(math.atan2(0, 1))
    machine_angle = math.degrees(math.atan2(1, 0))
    assert (machine_angle - part_angle) % 360 == pytest.approx(90.0)
