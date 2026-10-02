# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for OnPtMeasReport's field selection/validation (6.12.1 Table 75)."""

from __future__ import annotations

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


async def test_on_pt_meas_report_accepts_q_and_er(started_client: IppDmeClient) -> None:
    await started_client.call(
        CommandName.ON_PT_MEAS_REPORT,
        NamedValue(ParameterName.X, ()),
        NamedValue(ParameterName.Q, ()),
        NamedValue(ParameterName.ER, ()),
    )
    (data,) = await started_client.call(
        CommandName.PT_MEAS, NamedValue(ParameterName.X, (Number.of(3),))
    )
    values = {nv.name: _num(nv) for nv in _items(data).values}
    assert values["X"] == pytest.approx(3.0)
    assert values["Q"] == 0.0
    assert values["ER"] == 0.0


async def test_on_pt_meas_report_accepts_r(started_client: IppDmeClient) -> None:
    await started_client.call(
        CommandName.ALIGN_PART,
        Number.of(1.0),
        Number.of(0.0),
        Number.of(0.0),
        Number.of(0.0),
        Number.of(1.0),
        Number.of(0.0),
        Number.of(0.0),
    )
    await started_client.call(CommandName.ON_PT_MEAS_REPORT, NamedValue(ParameterName.R, ()))
    (data,) = await started_client.call(CommandName.PT_MEAS)
    (value,) = _items(data).values
    assert value.name == "R"
    assert _num(value) == pytest.approx(90.0)


async def test_on_pt_meas_report_accepts_ijk_and_ijk_act(started_client: IppDmeClient) -> None:
    await started_client.call(
        CommandName.ON_PT_MEAS_REPORT,
        NamedValue(ParameterName.IJK, ()),
        NamedValue(ParameterName.IJK_ACT, ()),
    )
    (data,) = await started_client.call(
        CommandName.PT_MEAS,
        NamedValue(ParameterName.X, (Number.of(3),)),
        NamedValue("IJK", (Number.of(0.0), Number.of(0.0), Number.of(1.0))),
    )
    values = {
        nv.name: tuple(a.value for a in nv.args if isinstance(a, Number))
        for nv in _items(data).values
    }
    assert values["IJK"] == pytest.approx((0.0, 0.0, 1.0))
    assert values["IJKAct"] == pytest.approx((0.0, 0.0, 1.0))


async def test_on_pt_meas_report_ijk_falls_back_to_motion_vector(
    started_client: IppDmeClient,
) -> None:
    await started_client.call(CommandName.ON_PT_MEAS_REPORT, NamedValue(ParameterName.IJK, ()))
    (data,) = await started_client.call(
        CommandName.PT_MEAS, NamedValue(ParameterName.X, (Number.of(5),))
    )
    (value,) = _items(data).values
    assert value.name == "IJK"
    numbers = tuple(a.value for a in value.args if isinstance(a, Number))
    assert numbers == pytest.approx((1.0, 0.0, 0.0))


async def test_on_pt_meas_report_rejects_unknown_name(started_client: IppDmeClient) -> None:
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(CommandName.ON_PT_MEAS_REPORT, NamedValue("Bogus", ()))
    assert excinfo.value.error.number == "0506"
