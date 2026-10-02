# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for the Mover class subset (user-enable state, axes, scale temperatures)."""

from __future__ import annotations

import pytest

from pyippdme import IppDmeClient
from pyippdme.exceptions import IppDmeServerError
from pyippdme.protocol.ast import BasicName, DataPayload, Items, NamedValue, Number
from pyippdme.protocol.commands import CommandName
from pyippdme.protocol.parameters import ParameterName


def _items(data: DataPayload) -> Items:
    assert isinstance(data, Items)
    return data


def _num(nv: NamedValue) -> float:
    arg = nv.args[0]
    assert isinstance(arg, Number)
    return arg.value


async def test_is_user_enabled_defaults_to_true(started_client: IppDmeClient) -> None:
    (data,) = await started_client.call(CommandName.IS_USER_ENABLED)
    (value,) = _items(data).values
    assert _num(value) == 1.0


async def test_go_to_implicitly_disables_user(started_client: IppDmeClient) -> None:
    await started_client.call(CommandName.GO_TO, NamedValue(ParameterName.X, (Number.of(1),)))
    (data,) = await started_client.call(CommandName.IS_USER_ENABLED)
    (value,) = _items(data).values
    assert _num(value) == 0.0


async def test_pt_meas_implicitly_disables_user(started_client: IppDmeClient) -> None:
    await started_client.call(CommandName.PT_MEAS, NamedValue(ParameterName.X, (Number.of(1),)))
    (data,) = await started_client.call(CommandName.IS_USER_ENABLED)
    (value,) = _items(data).values
    assert _num(value) == 0.0


async def test_home_implicitly_disables_user(started_client: IppDmeClient) -> None:
    await started_client.call(CommandName.HOME)
    (data,) = await started_client.call(CommandName.IS_USER_ENABLED)
    (value,) = _items(data).values
    assert _num(value) == 0.0


async def test_scan_on_line_implicitly_disables_user(started_client: IppDmeClient) -> None:
    await started_client.call(
        CommandName.SCAN_ON_LINE, *(Number.of(v) for v in (0, 0, 0, 10, 0, 0, 0, 0, 1, 5))
    )
    (data,) = await started_client.call(CommandName.IS_USER_ENABLED)
    (value,) = _items(data).values
    assert _num(value) == 0.0


async def test_enable_user_re_enables(started_client: IppDmeClient) -> None:
    await started_client.call(CommandName.GO_TO, NamedValue(ParameterName.X, (Number.of(1),)))
    await started_client.call(CommandName.ENABLE_USER)
    (data,) = await started_client.call(CommandName.IS_USER_ENABLED)
    (value,) = _items(data).values
    assert _num(value) == 1.0


async def test_disable_user_explicit(started_client: IppDmeClient) -> None:
    await started_client.call(CommandName.DISABLE_USER)
    (data,) = await started_client.call(CommandName.IS_USER_ENABLED)
    (value,) = _items(data).values
    assert _num(value) == 0.0


async def test_enumerate_mover_axes(started_client: IppDmeClient) -> None:
    (data,) = await started_client.call(CommandName.ENUMERATE_MOVER_AXES)
    names = [nv.name for nv in _items(data).values]
    assert names == ["X", "Y", "Z"]
    assert all(nv.args == () for nv in _items(data).values)


async def test_update_scale_temperatures_is_accepted(started_client: IppDmeClient) -> None:
    await started_client.call(CommandName.UPDATE_SCALE_TEMPERATURES)


async def test_get_scale_temperatures_default(started_client: IppDmeClient) -> None:
    (data,) = await started_client.call(
        CommandName.GET_SCALE_TEMPERATURES, NamedValue(ParameterName.X, ())
    )
    (value,) = _items(data).values
    assert value.name == "X"
    assert _num(value) == 20.0


async def test_set_and_get_scale_temperatures_round_trip(started_client: IppDmeClient) -> None:
    await started_client.call(
        CommandName.SET_SCALE_TEMPERATURES,
        NamedValue(ParameterName.X, (Number.of(19.5),)),
        NamedValue(ParameterName.Y, (Number.of(20.5),)),
    )
    (data,) = await started_client.call(
        CommandName.GET_SCALE_TEMPERATURES,
        NamedValue(ParameterName.X, ()),
        NamedValue(ParameterName.Y, ()),
    )
    values = {nv.name: _num(nv) for nv in _items(data).values}
    assert values == {"X": 19.5, "Y": 20.5}


async def test_scale_temperatures_accept_rotary_axes(started_client: IppDmeClient) -> None:
    # Scale temperature is tracked as an encoder property, not tied to motion.
    await started_client.call(
        CommandName.SET_SCALE_TEMPERATURES, NamedValue(ParameterName.R, (Number.of(21.0),))
    )
    (data,) = await started_client.call(
        CommandName.GET_SCALE_TEMPERATURES, NamedValue(ParameterName.R, ())
    )
    (value,) = _items(data).values
    assert _num(value) == 21.0


async def test_get_scale_temperatures_unknown_axis_errors(started_client: IppDmeClient) -> None:
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(
            CommandName.GET_SCALE_TEMPERATURES, NamedValue(ParameterName.Q, ())
        )
    assert excinfo.value.error.number == "1002"


async def test_set_scale_temperatures_unknown_axis_errors(started_client: IppDmeClient) -> None:
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(
            CommandName.SET_SCALE_TEMPERATURES, NamedValue(ParameterName.Q, (Number.of(1.0),))
        )
    assert excinfo.value.error.number == "1002"


async def test_get_scale_temperatures_requires_arguments(started_client: IppDmeClient) -> None:
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(CommandName.GET_SCALE_TEMPERATURES)
    assert excinfo.value.error.number == "0502"


async def test_set_temperature_compensation_origin_accepts_xyz(
    started_client: IppDmeClient,
) -> None:
    await started_client.call(
        CommandName.SET_TEMPERATURE_COMPENSATION_ORIGIN,
        NamedValue(ParameterName.X, (Number.of(202.1),)),
        NamedValue(ParameterName.Y, (Number.of(201.0),)),
        NamedValue(ParameterName.Z, (Number.of(50.3),)),
    )


async def test_set_temperature_compensation_origin_rejects_non_xyz(
    started_client: IppDmeClient,
) -> None:
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(
            CommandName.SET_TEMPERATURE_COMPENSATION_ORIGIN,
            NamedValue(ParameterName.R, (Number.of(1.0),)),
        )
    assert excinfo.value.error.number == "1002"


async def test_set_coord_system_resets_temperature_compensation_origin(
    started_client: IppDmeClient,
) -> None:
    await started_client.call(
        CommandName.SET_TEMPERATURE_COMPENSATION_ORIGIN,
        NamedValue(ParameterName.X, (Number.of(202.1),)),
    )
    # Not directly observable, but must not error, and GetScaleTemperatures
    # (an unrelated store) must remain unaffected by this side effect.
    await started_client.call(CommandName.SET_COORD_SYSTEM, BasicName("PartCsy"))
    (data,) = await started_client.call(
        CommandName.GET_SCALE_TEMPERATURES, NamedValue(ParameterName.X, ())
    )
    (value,) = _items(data).values
    assert _num(value) == 20.0
