# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for CartCMM's temperature-sensor commands (6.5.2)."""

from __future__ import annotations

from pyippdme import IppDmeClient
from pyippdme.protocol.ast import (
    Argument,
    BasicName,
    Items,
    NamedValue,
    Number,
    NumericData,
    String,
)
from pyippdme.protocol.commands import CommandName
from pyippdme.protocol.parameters import ParameterName


def _fields(item: object) -> dict[str, Argument]:
    assert isinstance(item, Items)
    return {nv.name: nv.args[0] for nv in item.values}


def _string(fields: dict[str, Argument], name: str) -> str:
    value = fields[name]
    assert isinstance(value, String)
    return value.value


def _number(fields: dict[str, Argument], name: str) -> float:
    value = fields[name]
    assert isinstance(value, Number)
    return value.value


async def test_get_temperature_sensors_lists_all_five(started_client: IppDmeClient) -> None:
    data = await started_client.call(CommandName.GET_TEMPERATURE_SENSORS)
    assert len(data) == 5
    fields = [_fields(item) for item in data]
    names = {_string(f, "Name") for f in fields}
    assert names == {"PartSensor", "CMMSensor", "XAxisSensor", "YAxisSensor", "ZAxisSensor"}

    part = next(f for f in fields if _string(f, "Name") == "PartSensor")
    assert part["Kind"] == BasicName("Part")
    assert "Scale" not in part

    x_axis = next(f for f in fields if _string(f, "Name") == "XAxisSensor")
    assert x_axis["Kind"] == BasicName("Mover")
    assert _string(x_axis, "Scale") == "X"


async def test_read_temperature_sensor_defaults_to_room_temperature(
    started_client: IppDmeClient,
) -> None:
    (data,) = await started_client.call(CommandName.READ_TEMPERATURE_SENSOR, String("CMMSensor"))
    assert isinstance(data, NumericData)
    (value,) = data.values
    assert value.value == 20.0


async def test_read_temperature_sensor_unknown_name_reports_disconnected(
    started_client: IppDmeClient,
) -> None:
    (data,) = await started_client.call(CommandName.READ_TEMPERATURE_SENSOR, String("NoSuchSensor"))
    assert isinstance(data, NumericData)
    (value,) = data.values
    assert value.value == -273.0


async def test_part_sensor_reflects_part_temperature_property(
    started_client: IppDmeClient,
) -> None:
    await started_client.call(
        CommandName.SET_PROP, NamedValue("Part.Temperature", (Number.of(35.5),))
    )
    (data,) = await started_client.call(CommandName.READ_TEMPERATURE_SENSOR, String("PartSensor"))
    assert isinstance(data, NumericData)
    (value,) = data.values
    assert value.value == 35.5


async def test_mover_sensor_reflects_scale_temperature(started_client: IppDmeClient) -> None:
    await started_client.call(
        CommandName.SET_SCALE_TEMPERATURES, NamedValue(ParameterName.X, (Number.of(21.5),))
    )
    (data,) = await started_client.call(CommandName.READ_TEMPERATURE_SENSOR, String("XAxisSensor"))
    assert isinstance(data, NumericData)
    (value,) = data.values
    assert value.value == 21.5


async def test_read_all_temperatures_reports_every_sensor(started_client: IppDmeClient) -> None:
    data = await started_client.call(CommandName.READ_ALL_TEMPERATURES)
    assert len(data) == 5
    fields = [_fields(item) for item in data]
    readings = {_string(f, "Name"): _number(f, "Temperature") for f in fields}
    assert readings == {
        "PartSensor": 20.0,
        "CMMSensor": 20.0,
        "XAxisSensor": 20.0,
        "YAxisSensor": 20.0,
        "ZAxisSensor": 20.0,
    }
