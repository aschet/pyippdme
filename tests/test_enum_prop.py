# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for EnumProp/EnumAllProp (VDMA 8722 6.3.1.1)."""

from __future__ import annotations

import pytest

from pyippdme import IppDmeClient
from pyippdme.exceptions import IppDmeServerError
from pyippdme.protocol.ast import NamedValue, PropertyData
from pyippdme.protocol.commands import CommandName


def _pairs(data: object) -> list[tuple[str, str]]:
    result = []
    for item in data:  # type: ignore[attr-defined]
        assert isinstance(item, PropertyData)
        result.append((item.first.value, item.second.value))
    return result


async def test_enum_prop_tool_go_to_par(started_client: IppDmeClient) -> None:
    data = await started_client.call(CommandName.ENUM_PROP, NamedValue("Tool.GoToPar", ()))
    assert _pairs(data) == [("Speed", "Property"), ("Accel", "Property")]


async def test_enum_prop_tool_go_to_par_speed_lists_min_max_def_act(
    started_client: IppDmeClient,
) -> None:
    data = await started_client.call(CommandName.ENUM_PROP, NamedValue("Tool.GoToPar.Speed", ()))
    assert _pairs(data) == [
        ("Min", "Number"),
        ("Max", "Number"),
        ("Def", "Number"),
        ("Act", "Number"),
    ]


async def test_enum_prop_part(started_client: IppDmeClient) -> None:
    data = await started_client.call(CommandName.ENUM_PROP, NamedValue("Part", ()))
    assert _pairs(data) == [
        ("Temperature", "Number"),
        ("XpanCoefficient", "Number"),
        ("Approach", "Number"),
        ("Search", "Number"),
        ("Retract", "Number"),
    ]


async def test_enum_prop_unknown_reference(started_client: IppDmeClient) -> None:
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(CommandName.ENUM_PROP, NamedValue("Bogus", ()))
    assert excinfo.value.error.number == "0505"


async def test_enum_prop_rejects_non_reference_argument(started_client: IppDmeClient) -> None:
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(CommandName.ENUM_PROP)
    assert excinfo.value.error.number == "0509"


async def test_enum_all_prop_part_is_flat(started_client: IppDmeClient) -> None:
    # Part's properties are all leaves (Number), so EnumAllProp matches EnumProp.
    data = await started_client.call(CommandName.ENUM_ALL_PROP, NamedValue("Part", ()))
    assert _pairs(data) == [
        ("Temperature", "Number"),
        ("XpanCoefficient", "Number"),
        ("Approach", "Number"),
        ("Search", "Number"),
        ("Retract", "Number"),
    ]


async def test_enum_all_prop_tool_go_to_par_recurses(started_client: IppDmeClient) -> None:
    data = await started_client.call(CommandName.ENUM_ALL_PROP, NamedValue("Tool.GoToPar", ()))
    assert _pairs(data) == [
        ("Speed", "Property"),
        ("Min", "Number"),
        ("Max", "Number"),
        ("Def", "Number"),
        ("Act", "Number"),
        ("Accel", "Property"),
        ("Min", "Number"),
        ("Max", "Number"),
        ("Def", "Number"),
        ("Act", "Number"),
    ]


async def test_enum_all_prop_tool_covers_every_block(started_client: IppDmeClient) -> None:
    data = await started_client.call(CommandName.ENUM_ALL_PROP, NamedValue("Tool", ()))
    names = [name for name, _kind in _pairs(data)]
    assert names[0] == "Id"
    assert "GoToPar" in names
    assert "PtMeasPar" in names
    assert "ScanPar" in names
    # Every leaf parameter's Min/Max/Def/Act should appear too.
    assert names.count("Min") == 10  # 2 (GoToPar) + 5 (PtMeasPar) + 3 (ScanPar) parameters
