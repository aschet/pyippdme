# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for the Part class properties (6.24.2): temperature/expansion/approach/search/retract."""

from __future__ import annotations

import pytest

from pyippdme import IppDmeClient
from pyippdme.protocol.ast import Items, NamedValue, Number
from pyippdme.protocol.commands import CommandName


async def _get(client: IppDmeClient, name: str) -> float:
    (data,) = await client.call(CommandName.GET_PROP, NamedValue(name, ()))
    assert isinstance(data, Items)
    (value,) = data.values
    arg = value.args[0]
    assert isinstance(arg, Number)
    return arg.value


@pytest.mark.parametrize(
    ("name", "default"),
    [
        ("Part.Temperature", 20.0),
        ("Part.XpanCoefficient", 0.0),
        ("Part.Approach", 0.0),
        ("Part.Search", 0.0),
        ("Part.Retract", 0.0),
    ],
)
async def test_property_defaults(started_client: IppDmeClient, name: str, default: float) -> None:
    assert await _get(started_client, name) == default


@pytest.mark.parametrize(
    "name",
    ["Part.Temperature", "Part.XpanCoefficient", "Part.Approach", "Part.Search", "Part.Retract"],
)
async def test_set_then_get_round_trips(started_client: IppDmeClient, name: str) -> None:
    await started_client.call(CommandName.SET_PROP, NamedValue(name, (Number.of(12.5),)))
    assert await _get(started_client, name) == 12.5


async def test_unrelated_property_still_uses_generic_store(started_client: IppDmeClient) -> None:
    await started_client.call(
        CommandName.SET_PROP, NamedValue("XXSomeCustomThing", (Number.of(3),))
    )
    assert await _get(started_client, "XXSomeCustomThing") == 3.0
