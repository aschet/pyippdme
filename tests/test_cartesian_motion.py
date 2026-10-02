# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for Cartesian's Step/GoToOnCircle/GoToOnSpiral (6.8.1)."""

from __future__ import annotations

import pytest

from pyippdme import IppDmeClient
from pyippdme.exceptions import IppDmeServerError
from pyippdme.protocol.ast import Items, NamedValue, Number
from pyippdme.protocol.commands import CommandName
from pyippdme.protocol.parameters import ParameterName


def _position(data: object) -> tuple[float, float, float]:
    assert isinstance(data, Items)
    values = {nv.name: nv.args[0] for nv in data.values}
    x, y, z = values["X"], values["Y"], values["Z"]
    assert isinstance(x, Number)
    assert isinstance(y, Number)
    assert isinstance(z, Number)
    return x.value, y.value, z.value


async def _get_position(client: IppDmeClient) -> tuple[float, float, float]:
    (data,) = await client.call(
        CommandName.GET,
        NamedValue(ParameterName.X, ()),
        NamedValue(ParameterName.Y, ()),
        NamedValue(ParameterName.Z, ()),
    )
    return _position(data)


async def test_step_adds_to_current_position(started_client: IppDmeClient) -> None:
    await started_client.call(CommandName.GO_TO, NamedValue(ParameterName.X, (Number.of(10),)))
    await started_client.call(
        CommandName.STEP,
        NamedValue(ParameterName.X, (Number.of(1),)),
        NamedValue(ParameterName.Z, (Number.of(2),)),
    )
    assert await _get_position(started_client) == (11.0, 0.0, 2.0)


async def test_step_leaves_unmentioned_axes_untouched(started_client: IppDmeClient) -> None:
    await started_client.call(CommandName.GO_TO, NamedValue(ParameterName.Y, (Number.of(5),)))
    await started_client.call(CommandName.STEP, NamedValue(ParameterName.X, (Number.of(3),)))
    assert await _get_position(started_client) == (3.0, 5.0, 0.0)


async def test_step_rejects_tool_alignment(started_client: IppDmeClient) -> None:
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(
            CommandName.STEP,
            NamedValue("Tool.Alignment", (Number.of(0), Number.of(0), Number.of(1))),
        )
    assert excinfo.value.error.number == "0506"


async def test_go_to_on_circle_moves_to_target_on_plane(started_client: IppDmeClient) -> None:
    await started_client.call(
        CommandName.GO_TO,
        NamedValue(ParameterName.X, (Number.of(10),)),
        NamedValue(ParameterName.Y, (Number.of(0),)),
        NamedValue(ParameterName.Z, (Number.of(0),)),
    )
    await started_client.call(
        CommandName.GO_TO_ON_CIRCLE,
        NamedValue(ParameterName.CENTER, (Number.of(0), Number.of(0), Number.of(0))),
        NamedValue(ParameterName.IJK, (Number.of(0), Number.of(0), Number.of(1))),
        NamedValue(ParameterName.X, (Number.of(0),)),
        NamedValue(ParameterName.Y, (Number.of(10),)),
        NamedValue(ParameterName.Z, (Number.of(0),)),
    )
    assert await _get_position(started_client) == (0.0, 10.0, 0.0)


async def test_go_to_on_circle_rejects_off_plane_target(started_client: IppDmeClient) -> None:
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(
            CommandName.GO_TO_ON_CIRCLE,
            NamedValue(ParameterName.CENTER, (Number.of(0), Number.of(0), Number.of(0))),
            NamedValue(ParameterName.IJK, (Number.of(0), Number.of(0), Number.of(1))),
            NamedValue(ParameterName.X, (Number.of(0),)),
            NamedValue(ParameterName.Y, (Number.of(10),)),
            NamedValue(ParameterName.Z, (Number.of(5),)),
        )
    assert excinfo.value.error.number == "0502"


async def test_go_to_on_circle_rejects_off_plane_start(started_client: IppDmeClient) -> None:
    await started_client.call(
        CommandName.GO_TO,
        NamedValue(ParameterName.X, (Number.of(10),)),
        NamedValue(ParameterName.Y, (Number.of(0),)),
        NamedValue(ParameterName.Z, (Number.of(3),)),
    )
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(
            CommandName.GO_TO_ON_CIRCLE,
            NamedValue(ParameterName.CENTER, (Number.of(0), Number.of(0), Number.of(0))),
            NamedValue(ParameterName.IJK, (Number.of(0), Number.of(0), Number.of(1))),
            NamedValue(ParameterName.X, (Number.of(0),)),
            NamedValue(ParameterName.Y, (Number.of(10),)),
            NamedValue(ParameterName.Z, (Number.of(0),)),
        )
    assert excinfo.value.error.number == "0502"


async def test_go_to_on_spiral_allows_off_plane_target(started_client: IppDmeClient) -> None:
    await started_client.call(
        CommandName.GO_TO,
        NamedValue(ParameterName.X, (Number.of(10),)),
        NamedValue(ParameterName.Y, (Number.of(0),)),
        NamedValue(ParameterName.Z, (Number.of(0),)),
    )
    await started_client.call(
        CommandName.GO_TO_ON_SPIRAL,
        NamedValue(ParameterName.CENTER, (Number.of(0), Number.of(0), Number.of(0))),
        NamedValue(ParameterName.IJK, (Number.of(0), Number.of(0), Number.of(1))),
        NamedValue(ParameterName.X, (Number.of(0),)),
        NamedValue(ParameterName.Y, (Number.of(10),)),
        NamedValue(ParameterName.Z, (Number.of(5),)),
    )
    assert await _get_position(started_client) == (0.0, 10.0, 5.0)


async def test_go_to_on_circle_requires_all_arguments(started_client: IppDmeClient) -> None:
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(
            CommandName.GO_TO_ON_CIRCLE,
            NamedValue(ParameterName.CENTER, (Number.of(0), Number.of(0), Number.of(0))),
        )
    assert excinfo.value.error.number == "0509"
