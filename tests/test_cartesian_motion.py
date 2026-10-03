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


async def test_enumerations_must_not_be_empty_or_repeat_an_item(
    started_client: IppDmeClient,
) -> None:
    """5.3.4 and Tables 59, 60, 76: the enumeration is mandatory and lists each item once."""
    from pyippdme.exceptions import IppDmeServerError

    for command in (CommandName.GO_TO, CommandName.STEP, CommandName.PT_MEAS):
        with pytest.raises(IppDmeServerError, match="0502"):
            await started_client.call(command)
        await started_client.clear_all_errors()
    with pytest.raises(IppDmeServerError, match="0502"):
        await started_client.call(CommandName.GO_TO, NamedValue("Sync", (Number.of(1),)))
    await started_client.clear_all_errors()
    twice = (NamedValue("X", (Number.of(1),)), NamedValue("X", (Number.of(2),)))
    with pytest.raises(IppDmeServerError, match="0509"):
        await started_client.call(CommandName.GO_TO, *twice)


async def test_a_zero_probing_vector_has_no_norm(started_client: IppDmeClient) -> None:
    """Table 31: ``IJK(0,0,0)`` cannot be normalized (1010)."""
    from pyippdme.exceptions import IppDmeServerError

    zero = NamedValue("IJK", (Number.of(0), Number.of(0), Number.of(0)))
    with pytest.raises(IppDmeServerError, match="1010"):
        await started_client.call(CommandName.PT_MEAS, NamedValue("X", (Number.of(1),)), zero)


def test_the_rotary_table_takes_the_shortest_way() -> None:
    from pyippdme.simulation.classes.cartcmm_class import shortest_rotary_end

    assert shortest_rotary_end(0.0, 350.0) == pytest.approx(-10.0)
    assert shortest_rotary_end(350.0, 20.0) == pytest.approx(380.0)
    assert shortest_rotary_end(10.0, 100.0) == pytest.approx(100.0)
    assert shortest_rotary_end(0.0, 180.0) == pytest.approx(180.0)  # the server decides: positive
    assert shortest_rotary_end(0.0, 360.0) == pytest.approx(0.0)
