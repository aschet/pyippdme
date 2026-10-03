# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""The events a server sends on its own (5.5.3) and the object model that reads them."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from pyippdme import IppDmeMachine, VirtualCMM
from pyippdme.client.events import (
    ClearancePoint,
    KeyPress,
    ManualPoint,
    PropertyChanged,
    ServerEvent,
    ToolChanged,
    ToolCollectionOpened,
    UnknownEvent,
    parse_event,
)
from pyippdme.protocol.ast import DataResponse, EventTag, Items, NamedValue, Number, String
from pyippdme.protocol.network import MemoryNetwork
from pyippdme.server import builders
from pyippdme.simulation.classes.tool_class import DEFAULT_TOOL_COLLECTION
from pyippdme.types.csy import InMemoryCsyStore


@pytest.fixture
async def virtual(network: MemoryNetwork) -> AsyncIterator[tuple[VirtualCMM, IppDmeMachine]]:
    server = VirtualCMM(network=network, csy_store=InMemoryCsyStore())
    port = await server.start()
    machine = await IppDmeMachine.connect("x", port, network=network)
    await machine.start_session()
    try:
        yield server, machine
    finally:
        await machine.close()
        await server.close()


async def _next(machine: IppDmeMachine) -> ServerEvent:
    return await asyncio.wait_for(anext(machine.server.events()), 1.0)


async def test_a_key_press_reaches_the_client(virtual: tuple[VirtualCMM, IppDmeMachine]) -> None:
    server, machine = virtual
    await machine.mover.enable_user()
    assert await server.key_press("F1") is True
    assert await _next(machine) == KeyPress("F1")


async def test_no_key_press_is_sent_while_the_user_is_disabled(
    virtual: tuple[VirtualCMM, IppDmeMachine],
) -> None:
    """5.5.3: the server sends it "if the user is enabled"."""
    server, machine = virtual
    await machine.mover.disable_user()
    assert await server.key_press("Done") is False
    assert await server.manual_point(1, 2, 3) is False


async def test_a_manual_point_has_the_fields_of_on_pt_meas_report(
    virtual: tuple[VirtualCMM, IppDmeMachine],
) -> None:
    server, machine = virtual
    await machine.mover.enable_user()
    await machine.cart_cmm.on_pt_meas_report("X", "Y", "Z", "IJK")
    assert await server.manual_point(1, 2, 3, ijk=(0, 0, 1)) is True
    event = await _next(machine)
    assert isinstance(event, ManualPoint)
    assert event.values.number("Y") == 2.0
    assert event.values.vector("IJK") == (0.0, 0.0, 1.0)
    assert await machine.cart_cmm.get_position() == (1.0, 2.0, 3.0)


async def test_a_clearance_point_reports_zero_vectors(
    virtual: tuple[VirtualCMM, IppDmeMachine],
) -> None:
    """5.5.3: "If vectors are defined they have to be set to IJK(0,0,0)"."""
    server, machine = virtual
    await machine.mover.enable_user()
    await machine.cart_cmm.on_pt_meas_report("X", "IJK")
    await server.clearance_point(5, 0, 0)
    event = await _next(machine)
    assert isinstance(event, ClearancePoint)
    assert event.values.vector("IJK") == (0.0, 0.0, 0.0)


async def test_a_tool_change_at_the_machine_is_reported(
    virtual: tuple[VirtualCMM, IppDmeMachine],
) -> None:
    server, machine = virtual
    assert await server.change_tool("RefTool2") is True
    assert await _next(machine) == ToolChanged("RefTool2")
    assert await machine.tool.get_name() == "RefTool2"
    assert await server.change_tool("NoSuchTool") is False


async def test_a_property_changed_at_the_machine_is_reported(
    virtual: tuple[VirtualCMM, IppDmeMachine],
) -> None:
    server, machine = virtual
    assert await server.set_property("Part.Temperature", 25.0) is True
    assert await _next(machine) == PropertyChanged("Part.Temperature", 25.0)
    assert await machine.part.get_temperature() == 25.0


async def test_an_opened_tool_collection_is_reported(
    virtual: tuple[VirtualCMM, IppDmeMachine],
) -> None:
    server, machine = virtual
    assert await server.open_tool_collection(DEFAULT_TOOL_COLLECTION) is True
    assert await _next(machine) == ToolCollectionOpened(DEFAULT_TOOL_COLLECTION)
    assert await server.open_tool_collection("NoSuchCollection") is False


async def test_events_are_not_sent_without_a_client(network: MemoryNetwork) -> None:
    server = VirtualCMM(network=network, csy_store=InMemoryCsyStore())
    await server.start()
    try:
        assert await server.send_event(builders.key_press("Done")) is False
        assert await server.key_press("Done") is False
        assert await server.change_tool("RefTool2") is False
        assert await server.set_property("Part.Temperature", 1) is False
        assert await server.open_tool_collection(DEFAULT_TOOL_COLLECTION) is False
    finally:
        await server.close()


def _response(*named: NamedValue) -> DataResponse:
    return DataResponse(EventTag.UNSOLICITED, Items(named))


def test_parse_event_reads_every_predefined_event() -> None:
    assert parse_event(_response(NamedValue("KeyPress", (String("F3"),)))) == KeyPress("F3")
    assert parse_event(_response(NamedValue("ChangeTool", (String("T"),)))) == ToolChanged("T")
    assert parse_event(_response(NamedValue("OpenToolCollection", (String("C"),)))) == (
        ToolCollectionOpened("C")
    )
    point = parse_event(_response(NamedValue("GoTo", (NamedValue("X", (Number.of(1),)),))))
    assert isinstance(point, ClearancePoint)
    assert point.values.number("X") == 1.0


def test_parse_event_reads_set_prop_with_numbers_vectors_and_strings() -> None:
    one = NamedValue("SetProp", (NamedValue("A.B", (Number.of(2),)),))
    several = NamedValue("SetProp", (NamedValue("A.B", (Number.of(1), Number.of(2))),))
    text = NamedValue("SetProp", (NamedValue("A.B", (String("x"),)),))
    assert parse_event(_response(one)) == PropertyChanged("A.B", 2.0)
    assert parse_event(_response(several)) == PropertyChanged("A.B", (1.0, 2.0))
    assert parse_event(_response(text)) == PropertyChanged("A.B", "x")


def test_parse_event_keeps_unknown_events() -> None:
    unknown = _response(NamedValue("XXMyEvent", ()))
    assert parse_event(unknown) == UnknownEvent(unknown.data)
    two = _response(NamedValue("KeyPress", (String("a"),)), NamedValue("KeyPress", (String("b"),)))
    assert isinstance(parse_event(two), UnknownEvent)
