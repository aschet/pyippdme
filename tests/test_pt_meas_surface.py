# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for PtMeas's approach/search/retract behaviour (6.12.1) against a SampleSurface."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from pyippdme import IppDmeClient, VirtualCMM
from pyippdme.exceptions import IppDmeServerError
from pyippdme.protocol.ast import DataPayload, Items, NamedValue, Number
from pyippdme.protocol.commands import CommandName
from pyippdme.server.surface import SampleSurface
from pyippdme.simulation.surface import PlaneSurface


def _items(data: DataPayload) -> Items:
    assert isinstance(data, Items)
    return data


def _axes(data: DataPayload) -> dict[str, float]:
    values: dict[str, float] = {}
    for nv in _items(data).values:
        (arg,) = nv.args
        assert isinstance(arg, Number)
        values[nv.name] = arg.value
    return values


async def _connect_to_a_server_with(
    surface: SampleSurface | None,
) -> tuple[IppDmeClient, VirtualCMM]:
    srv = VirtualCMM(sample_surface=surface)
    port = await srv.start("127.0.0.1", 0)
    client = await IppDmeClient.connect("127.0.0.1", port)
    await client.start_session()
    return client, srv


@pytest.fixture
async def plane_client() -> AsyncIterator[IppDmeClient]:
    # A flat part surface at z=0, facing +Z.
    client, srv = await _connect_to_a_server_with(
        PlaneSurface(point=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0))
    )
    try:
        yield client
    finally:
        await client.close()
        await srv.close()


@pytest.fixture
async def no_surface_client() -> AsyncIterator[IppDmeClient]:
    client, srv = await _connect_to_a_server_with(None)
    try:
        yield client
    finally:
        await client.close()
        await srv.close()


async def test_pt_meas_without_a_surface_reports_the_commanded_position_exactly(
    no_surface_client: IppDmeClient,
) -> None:
    (data,) = await no_surface_client.call(
        CommandName.PT_MEAS,
        NamedValue("Z", (Number.of(0.0),)),
        NamedValue("IJK", (Number.of(0.0), Number.of(0.0), Number.of(1.0))),
    )
    axes = _axes(data)
    assert axes == pytest.approx({"X": 0.0, "Y": 0.0, "Z": 0.0})


async def test_pt_meas_with_ijk_reports_the_contact_point_on_the_surface(
    plane_client: IppDmeClient,
) -> None:
    # Nominal touch point (0, 0, 0), retreating/approaching along +Z (the
    # plane's own normal) - see cartcmm_class's module docstring for why
    # IJK points *away* from the part, not toward it.
    (data,) = await plane_client.call(
        CommandName.PT_MEAS,
        NamedValue("Z", (Number.of(0.0),)),
        NamedValue("IJK", (Number.of(0.0), Number.of(0.0), Number.of(1.0))),
    )
    axes = _axes(data)
    assert axes == pytest.approx({"X": 0.0, "Y": 0.0, "Z": 0.0}, abs=1e-6)


async def test_pt_meas_diverges_from_the_commanded_point_when_the_surface_does(
    plane_client: IppDmeClient,
) -> None:
    # Command a nominal point 1mm above the real surface (z=0); the reported
    # measurement should land on the actual surface, not the commanded one.
    (data,) = await plane_client.call(
        CommandName.PT_MEAS,
        NamedValue("Z", (Number.of(1.0),)),
        NamedValue("IJK", (Number.of(0.0), Number.of(0.0), Number.of(1.0))),
    )
    axes = _axes(data)
    assert axes["Z"] == pytest.approx(0.0, abs=1e-6)


async def test_pt_meas_resting_position_reflects_retract(plane_client: IppDmeClient) -> None:
    await plane_client.call(
        CommandName.PT_MEAS,
        NamedValue("Z", (Number.of(0.0),)),
        NamedValue("IJK", (Number.of(0.0), Number.of(0.0), Number.of(1.0))),
    )
    # Default Tool.PtMeasPar.Retract() is 2.0mm: the machine's resting
    # position afterwards is contact + IJK * retract, not the contact point
    # itself.
    (data,) = await plane_client.call(CommandName.GET, NamedValue("Z", ()))
    axes = _axes(data)
    assert axes["Z"] == pytest.approx(2.0, abs=1e-6)


async def test_pt_meas_infers_direction_from_the_previous_position_without_ijk(
    plane_client: IppDmeClient,
) -> None:
    # No IJK given: 6.12.1 falls back to the direction from the nominal
    # point to wherever the machine was just before this PtMeas.
    await plane_client.call(CommandName.GO_TO, NamedValue("Z", (Number.of(5.0),)))
    (data,) = await plane_client.call(CommandName.PT_MEAS, NamedValue("Z", (Number.of(0.0),)))
    axes = _axes(data)
    assert axes["Z"] == pytest.approx(0.0, abs=1e-6)


async def test_pt_meas_reports_surface_not_found_when_out_of_search_range(
    plane_client: IppDmeClient,
) -> None:
    # The surface is at z=0, but the nominal point is far enough away (and
    # pointed the wrong way) that approach+search never reaches it.
    with pytest.raises(IppDmeServerError) as excinfo:
        await plane_client.call(
            CommandName.PT_MEAS,
            NamedValue("Z", (Number.of(100.0),)),
            NamedValue("IJK", (Number.of(0.0), Number.of(0.0), Number.of(1.0))),
        )
    assert excinfo.value.error.number == "1006"


async def test_pt_meas_degenerate_direction_falls_back_to_exact_position(
    plane_client: IppDmeClient,
) -> None:
    # First PtMeas of the session, no IJK, no prior GoTo: nominal == previous
    # position, so there's no direction to probe along at all.
    (data,) = await plane_client.call(CommandName.PT_MEAS, NamedValue("Z", (Number.of(0.0),)))
    axes = _axes(data)
    assert axes == pytest.approx({"X": 0.0, "Y": 0.0, "Z": 0.0})
