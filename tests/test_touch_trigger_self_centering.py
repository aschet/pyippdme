# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for TouchTrigger_SelfCentering (6.14): PtMeasSelfCenter/PtMeasSelfCenterLocked."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from pyippdme import IppDmeClient, VirtualCMM
from pyippdme.exceptions import IppDmeServerError
from pyippdme.protocol.ast import DataPayload, Items, NamedValue, Number
from pyippdme.protocol.commands import CommandName
from pyippdme.simulation.surface import PlaneSurface


def _axes(data: DataPayload) -> dict[str, float]:
    assert isinstance(data, Items)
    values: dict[str, float] = {}
    for nv in data.values:
        (arg,) = nv.args
        assert isinstance(arg, Number)
        values[nv.name] = arg.value
    return values


async def test_pt_meas_self_center_reports_the_commanded_position_without_a_surface(
    started_client: IppDmeClient,
) -> None:
    (data,) = await started_client.call(
        CommandName.PT_MEAS_SELF_CENTER,
        NamedValue("X", (Number.of(1.0),)),
        NamedValue("Y", (Number.of(2.0),)),
        NamedValue("Z", (Number.of(3.0),)),
        NamedValue("IJK", (Number.of(0.0), Number.of(0.0), Number.of(1.0))),
    )
    assert _axes(data) == pytest.approx({"X": 1.0, "Y": 2.0, "Z": 3.0})


async def test_pt_meas_self_center_locked_accepts_orthogonal_ijk_and_lmn(
    started_client: IppDmeClient,
) -> None:
    (data,) = await started_client.call(
        CommandName.PT_MEAS_SELF_CENTER_LOCKED,
        NamedValue("X", (Number.of(0.0),)),
        NamedValue("Y", (Number.of(0.0),)),
        NamedValue("Z", (Number.of(0.0),)),
        NamedValue("IJK", (Number.of(0.0), Number.of(0.0), Number.of(1.0))),
        NamedValue("LMN", (Number.of(1.0), Number.of(0.0), Number.of(0.0))),
    )
    assert _axes(data) == pytest.approx({"X": 0.0, "Y": 0.0, "Z": 0.0})


async def test_pt_meas_self_center_locked_rejects_non_orthogonal_ijk_and_lmn(
    started_client: IppDmeClient,
) -> None:
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(
            CommandName.PT_MEAS_SELF_CENTER_LOCKED,
            NamedValue("X", (Number.of(0.0),)),
            NamedValue("IJK", (Number.of(0.0), Number.of(0.0), Number.of(1.0))),
            NamedValue("LMN", (Number.of(0.0), Number.of(0.0), Number.of(1.0))),
        )
    assert excinfo.value.error.number == "0502"


@pytest.fixture
async def plane_client() -> AsyncIterator[IppDmeClient]:
    srv = VirtualCMM(sample_surface=PlaneSurface(point=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0)))
    port = await srv.start("127.0.0.1", 0)
    client = await IppDmeClient.connect("127.0.0.1", port)
    await client.start_session()
    try:
        yield client
    finally:
        await client.close()
        await srv.close()


async def test_pt_meas_self_center_probes_a_configured_surface_like_pt_meas(
    plane_client: IppDmeClient,
) -> None:
    # Same real approach/search/retract PtMeas() itself does (see
    # cartcmm_class's module docstring): commanding a nominal point 1mm
    # above the real surface (z=0) still reports the actual surface
    # contact, not the commanded point - this isn't a stub that always
    # echoes the commanded position back.
    (data,) = await plane_client.call(
        CommandName.PT_MEAS_SELF_CENTER,
        NamedValue("Z", (Number.of(1.0),)),
        NamedValue("IJK", (Number.of(0.0), Number.of(0.0), Number.of(1.0))),
    )
    assert _axes(data)["Z"] == pytest.approx(0.0, abs=1e-6)
