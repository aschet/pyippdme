# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for Mover's OnMoveReport/OnMoveReportE daemon (VDMA 8722 6.7.1 Table 68)."""

from __future__ import annotations

import asyncio

import pytest

from pyippdme import IppDmeClient
from pyippdme.exceptions import IppDmeServerError
from pyippdme.protocol.ast import DataPayload, EventTag, Items, NamedValue, Number, String
from pyippdme.protocol.commands import CommandName
from pyippdme.protocol.parameters import ParameterName


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


async def test_on_move_report_reports_after_go_to(started_client: IppDmeClient) -> None:
    txn = await started_client.start_daemon(
        CommandName.ON_MOVE_REPORT_E,
        NamedValue("Time", (Number.of(0.5),)),
        NamedValue("Dis", (Number.of(2.0),)),
        NamedValue("X", ()),
        NamedValue("Y", ()),
    )

    await started_client.call(CommandName.GO_TO, NamedValue(ParameterName.X, (Number.of(5.0),)))

    events = txn.events()
    report = await asyncio.wait_for(anext(events), timeout=1.0)
    assert _axes(report) == pytest.approx({"X": 5.0, "Y": 0.0})


async def test_on_move_report_reports_after_change_tool(started_client: IppDmeClient) -> None:
    txn = await started_client.start_daemon(
        CommandName.ON_MOVE_REPORT_E,
        NamedValue("Time", (Number.of(0.5),)),
        NamedValue("Dis", (Number.of(2.0),)),
        NamedValue("X", ()),
    )
    await started_client.call(CommandName.CHANGE_TOOL, String("RefTool2"))
    report = await asyncio.wait_for(anext(txn.events()), timeout=1.0)
    assert "X" in _axes(report)


async def test_stop_daemon_stops_further_reports(started_client: IppDmeClient) -> None:
    txn = await started_client.start_daemon(
        CommandName.ON_MOVE_REPORT_E,
        NamedValue("Time", (Number.of(0.5),)),
        NamedValue("Dis", (Number.of(2.0),)),
        NamedValue("X", ()),
    )
    assert isinstance(txn.tag, EventTag)
    await started_client.stop_daemon(txn.tag)

    await started_client.call(CommandName.GO_TO, NamedValue(ParameterName.X, (Number.of(1.0),)))
    # stop_daemon() closes the transaction's events stream outright (no more
    # reports will ever arrive), rather than leaving the caller to time out.
    with pytest.raises(StopAsyncIteration):
        await anext(txn.events())


async def test_starting_a_second_daemon_while_one_is_active_errors(
    started_client: IppDmeClient,
) -> None:
    await started_client.start_daemon(
        CommandName.ON_MOVE_REPORT_E,
        NamedValue("Time", (Number.of(0.5),)),
        NamedValue("Dis", (Number.of(2.0),)),
        NamedValue("X", ()),
    )
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.start_daemon(
            CommandName.ON_MOVE_REPORT_E,
            NamedValue("Time", (Number.of(0.5),)),
            NamedValue("Dis", (Number.of(2.0),)),
            NamedValue("Y", ()),
        )
    assert excinfo.value.error.number == "0515"


async def test_on_move_report_with_a_plain_tag_is_rejected(started_client: IppDmeClient) -> None:
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(
            CommandName.ON_MOVE_REPORT_E,
            NamedValue("Time", (Number.of(0.5),)),
            NamedValue("Dis", (Number.of(2.0),)),
            NamedValue("X", ()),
        )
    assert excinfo.value.error.number == "0502"


async def test_stop_daemon_with_the_matching_tag_succeeds(started_client: IppDmeClient) -> None:
    txn = await started_client.start_daemon(
        CommandName.ON_MOVE_REPORT_E,
        NamedValue("Time", (Number.of(0.5),)),
        NamedValue("Dis", (Number.of(2.0),)),
        NamedValue("X", ()),
    )
    assert isinstance(txn.tag, EventTag)
    await started_client.stop_daemon(txn.tag)  # must not raise


async def test_stop_daemon_with_a_mismatched_tag_errors(started_client: IppDmeClient) -> None:
    await started_client.start_daemon(
        CommandName.ON_MOVE_REPORT_E,
        NamedValue("Time", (Number.of(0.5),)),
        NamedValue("Dis", (Number.of(2.0),)),
        NamedValue("X", ()),
    )
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(CommandName.STOP_DAEMON, EventTag.of(9999))
    assert excinfo.value.error.number == "0513"
