# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""End-to-end tests driving a real IppDmeServer over a loopback TCP socket."""

from __future__ import annotations

import pytest

from pyippdme import IppDmeClient, VirtualCMM
from pyippdme.exceptions import IppDmeServerError
from pyippdme.protocol.ast import (
    DataPayload,
    Items,
    NamedValue,
    Number,
    NumericData,
    String,
    StringValue,
)
from pyippdme.protocol.commands import CommandName
from pyippdme.protocol.parameters import ParameterName


def _items(data: DataPayload) -> Items:
    assert isinstance(data, Items)
    return data


def _num(nv: NamedValue) -> float:
    arg = nv.args[0]
    assert isinstance(arg, Number)
    return arg.value


async def test_start_returns_the_bound_port_and_matches_the_port_property() -> None:
    server = VirtualCMM()
    assert server.port is None
    try:
        port = await server.start("127.0.0.1", 0)
        assert port != 0
        assert server.port == port

        client = await IppDmeClient.connect("127.0.0.1", port)
        try:
            await client.start_session()
            (version,) = await client.call(CommandName.GET_DME_VERSION)
            assert version is not None
        finally:
            await client.close()
    finally:
        await server.close()
    assert server.port is None


async def test_commands_rejected_before_start_session(client: IppDmeClient) -> None:
    with pytest.raises(IppDmeServerError) as excinfo:
        await client.call(CommandName.GET_DME_VERSION)
    assert excinfo.value.error.number == "0008"


async def test_start_and_end_session(client: IppDmeClient) -> None:
    await client.start_session()
    await client.end_session()


async def test_double_start_session_is_protocol_error(client: IppDmeClient) -> None:
    await client.start_session()
    with pytest.raises(IppDmeServerError) as excinfo:
        await client.call(CommandName.START_SESSION)
    assert excinfo.value.error.number == "0008"


async def test_get_dme_version(client: IppDmeClient) -> None:
    await client.start_session()
    (data,) = await client.call(CommandName.GET_DME_VERSION)
    (version,) = _items(data).values
    assert version.name == "GetDMEVersion"  # Table 22: Kind N*
    assert version.args == (String("2.5"),)


async def test_go_to_and_get_round_trip(client: IppDmeClient) -> None:
    await client.start_session()
    await client.call(
        CommandName.GO_TO,
        NamedValue(ParameterName.X, (Number.of(10),)),
        NamedValue(ParameterName.Y, (Number.of(20),)),
    )
    (data,) = await client.call(
        CommandName.GET,
        NamedValue(ParameterName.X, ()),
        NamedValue(ParameterName.Y, ()),
        NamedValue(ParameterName.Z, ()),
    )
    values = {nv.name: _num(nv) for nv in _items(data).values}
    assert values == {"X": 10.0, "Y": 20.0, "Z": 0.0}


async def test_go_to_preserves_unset_axes(client: IppDmeClient) -> None:
    await client.start_session()
    await client.call(CommandName.GO_TO, NamedValue(ParameterName.X, (Number.of(5),)))
    await client.call(CommandName.GO_TO, NamedValue(ParameterName.Y, (Number.of(7),)))
    (data,) = await client.call(
        CommandName.GET, NamedValue(ParameterName.X, ()), NamedValue(ParameterName.Y, ())
    )
    values = {nv.name: _num(nv) for nv in _items(data).values}
    assert values == {"X": 5.0, "Y": 7.0}


async def test_pt_meas_uses_default_report_format(client: IppDmeClient) -> None:
    await client.start_session()
    (data,) = await client.call(
        CommandName.PT_MEAS,
        NamedValue(ParameterName.X, (Number.of(1),)),
        NamedValue(ParameterName.Y, (Number.of(2),)),
    )
    assert [nv.name for nv in _items(data).values] == ["X", "Y", "Z"]


async def test_pt_meas_uses_configured_report_format(client: IppDmeClient) -> None:
    await client.start_session()
    await client.call(
        CommandName.ON_PT_MEAS_REPORT,
        NamedValue(ParameterName.X, ()),
        NamedValue(ParameterName.Y, ()),
    )
    (data,) = await client.call(CommandName.PT_MEAS, NamedValue(ParameterName.X, (Number.of(3),)))
    assert [nv.name for nv in _items(data).values] == ["X", "Y"]


async def test_set_prop_get_prop_round_trip(client: IppDmeClient) -> None:
    await client.start_session()
    await client.set_prop(NamedValue("Tool.GoToPar.Speed", (Number.of(5),)))
    (data,) = await client.get_prop("Tool.GoToPar.Speed")
    assert _items(data).values == (NamedValue("Tool.GoToPar.Speed", (Number.of(5),)),)


async def test_get_prop_unknown(client: IppDmeClient) -> None:
    await client.start_session()
    with pytest.raises(IppDmeServerError) as excinfo:
        await client.call(CommandName.GET_PROP, NamedValue("NoSuchProperty", ()))
    assert excinfo.value.error.number == "0505"


async def test_unsupported_command(client: IppDmeClient) -> None:
    await client.start_session()
    with pytest.raises(IppDmeServerError) as excinfo:
        await client.call("ThisCommandDoesNotExist")
    assert excinfo.value.error.number == "0501"


async def test_error_recovery_via_clear_all_errors(client: IppDmeClient) -> None:
    await client.start_session()
    with pytest.raises(IppDmeServerError):
        await client.call("ThisCommandDoesNotExist")
    with pytest.raises(IppDmeServerError) as excinfo:
        await client.call(CommandName.GET_DME_VERSION)
    assert excinfo.value.error.number == "0514"
    await client.clear_all_errors()
    await client.call(CommandName.GET_DME_VERSION)  # no longer raises


async def test_get_err_status(client: IppDmeClient) -> None:
    await client.start_session()
    (data,) = await client.call(CommandName.GET_ERR_STATUS_E)
    assert isinstance(data, NumericData)
    assert data.values[0].value == 0.0


async def test_home_and_is_homed(client: IppDmeClient) -> None:
    await client.start_session()
    (before,) = await client.call(CommandName.IS_HOMED)
    assert _num(_items(before).values[0]) == 0.0
    await client.call(CommandName.HOME)
    (after,) = await client.call(CommandName.IS_HOMED)
    assert _num(_items(after).values[0]) == 1.0


async def test_get_machine_class(client: IppDmeClient) -> None:
    await client.start_session()
    (data,) = await client.call(CommandName.GET_MACHINE_CLASS)
    assert isinstance(data, StringValue)


async def test_get_supported_commands_includes_registered_names(client: IppDmeClient) -> None:
    await client.start_session()
    results = await client.call(CommandName.GET_SUPPORTED_COMMANDS)
    names = set[str]()
    for r in results:
        assert isinstance(r, StringValue)
        names.add(r.value.value)
    assert {"StartSession", "GoTo", "PtMeas", "GetDMEVersion"} <= names


async def test_stop_daemon_without_daemons_errors(client: IppDmeClient) -> None:
    from pyippdme.protocol.ast import EventTag

    await client.start_session()
    with pytest.raises(IppDmeServerError) as excinfo:
        await client.call(CommandName.STOP_DAEMON, EventTag.of(1))
    assert excinfo.value.error.number == "0513"


async def test_set_coord_system_round_trip(client: IppDmeClient) -> None:
    from pyippdme.protocol.ast import BasicName, NameValue

    await client.start_session()
    await client.call(CommandName.SET_COORD_SYSTEM, BasicName("PartCsy"))
    (data,) = await client.call(CommandName.GET_COORD_SYSTEM)
    assert data == NameValue("PartCsy")


async def test_set_coord_system_rejects_unknown_name(client: IppDmeClient) -> None:
    from pyippdme.protocol.ast import BasicName

    await client.start_session()
    with pytest.raises(IppDmeServerError) as excinfo:
        await client.call(CommandName.SET_COORD_SYSTEM, BasicName("NotACsy"))
    assert excinfo.value.error.number == "0505"


async def test_set_prop_unknown_name_is_not_recognized(client: IppDmeClient) -> None:
    """5.6 / 6.3.1.1: a name that is neither a standard nor a proprietary property is 0505."""
    await client.start_session()
    with pytest.raises(IppDmeServerError) as excinfo:
        await client.set_prop(NamedValue("NoSuchProperty", (Number.of(1),)))
    assert excinfo.value.error.number == "0505"
