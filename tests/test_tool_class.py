# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for the Tool parameter blocks (GoToPar/PtMeasPar/ScanPar) and Tool.Id()."""

from __future__ import annotations

import pytest

from pyippdme import IppDmeClient
from pyippdme.exceptions import IppDmeServerError
from pyippdme.protocol.ast import DataPayload, Items, NamedValue, Number, PropertyData, String, Xml
from pyippdme.protocol.commands import CommandName
from pyippdme.types.tool_id import from_xml


def _items(data: DataPayload) -> Items:
    assert isinstance(data, Items)
    return data


def _num(nv: NamedValue) -> float:
    arg = nv.args[0]
    assert isinstance(arg, Number)
    return arg.value


async def test_get_default_act_value(started_client: IppDmeClient) -> None:
    (data,) = await started_client.call(CommandName.GET_PROP, NamedValue("Tool.GoToPar.Speed", ()))
    (value,) = _items(data).values
    assert value.name == "Tool.GoToPar.Speed"
    assert _num(value) == pytest.approx(100.0)


async def test_act_and_bare_spelling_agree(started_client: IppDmeClient) -> None:
    (bare,) = await started_client.call(
        CommandName.GET_PROP, NamedValue("Tool.PtMeasPar.Retract", ())
    )
    (act,) = await started_client.call(
        CommandName.GET_PROP, NamedValue("Tool.PtMeasPar.Retract.Act", ())
    )
    assert _num(_items(bare).values[0]) == 2.0
    assert _num(_items(act).values[0]) == 2.0


async def test_set_act_within_range(started_client: IppDmeClient) -> None:
    await started_client.call(
        CommandName.SET_PROP, NamedValue("Tool.PtMeasPar.Retract", (Number.of(-1.0),))
    )
    (data,) = await started_client.call(
        CommandName.GET_PROP, NamedValue("Tool.PtMeasPar.Retract.Act", ())
    )
    assert _num(_items(data).values[0]) == -1.0


async def test_set_act_out_of_range_clamps_and_warns(started_client: IppDmeClient) -> None:
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(
            CommandName.SET_PROP, NamedValue("Tool.PtMeasPar.Retract", (Number.of(1000.0),))
        )
    assert excinfo.value.error.number == "0504"
    assert not excinfo.value.error.severity.requires_clear_all_errors

    (data,) = await started_client.call(
        CommandName.GET_PROP, NamedValue("Tool.PtMeasPar.Retract.Act", ())
    )
    assert _num(_items(data).values[0]) == 50.0  # clamped to Max


async def test_min_max_def_are_read_only(started_client: IppDmeClient) -> None:
    (data,) = await started_client.call(
        CommandName.GET_PROP, NamedValue("Tool.PtMeasPar.Retract.Min", ())
    )
    assert _num(_items(data).values[0]) == -1.0

    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(
            CommandName.SET_PROP, NamedValue("Tool.PtMeasPar.Retract.Min", (Number.of(0.0),))
        )
    assert excinfo.value.error.number == "0510"


async def test_unrelated_property_falls_back_to_generic_store(
    started_client: IppDmeClient,
) -> None:
    await started_client.call(CommandName.SET_PROP, NamedValue("XXCustom.Thing", (Number.of(42),)))
    (data,) = await started_client.call(CommandName.GET_PROP, NamedValue("XXCustom.Thing", ()))
    assert _num(_items(data).values[0]) == 42


async def test_get_prop_e_uses_its_own_cause_on_unknown_property(
    started_client: IppDmeClient,
) -> None:
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(CommandName.GET_PROP_E, NamedValue("Never.Set", ()))
    assert excinfo.value.error.cause == "GetPropE"


async def test_tool_id_returns_parseable_xml(started_client: IppDmeClient) -> None:
    (data,) = await started_client.call(CommandName.GET_PROP, NamedValue("Tool.Id", ()))
    (value,) = _items(data).values
    assert value.name == "Tool.Id"
    assert isinstance(value.xml, Xml)
    tool = from_xml(value.xml.raw)
    assert tool.id == "RefTool"


async def test_tool_command_returns_active_tool_name(started_client: IppDmeClient) -> None:
    (data,) = await started_client.call(CommandName.TOOL)
    assert data.to_wire() == "RefTool"


async def test_is_alignable_is_false_for_fixed_tool(started_client: IppDmeClient) -> None:
    (data,) = await started_client.call(CommandName.IS_ALIGNABLE)
    assert data.to_wire() == "IsAlignable(0)"


async def test_tool_name_reports_the_active_tool(started_client: IppDmeClient) -> None:
    (data,) = await started_client.call(CommandName.GET_PROP, NamedValue("Tool.Name", ()))
    (value,) = _items(data).values
    arg = value.args[0]
    assert isinstance(arg, String)
    assert arg.value == "RefTool"


async def test_found_tool_name_reports_undef_tool_before_find_tool(
    started_client: IppDmeClient,
) -> None:
    (data,) = await started_client.call(CommandName.GET_PROP, NamedValue("FoundTool.Name", ()))
    (value,) = _items(data).values
    arg = value.args[0]
    assert isinstance(arg, String)
    assert arg.value == "UnDefTool"


async def test_found_tool_id_after_find_tool(started_client: IppDmeClient) -> None:
    await started_client.call(CommandName.FIND_TOOL, String("RefTool2"))
    (data,) = await started_client.call(CommandName.GET_PROP, NamedValue("FoundTool.Id", ()))
    (value,) = _items(data).values
    assert isinstance(value.xml, Xml)
    tool = from_xml(value.xml.raw)
    assert tool.id == "RefTool2"


async def test_found_tool_id_before_find_tool_is_undefined(
    started_client: IppDmeClient,
) -> None:
    # Table 72's own error table: "1503 Tool not defined" when there is no
    # real tool to report an Id for (found_name is None here - FindTool()
    # was never called this session).
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(CommandName.GET_PROP, NamedValue("FoundTool.Id", ()))
    assert excinfo.value.error.number == "1503"


async def test_tool_collection_reports_a_fixed_root_path(started_client: IppDmeClient) -> None:
    (data,) = await started_client.call(CommandName.GET_PROP, NamedValue("Tool.Collection", ()))
    (value,) = _items(data).values
    arg = value.args[0]
    assert isinstance(arg, String)
    assert arg.value == "Tools"


async def test_tool_last_qualified_defaults_to_never(started_client: IppDmeClient) -> None:
    (data,) = await started_client.call(CommandName.GET_PROP, NamedValue("Tool.LastQualified", ()))
    (value,) = _items(data).values
    arg = value.args[0]
    assert isinstance(arg, String)
    assert arg.value == "00000000T000000Z"


async def test_re_qualify_updates_last_qualified(started_client: IppDmeClient) -> None:
    await started_client.call(CommandName.RE_QUALIFY)
    (data,) = await started_client.call(CommandName.GET_PROP, NamedValue("Tool.LastQualified", ()))
    (value,) = _items(data).values
    arg = value.args[0]
    assert isinstance(arg, String)
    assert arg.value != "00000000T000000Z"
    assert arg.value.endswith("Z")


async def test_re_qualify_is_per_tool(started_client: IppDmeClient) -> None:
    await started_client.call(CommandName.RE_QUALIFY)  # qualifies RefTool
    await started_client.call(CommandName.CHANGE_TOOL, String("RefTool2"))
    (data,) = await started_client.call(CommandName.GET_PROP, NamedValue("Tool.LastQualified", ()))
    (value,) = _items(data).values
    arg = value.args[0]
    assert isinstance(arg, String)
    assert arg.value == "00000000T000000Z"  # RefTool2 was never qualified


async def test_enum_prop_lists_the_new_tool_properties(started_client: IppDmeClient) -> None:
    data = await started_client.call(CommandName.ENUM_PROP, NamedValue("Tool", ()))
    names = {item.first.value for item in data if isinstance(item, PropertyData)}
    assert {"Name", "Collection", "LastQualified", "Id"} <= names
