# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for the ToolChanger class subset: EnumTools/ChangeTool/FindTool/FoundTool/SetTool."""

from __future__ import annotations

import pytest

from pyippdme import IppDmeClient
from pyippdme.exceptions import IppDmeServerError
from pyippdme.protocol.ast import (
    BasicName,
    Items,
    NamedValue,
    NameValue,
    Number,
    PropertyData,
    String,
)
from pyippdme.protocol.commands import CommandName


async def test_enum_tools_lists_the_catalog(started_client: IppDmeClient) -> None:
    data = await started_client.call(CommandName.ENUM_TOOLS)
    names = {item.value for item in data if isinstance(item, NameValue)}
    assert names == {"RefTool", "RefTool2", "AlignProbe"}


async def test_active_tool_defaults_to_ref_tool(started_client: IppDmeClient) -> None:
    (data,) = await started_client.call(CommandName.TOOL)
    assert isinstance(data, NameValue)
    assert data.value == "RefTool"


async def test_change_tool_switches_active_tool(started_client: IppDmeClient) -> None:
    await started_client.call(CommandName.CHANGE_TOOL, String("RefTool2"))
    (data,) = await started_client.call(CommandName.TOOL)
    assert isinstance(data, NameValue)
    assert data.value == "RefTool2"


async def test_change_tool_updates_tool_id(started_client: IppDmeClient) -> None:
    await started_client.call(CommandName.CHANGE_TOOL, String("RefTool2"))
    (data,) = await started_client.call(CommandName.GET_PROP, NamedValue("Tool.Id", ()))
    assert isinstance(data, Items)
    (value,) = data.values
    assert value.xml is not None
    assert "RefTool2" in value.xml.raw


async def test_change_tool_rejects_unknown_tool(started_client: IppDmeClient) -> None:
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(CommandName.CHANGE_TOOL, String("NoSuchTool"))
    assert excinfo.value.error.number == "1502"


async def test_set_tool_rejects_unknown_tool(started_client: IppDmeClient) -> None:
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(CommandName.SET_TOOL, String("NoSuchTool"))
    assert excinfo.value.error.number == "1502"


async def test_found_tool_defaults_to_no_tool(started_client: IppDmeClient) -> None:
    (data,) = await started_client.call(CommandName.FOUND_TOOL)
    assert isinstance(data, NameValue)
    assert data.value == "NoTool"


async def test_find_tool_then_found_tool(started_client: IppDmeClient) -> None:
    await started_client.call(CommandName.FIND_TOOL, String("RefTool2"))
    (data,) = await started_client.call(CommandName.FOUND_TOOL)
    assert isinstance(data, NameValue)
    assert data.value == "RefTool2"


async def test_find_tool_unknown_sets_undef_tool(started_client: IppDmeClient) -> None:
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(CommandName.FIND_TOOL, String("NoSuchTool"))
    assert excinfo.value.error.number == "1502"
    await started_client.call(CommandName.CLEAR_ALL_ERRORS)

    (data,) = await started_client.call(CommandName.FOUND_TOOL)
    assert isinstance(data, NameValue)
    assert data.value == "UnDefTool"


async def test_get_change_tool_action_reports_move_auto_for_a_far_tool(
    started_client: IppDmeClient,
) -> None:
    # RefTool -> RefTool2 is offset by (10, 0, 50), ~51mm - past the
    # in-place "Switch" threshold, so a rack fetch ("MoveAuto") is reported.
    (data,) = await started_client.call(CommandName.GET_CHANGE_TOOL_ACTION, String("RefTool2"))
    assert isinstance(data, Items)
    values = {nv.name: nv for nv in data.values}
    action = values["Argument"].args[0]
    assert isinstance(action, BasicName)
    assert action.value == "MoveAuto"
    x = values["X"].args[0]
    assert isinstance(x, Number)
    assert x.value == 10.0


async def test_get_change_tool_action_reports_switch_for_the_active_tool(
    started_client: IppDmeClient,
) -> None:
    (data,) = await started_client.call(CommandName.GET_CHANGE_TOOL_ACTION, String("RefTool"))
    assert isinstance(data, Items)
    values = {nv.name: nv for nv in data.values}
    action = values["Argument"].args[0]
    assert isinstance(action, BasicName)
    assert action.value == "Switch"


async def test_get_change_tool_action_rejects_unknown_tool(
    started_client: IppDmeClient,
) -> None:
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(CommandName.GET_CHANGE_TOOL_ACTION, String("NoSuchTool"))
    assert excinfo.value.error.number == "1502"


async def test_enum_tool_collection_lists_the_catalog(started_client: IppDmeClient) -> None:
    data = await started_client.call(CommandName.ENUM_TOOL_COLLECTION, String("Tools"))
    pairs = {
        (item.first.value, item.second.value) for item in data if isinstance(item, PropertyData)
    }
    assert pairs == {("RefTool", "Tool"), ("RefTool2", "Tool"), ("AlignProbe", "Tool")}


async def test_enum_tool_collection_unknown_node_errors(started_client: IppDmeClient) -> None:
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(CommandName.ENUM_TOOL_COLLECTION, String("NoSuchNode"))
    assert excinfo.value.error.number == "1504"


async def test_enum_all_tool_collections_matches_enum_tool_collection(
    started_client: IppDmeClient,
) -> None:
    flat = await started_client.call(CommandName.ENUM_TOOL_COLLECTION, String("Tools"))
    recursive = await started_client.call(CommandName.ENUM_ALL_TOOL_COLLECTIONS, String("Tools"))
    flat_pairs = {
        (item.first.value, item.second.value) for item in flat if isinstance(item, PropertyData)
    }
    recursive_pairs = {
        (item.first.value, item.second.value)
        for item in recursive
        if isinstance(item, PropertyData)
    }
    assert (
        flat_pairs
        == recursive_pairs
        == {("RefTool", "Tool"), ("RefTool2", "Tool"), ("AlignProbe", "Tool")}
    )


async def test_enum_all_tool_collections_unknown_node_errors(started_client: IppDmeClient) -> None:
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(CommandName.ENUM_ALL_TOOL_COLLECTIONS, String("NoSuchNode"))
    assert excinfo.value.error.number == "1504"


async def test_open_tool_collection_succeeds_for_the_root(started_client: IppDmeClient) -> None:
    await started_client.call(CommandName.OPEN_TOOL_COLLECTION, String("Tools"))  # must not raise


async def test_open_tool_collection_unknown_node_errors(started_client: IppDmeClient) -> None:
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(CommandName.OPEN_TOOL_COLLECTION, String("NoSuchNode"))
    assert excinfo.value.error.number == "1504"
