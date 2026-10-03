# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Tool collections (6.22.1.1, Figure 56): a tree over the tool list, paths like PartXYZ.Rear."""

from __future__ import annotations

import pytest

from pyippdme import IppDmeClient
from pyippdme.exceptions import IppDmeServerError
from pyippdme.protocol.commands import CommandName
from pyippdme.simulation.classes.tool_class import register_collection_entry
from pyippdme.types.toolcollection import (
    CollectionNode,
    add_reference,
    children_of,
    descendants_of,
    find_node,
    resolve_tool,
)


def _tree() -> CollectionNode:
    root = CollectionNode("")
    add_reference(root, "PartXYZ.Rear", "X-Short", "RefTool")
    add_reference(root, "PartXYZ.Rear", "Probe2", "RefTool2")
    add_reference(root, "Rackset1", "Main", "RefTool")
    return root


def test_children_and_descendants_use_dotted_paths() -> None:
    root = _tree()
    assert children_of(root, "") == [("PartXYZ", "Collection"), ("Rackset1", "Collection")]
    assert children_of(root, "PartXYZ") == [("Rear", "Collection")]
    assert children_of(root, "PartXYZ.Rear") == [("X-Short", "Tool"), ("Probe2", "Tool")]
    assert children_of(root, "PartXYZ/Rear") == children_of(root, "PartXYZ.Rear")  # lenient
    assert children_of(root, "Missing") is None
    assert descendants_of(root, "PartXYZ") == [
        ("Rear", "Collection"),
        ("Rear.X-Short", "Tool"),
        ("Rear.Probe2", "Tool"),
    ]
    assert find_node(root, "PartXYZ.Rear.X-Short") is None  # a tool is no node


def test_names_resolve_in_the_opened_collection() -> None:
    root = _tree()
    assert resolve_tool(root, "PartXYZ.Rear", "X-Short") == "RefTool"
    assert resolve_tool(root, "Rackset1", "X-Short") is None
    assert resolve_tool(root, None, "X-Short") is None
    with pytest.raises(ValueError, match="is a tool"):
        add_reference(root, "PartXYZ.Rear.X-Short", "Y", "RefTool")


async def test_collections_over_the_protocol(started_client: IppDmeClient) -> None:
    register_collection_entry("PartXYZ.Rear", "X-Short", "RefTool")
    register_collection_entry("PartXYZ.Rear", "Long", "RefTool2")
    client = started_client
    names = [
        p.to_wire() for p in await client.call(CommandName.ENUM_TOOL_COLLECTION, _s("PartXYZ"))
    ]
    assert names == ['"Rear","Collection"']
    all_below = [
        p.to_wire() for p in await client.call(CommandName.ENUM_ALL_TOOL_COLLECTIONS, _s("PartXYZ"))
    ]
    assert '"Rear.X-Short","Tool"' in all_below
    with pytest.raises(IppDmeServerError, match="1504"):
        await client.call(CommandName.OPEN_TOOL_COLLECTION, _s("Nope"))
    await client.clear_all_errors()
    await client.call(CommandName.OPEN_TOOL_COLLECTION, _s("PartXYZ.Rear"))
    await client.call(CommandName.CHANGE_TOOL, _s("Long"))
    (name,) = await client.call(CommandName.GET_PROP, _prop("Tool.Name"))
    assert "RefTool2" in name.to_wire()


def _s(text: str):  # type: ignore[no-untyped-def]
    from pyippdme.protocol.ast import String

    return String(text)


def _prop(name: str):  # type: ignore[no-untyped-def]
    from pyippdme.protocol.ast import NamedValue

    return NamedValue(name, ())
