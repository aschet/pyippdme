# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""The tree of a ``ToolCollection`` (6.22, Figure 56).

A server keeps one flat list of tools. A tool collection is a tree over it: collections
(nodes) hold other collections and named references to entries of the tool list, level by
level (``Carbody123`` / ``EngineCompartment`` / ``X+Short``). An entry of the tool list can be
referenced from any number of collections, under any name, so the same qualified stylus can be
``X-Short`` for the front of one part and ``Y+Long`` for the engine compartment of another. After
``OpenToolCollection`` a client names tools by the entries of the opened collection.

Paths join node names with ``.`` as in the standard's examples (``PartXYZ.Rear``); ``/`` is
accepted when reading a path. The empty path is the root, whose children are level 0.
Pure Python, no simulation: use it from your own server.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Joins the node names of a path (the examples of Tables 125-127 use ``PartXYZ.Rear``).
SEPARATOR = "."
#: The kinds an entry is reported with by ``EnumToolCollection``.
KIND_COLLECTION = "Collection"
KIND_TOOL = "Tool"


@dataclass(frozen=True, slots=True)
class ToolRef:
    """A named reference to an entry of the tool list."""

    name: str
    tool: str


@dataclass(slots=True)
class CollectionNode:
    """A collection: holds collections and tool references."""

    name: str
    children: list[CollectionNode | ToolRef] = field(default_factory=list)

    def child(self, name: str) -> CollectionNode | ToolRef | None:
        return next((c for c in self.children if c.name == name), None)


def split_path(path: str) -> list[str]:
    """Split ``path`` into node names; ``""`` and ``"."`` are the root."""
    return [part for part in path.replace("/", SEPARATOR).split(SEPARATOR) if part]


def find_node(root: CollectionNode, path: str) -> CollectionNode | None:
    """Return the collection at ``path``, or ``None`` (a tool reference is no node)."""
    node = root
    for part in split_path(path):
        nxt = node.child(part)
        if not isinstance(nxt, CollectionNode):
            return None
        node = nxt
    return node


def kind_of(entry: CollectionNode | ToolRef) -> str:
    return KIND_COLLECTION if isinstance(entry, CollectionNode) else KIND_TOOL


def children_of(root: CollectionNode, path: str) -> list[tuple[str, str]] | None:
    """``(name, kind)`` of the direct children at ``path`` (``EnumToolCollection``), or ``None``."""
    node = find_node(root, path)
    return None if node is None else [(c.name, kind_of(c)) for c in node.children]


def descendants_of(root: CollectionNode, path: str) -> list[tuple[str, str]] | None:
    """``(relative path, kind)`` of everything below ``path`` (``EnumAllToolCollections``)."""
    node = find_node(root, path)
    if node is None:
        return None
    found: list[tuple[str, str]] = []

    def walk(current: CollectionNode, prefix: str) -> None:
        for child in current.children:
            full = f"{prefix}{SEPARATOR}{child.name}" if prefix else child.name
            found.append((full, kind_of(child)))
            if isinstance(child, CollectionNode):
                walk(child, full)

    walk(node, "")
    return found


def resolve_tool(root: CollectionNode, open_path: str | None, name: str) -> str | None:
    """Return the tool-list entry that ``name`` stands for in the opened collection, or ``None``."""
    if open_path is None:
        return None
    node = find_node(root, open_path)
    entry = node.child(name) if node is not None else None
    return entry.tool if isinstance(entry, ToolRef) else None


def add_reference(root: CollectionNode, path: str, name: str, tool: str) -> None:
    """Add ``tool`` as ``name`` in the collection at ``path``, creating nodes on the way."""
    node = root
    for part in split_path(path):
        nxt = node.child(part)
        if nxt is None:
            nxt = CollectionNode(part)
            node.children.append(nxt)
        if not isinstance(nxt, CollectionNode):
            raise ValueError(f"{part!r} in {path!r} is a tool, not a collection")
        node = nxt
    node.children = [c for c in node.children if c.name != name]
    node.children.append(ToolRef(name, tool))
