# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""The ``ToolChanger`` command class (6.22).

Each CMM implements exactly one ``ToolChanger`` instance (6.22's own
wording), which installs/changes tools from
:data:`pyippdme.simulation.classes.tool_class.TOOL_CATALOG` - the same small, fixed
catalog :mod:`pyippdme.simulation.classes.tool_class` uses to answer ``Tool()``/
``Tool.Id()`` for whichever tool is currently active.

Scope note: this simulation has no real probe-changer rack or manual-change
dialog to drive - there's nowhere for a physical ``GoTo`` to actually go -
so ``ChangeTool`` never generates an implicit motion command.
``EnumToolCollection``/``EnumAllToolCollections`` (6.22.1.1) both answer
from the one flat :data:`~pyippdme.simulation.classes.tool_class.TOOL_CATALOG` -
with no real hierarchy to traverse, "immediate children" and "all
(grand-)children" are the same query here - and ``OpenToolCollection`` is a
no-op, since every catalog tool is already referenced directly by its bare
name. All three raise ``1504 Collection not found`` for any ``NodeName``
other than :data:`~pyippdme.simulation.classes.tool_class.DEFAULT_TOOL_COLLECTION`.
``GetChangeToolAction`` is still made plausible without any of that:
:func:`_classify_change_tool_action`
reports ``Switch`` (an in-place, no-rack swap, e.g. adjacent styli on one
probe head) for a short offset and ``MoveAuto`` (an automatic rack fetch)
for a longer one, instead of always ``Switch`` - a guessed, clearly-labelled
threshold (no vendor publishes one), not a measured spec. No simulated
timing is added anywhere in this module for now (deliberately, alongside
``CartCMM``'s ``GoTo``/``PtMeas`` staying instant too) - only the reported
*values* change, not how long anything takes.
"""

from __future__ import annotations

from pyippdme.protocol.ast import Argument, String
from pyippdme.protocol.commands import CommandName
from pyippdme.protocol.errors import ErrorCode, ErrorSeverity, ServerError
from pyippdme.protocol.signature import DataType, Parameter
from pyippdme.server import builders
from pyippdme.server._util import bad_argument
from pyippdme.server.registry import CommandRegistry, HandlerResult
from pyippdme.simulation.classes.mover_class import report_move
from pyippdme.simulation.classes.tool_class import DEFAULT_TOOL_COLLECTION, TOOL_CATALOG
from pyippdme.simulation.context import Ctx
from pyippdme.types.vec3 import Vec3, norm, sub

_TOOL_NAME_PARAMS = (Parameter("ToolName", DataType.STRING, positional=True),)
_NODE_NAME_PARAMS = (Parameter("NodeName", DataType.STRING, positional=True),)
#: Offsets at or below this are treated as one probe head's own styli - an
#: in-place kinematic swap, no rack fetch needed (a guessed threshold, see
#: the module docstring).
_SWITCH_MAX_DISTANCE_MM = 25.0


def _classify_change_tool_action(offset: Vec3) -> str:
    return "Switch" if norm(offset) <= _SWITCH_MAX_DISTANCE_MM else "MoveAuto"


def _require_tool_name(args: tuple[Argument, ...], cause: str) -> str:
    if len(args) != 1 or not isinstance(args[0], String):
        raise bad_argument(cause, f"Expected {cause}(<ToolName>)")
    return args[0].value


def _require_node_name(args: tuple[Argument, ...], cause: str) -> str:
    if len(args) != 1 or not isinstance(args[0], String):
        raise bad_argument(cause, f"Expected {cause}(<NodeName>)")
    return args[0].value


def _require_root_collection(node_name: str, cause: str) -> None:
    if node_name != DEFAULT_TOOL_COLLECTION:
        raise ServerError(
            ErrorSeverity.ERROR, ErrorCode.COLLECTION_NOT_FOUND, cause, "Collection not found"
        )


async def _enum_tools(_ctx: Ctx, _args: tuple[Argument, ...]) -> HandlerResult:
    return builders.name_list(TOOL_CATALOG)


async def _enum_tool_collection(_ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
    node_name = _require_node_name(args, CommandName.ENUM_TOOL_COLLECTION)
    _require_root_collection(node_name, CommandName.ENUM_TOOL_COLLECTION)
    return [builders.property_entry(name, "Tool") for name in TOOL_CATALOG]


async def _enum_all_tool_collections(ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
    # No nested collections to recurse into - the catalog is flat, so this
    # answers exactly like EnumToolCollection for the one root that exists.
    node_name = _require_node_name(args, CommandName.ENUM_ALL_TOOL_COLLECTIONS)
    _require_root_collection(node_name, CommandName.ENUM_ALL_TOOL_COLLECTIONS)
    return await _enum_tool_collection(ctx, args)


async def _open_tool_collection(_ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
    node_name = _require_node_name(args, CommandName.OPEN_TOOL_COLLECTION)
    _require_root_collection(node_name, CommandName.OPEN_TOOL_COLLECTION)
    # Every catalog tool is already referenced directly by its bare name -
    # the catalog only ever had the one (root) collection to "open".
    return None


async def _change_tool(ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
    name = _require_tool_name(args, CommandName.CHANGE_TOOL)
    if name not in TOOL_CATALOG:
        raise ServerError(
            ErrorSeverity.ERROR, ErrorCode.TOOL_NOT_FOUND, CommandName.CHANGE_TOOL, "Tool not found"
        )
    ctx.state.tool.active_name = name
    # Mover 6.7.1's OnMoveReport() daemon, if any - its own "Server Remarks"
    # explicitly include ChangeTool() as a "virtual movement" trigger.
    await report_move(ctx)
    return None


async def _find_tool(ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
    name = _require_tool_name(args, CommandName.FIND_TOOL)
    if name not in TOOL_CATALOG:
        ctx.state.tool.found_name = "UnDefTool"
        raise ServerError(
            ErrorSeverity.ERROR, ErrorCode.TOOL_NOT_FOUND, CommandName.FIND_TOOL, "Tool not found"
        )
    ctx.state.tool.found_name = name
    return None


async def _found_tool(ctx: Ctx, _args: tuple[Argument, ...]) -> HandlerResult:
    return builders.name_value(ctx.state.tool.found_name or "NoTool")


async def _set_tool(ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
    name = _require_tool_name(args, CommandName.SET_TOOL)
    if name not in TOOL_CATALOG:
        raise ServerError(
            ErrorSeverity.ERROR, ErrorCode.TOOL_NOT_FOUND, CommandName.SET_TOOL, "Tool not found"
        )
    ctx.state.tool.active_name = name
    return None


async def _get_change_tool_action(ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
    name = _require_tool_name(args, CommandName.GET_CHANGE_TOOL_ACTION)
    if name not in TOOL_CATALOG:
        raise ServerError(
            ErrorSeverity.ERROR,
            ErrorCode.TOOL_NOT_FOUND,
            CommandName.GET_CHANGE_TOOL_ACTION,
            "Tool not found",
        )
    _current_id, current_offset = TOOL_CATALOG[ctx.state.tool.active_name]
    _target_id, target_offset = TOOL_CATALOG[name]
    offset = sub(target_offset, current_offset)
    action = _classify_change_tool_action(offset)
    dx, dy, dz = offset
    return builders.get_change_tool_action(dx, dy, dz, action=action)


def register(registry: CommandRegistry) -> None:
    registry.register(CommandName.ENUM_TOOLS, _enum_tools, arguments=())
    registry.register(CommandName.CHANGE_TOOL, _change_tool, arguments=_TOOL_NAME_PARAMS)
    registry.register(CommandName.FIND_TOOL, _find_tool, arguments=_TOOL_NAME_PARAMS)
    registry.register(CommandName.FOUND_TOOL, _found_tool, arguments=())
    registry.register(CommandName.SET_TOOL, _set_tool, arguments=_TOOL_NAME_PARAMS)
    registry.register(
        CommandName.GET_CHANGE_TOOL_ACTION, _get_change_tool_action, arguments=_TOOL_NAME_PARAMS
    )
    registry.register(
        CommandName.ENUM_TOOL_COLLECTION, _enum_tool_collection, arguments=_NODE_NAME_PARAMS
    )
    registry.register(
        CommandName.ENUM_ALL_TOOL_COLLECTIONS,
        _enum_all_tool_collections,
        arguments=_NODE_NAME_PARAMS,
    )
    registry.register(
        CommandName.OPEN_TOOL_COLLECTION, _open_tool_collection, arguments=_NODE_NAME_PARAMS
    )
