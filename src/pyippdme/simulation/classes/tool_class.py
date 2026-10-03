# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""A subset of the ``Tool`` command class (6.10).

Implements the ``Tool``/``IsAlignable``/``Id``/``Name``/``Collection``/
``LastQualified``/``ReQualify`` properties and commands (6.10.2-6.10.3)
against :data:`TOOL_CATALOG`, the small, fixed set of simulated tools this
library's ``CartCMM``/``TouchTrigger`` simulation and
:mod:`pyippdme.simulation.classes.toolchanger_class` share, and registers a
:meth:`~pyippdme.server.registry.CommandRegistry.register_property_resolver`
resolver (rather than overriding ``SetProp``/``GetProp`` outright, which
would clobber other classes' properties, e.g.
:mod:`pyippdme.simulation.classes.part_class`'s) to give the
``GoToPar``/``PtMeasPar``/``ScanPar`` parameter blocks (6.10.4; see
:mod:`pyippdme.server.tool` for the mechanism, :mod:`pyippdme.simulation.tool`
for this simulation's own default numbers) real Min/Max/Def/Act semantics -
including clamping and the ``0504`` warning - and a
:meth:`~pyippdme.server.registry.CommandRegistry.register_property_children`
resolver so ``Server``'s ``EnumProp``/``EnumAllProp`` (6.3.1.1) can walk
that same parameter-block tree. ``Tool.Collection()`` always reports a
single fixed root path (:data:`DEFAULT_TOOL_COLLECTION`) - the catalog is
flat, with no real ``ToolCollection`` hierarchy to report a more specific
path from (see :mod:`pyippdme.simulation.classes.toolchanger_class` for
``EnumToolCollection``/etc.).

``ER``/``Q`` (report-only properties, usable only as arguments of
``OnPtMeasReport``/``OnScanReport``) are handled in
:mod:`pyippdme.simulation.classes.cartcmm_class`/
:mod:`pyippdme.simulation.classes.scanning_class` instead, where the actual
report is built.

``Alignable_AB``/``Alignable_ABC`` (6.20/6.21) are implemented for one
catalog tool, :data:`_ALIGNABLE_TOOLS`' ``"AlignProbe"`` - every other
catalog tool stays fixed/non-alignable, unchanged. Per explicit direction,
this is deliberately a plausible-response simplification, not a real
kinematic simulation: ``AlignTool()`` simply stores whatever orientation
vector(s) it is given (normalized) as reached exactly, with no mechanism,
travel time, or "would rotate more than 180 degrees" check behind it -
``UseSmallestAngletoAlignTool()``'s flag is accordingly stored but inert
(see :func:`_use_smallest_angle_to_align_tool`). ``A()``/``B()``/``C()``
and ``CalcToolAngles()`` derive two (or three) angles from that stored
vector via one documented, simple convention (:func:`tool_angles`)
rather than any real tool-head geometry - there being no genuine multi-DOF
kinematic tool model to derive them from otherwise. The same orientation is
moved by ``AlignTool()``, by the ``Tool.A``/``Tool.B``/``Tool.Alignment``
arguments of ``GoTo``/``Step``/``PtMeas`` (6.8.1, 6.12.1; see
:func:`apply_tool_orientation`) and reported by ``Tool.Alignment()`` (6.20.2). The
catalog tool has the two rotation axes ``A`` and ``B``, so ``Tool.C`` is
"Argument not supported".

``NoTool`` is a catalog entry that stands for no tool being loaded: the
machine can move with it but not measure (6.10.1, :func:`require_measuring_tool`).
``AvrOffsets``, ``CollisionVolume`` and ``AlignmentVolume`` (6.20.2) are properties:
they are read with ``GetProp(Tool.AvrOffsets())`` and so on, not called.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

from pyippdme.protocol.ast import (
    Argument,
    BasicName,
    Items,
    NamedValue,
    Number,
    NumericData,
    String,
    Xml,
)
from pyippdme.protocol.commands import CommandName
from pyippdme.protocol.errors import ErrorCode, ErrorSeverity, ServerError
from pyippdme.protocol.signature import DataType, Parameter
from pyippdme.server import builders
from pyippdme.server._util import bad_argument
from pyippdme.server.registry import CommandHandler, CommandRegistry, HandlerResult, PropertyKind
from pyippdme.server.tool import PARAMETER_FIELDS, ParameterField, ToolParameter, ToolParameters
from pyippdme.simulation.context import Ctx, csy_context
from pyippdme.types.obb import OBB_TOKEN
from pyippdme.types.tool_id import (
    BASIC_FUNCTIONS,
    ContinuousAlignMode,
    FixedAlignMode,
    ToolId,
    ToolIdTactileTouchTrigger,
)
from pyippdme.types.tool_id import to_xml as tool_id_to_xml
from pyippdme.types.toolcollection import CollectionNode, ToolRef, add_reference, resolve_tool
from pyippdme.types.vec3 import Vec3, dot, normalize

#: What a catalog tool can do: move, measure single points and scan.
_MEASURING_FUNCTIONS = tuple(
    fn
    for fn in BASIC_FUNCTIONS
    if fn
    in (
        "GoTo",
        "PtMeas",
        "ScanOnLine",
        "ScanOnCircle",
        "ScanOnHelix",
        "ScanOnCurve",
        "PTMeasSelfCenter",
        "PTMeasSelfCenterLocked",
    )
)

#: The name of the pseudo tool that stands for "no tool loaded" (6.10.1).
NO_TOOL = "NoTool"
#: The pseudo tool ``FindTool`` yields when it found nothing (6.10.1, 6.22.1).
UNDEF_TOOL = "UnDefTool"


def _touch_trigger(tool_id: str) -> ToolIdTactileTouchTrigger:
    return ToolIdTactileTouchTrigger(
        id=tool_id,
        basic_functions=_MEASURING_FUNCTIONS,
        cnc_axes=("X", "Y", "Z"),
        supports_optimization_mode=False,
        align_mode=FixedAlignMode(),
        move_on_acquisition=False,
    )


#: The tools this simulation offers, keyed by name (6.22's
#: ``ToolChanger`` provisions tools out-of-band - there is no command to add
#: one - so this catalog is a fixed constant, not per-connection state).
#: Each value is ``(Tool.Id() description, offset from RefTool)``; the
#: offset is only used to answer ``GetChangeToolAction`` (6.22.1) with a
#: plausible ``Switch`` distance, it does not perturb ``CartCMM`` position
#: math (see :mod:`pyippdme.simulation.classes.cartcmm_class`'s own scope note).
TOOL_CATALOG: dict[str, tuple[ToolId, Vec3]] = {
    "RefTool": (_touch_trigger("RefTool"), (0.0, 0.0, 0.0)),
    "RefTool2": (_touch_trigger("RefTool2"), (10.0, 0.0, 50.0)),
    "AlignProbe": (
        ToolIdTactileTouchTrigger(
            id="AlignProbe",
            basic_functions=_MEASURING_FUNCTIONS,
            cnc_axes=("X", "Y", "Z", "A", "B"),
            supports_optimization_mode=False,
            align_mode=ContinuousAlignMode(aligncaa=True),
            move_on_acquisition=False,
        ),
        (5.0, 0.0, 30.0),
    ),
    NO_TOOL: (
        ToolIdTactileTouchTrigger(
            id=NO_TOOL,
            basic_functions=("GoTo",),
            cnc_axes=("X", "Y", "Z"),
            supports_optimization_mode=False,
            align_mode=FixedAlignMode(),
            move_on_acquisition=False,
        ),
        (0.0, 0.0, 0.0),
    ),
}

#: Catalog tools this simulation treats as alignable (``IsAlignable()`` (1);
#: see the module docstring for what "alignable" means here). Every other
#: :data:`TOOL_CATALOG` entry stays fixed/non-alignable.
_ALIGNABLE_TOOLS: set[str] = {"AlignProbe"}
_BUILTIN_TOOLS = frozenset(TOOL_CATALOG)


def register_tool(name: str, tool_id: ToolId, offset: Vec3, *, alignable: bool = False) -> None:
    """Add (or replace) a tool of :data:`TOOL_CATALOG` for every server in this process.

    ``offset`` is the tool's tip relative to ``RefTool``, used to answer
    ``GetChangeToolAction`` (6.22.1). A simulation with its own tools (see
    :mod:`pyippdme.twin`) registers them before it starts serving, and takes
    them out again with :func:`unregister_tool`. The catalog is process-wide, as
    :data:`TOOL_CATALOG` itself is.
    """
    TOOL_CATALOG[name] = (tool_id, offset)
    if alignable:
        _ALIGNABLE_TOOLS.add(name)
    else:
        _ALIGNABLE_TOOLS.discard(name)


#: References into the tool list that ``register_collection_entry`` added: a tree over
#: :data:`TOOL_CATALOG` (6.22, Figure 56). The flat collection ``DEFAULT_TOOL_COLLECTION``
#: always exists beside it and holds every tool under its own name.
_COLLECTIONS = CollectionNode("")


def collection_root() -> CollectionNode:
    """The root of all tool collections: the flat one of every tool, and the registered ones."""
    flat = CollectionNode(DEFAULT_TOOL_COLLECTION, [ToolRef(name, name) for name in TOOL_CATALOG])
    return CollectionNode("", [flat, *_COLLECTIONS.children])


def register_collection_entry(path: str, name: str, tool: str) -> None:
    """Reference the catalog tool ``tool`` as ``name`` in the collection at ``path``.

    Nodes on the way are created; the same tool can be referenced any number of times, under
    any name (Figure 56). Like :func:`register_tool` this is process-wide.
    """
    add_reference(_COLLECTIONS, path, name, tool)


def clear_collections() -> None:
    """Remove every collection that :func:`register_collection_entry` added."""
    _COLLECTIONS.children.clear()


def resolve_tool_name(ctx: Ctx, name: str) -> str | None:
    """The catalog tool that ``name`` means: an entry of the opened collection, else a tool name."""
    mapped = resolve_tool(collection_root(), ctx.state.tool.open_collection, name)
    if mapped is not None and mapped in TOOL_CATALOG:
        return mapped
    return name if name in TOOL_CATALOG else None


def reset_registered_tools() -> None:
    """Remove every tool and collection that was registered; only the built-in tools stay."""
    for name in [n for n in TOOL_CATALOG if n not in _BUILTIN_TOOLS]:
        unregister_tool(name)
    clear_collections()


def unregister_tool(name: str) -> None:
    """Remove a tool that :func:`register_tool` added; the built-in tools stay."""
    if name in _BUILTIN_TOOLS:
        raise ValueError(f"{name!r} is a built-in tool")
    TOOL_CATALOG.pop(name, None)
    _ALIGNABLE_TOOLS.discard(name)


#: A never-aligned tool's default orientation: pointing away from a surface
#: below it (6.20.1's own "away from the surface" convention for ``i1,j1,k1``),
#: no secondary vector set - a plausible "home" position, not a measured one.
_DEFAULT_ALIGNMENT: Vec3 = (0.0, 0.0, 1.0)

#: The single collection every :data:`TOOL_CATALOG` tool belongs to
#: (``Tool.Collection()``, 6.10.3) - shared with
#: :mod:`pyippdme.simulation.classes.toolchanger_class`'s
#: ``EnumToolCollection``/``EnumAllToolCollections``/``OpenToolCollection``
#: (6.22.1.1), since the catalog is flat and has no real hierarchy beyond
#: this one root.
DEFAULT_TOOL_COLLECTION = "Tools"


def _find_tool_parameter(
    tool_parameters: ToolParameters, name: str
) -> tuple[ToolParameter, ParameterField] | None:
    """Resolve ``Tool.<Block>.<Param>[.<Field>]`` to its :class:`ToolParameter` and field.

    Returns ``None`` if ``name`` does not address one of the modeled
    parameter blocks (in which case the caller should fall back to the
    generic property store), so unrelated properties like
    ``Tool.SomeCustomThing`` are unaffected.
    """
    parts = name.split(".")
    if len(parts) not in (3, 4) or parts[0] != CommandName.TOOL:
        return None
    field_text = parts[3] if len(parts) == 4 else ParameterField.ACT
    if field_text not in PARAMETER_FIELDS:
        return None
    block = tool_parameters.block(parts[1])
    if block is None:
        return None
    parameter = block.get(parts[2])
    if parameter is None:
        return None
    return parameter, ParameterField(field_text)


def _try_set_tool_param(ctx: Ctx, arg: NamedValue) -> bool:
    resolved = _find_tool_parameter(ctx.state.tool.parameters, arg.name)
    if resolved is None:
        return False
    parameter, field_name = resolved
    if field_name != ParameterField.ACT:
        raise ServerError(
            ErrorSeverity.CRITICAL,
            ErrorCode.BAD_PROPERTY,
            CommandName.SET_PROP,
            f"{arg.name} is read-only, only .Act() can be set (6.10.4)",
        )
    if len(arg.args) != 1 or not isinstance(arg.args[0], Number):
        raise ServerError(
            ErrorSeverity.CRITICAL,
            ErrorCode.BAD_ARGUMENT,
            CommandName.SET_PROP,
            f"Expected {arg.name}(<number>)",
        )
    if parameter.set_act(arg.args[0].value):
        raise ServerError(
            ErrorSeverity.WARNING,
            ErrorCode.ARGUMENT_OUT_OF_RANGE,
            CommandName.SET_PROP,
            f"{arg.name} clamped to [{parameter.minimum}, {parameter.maximum}]",
        )
    return True


#: What ``Tool``/``FoundTool`` resolve to: the active tool's name (always a
#: real :data:`TOOL_CATALOG` key), or whatever ``FindTool(...)`` last found
#: (a real key, ``"UnDefTool"`` if it failed, or unset if never called).
def _named_tool_name(ctx: Ctx, namespace: str) -> str | None:
    if namespace == CommandName.TOOL:
        return ctx.state.tool.active_name
    if namespace == CommandName.FOUND_TOOL:
        return ctx.state.tool.found_name
    return None


_NEVER_QUALIFIED = "00000000T000000Z"


def tool_alignment(ctx: Ctx, tool_name: str) -> tuple[Vec3, Vec3 | None]:
    """Look up the alignment ``AlignTool()`` last set for ``tool_name`` (6.20.1), or the default.

    Used by :mod:`pyippdme.simulation.classes.cartcmm_class`'s ``Get(A())``/
    ``Get(B())``/``Get(C())`` too - see the module docstring's
    simplification note on how those angles are derived.
    """
    return ctx.state.tool.alignment.get(tool_name, (_DEFAULT_ALIGNMENT, None))


def is_alignable_tool(tool_name: str) -> bool:
    """Whether the catalog tool ``tool_name`` can be aligned (``IsAlignable()``, 6.10.2)."""
    return tool_name in _ALIGNABLE_TOOLS


def _require_alignable(tool_name: str | None, cause: str) -> str:
    if tool_name is None or tool_name not in _ALIGNABLE_TOOLS:
        raise ServerError(
            ErrorSeverity.CRITICAL, ErrorCode.TOOL_NOT_ALIGNABLE, cause, "Tool not alignable"
        )
    return tool_name


def tool_angles(primary: Vec3, secondary: Vec3 | None) -> tuple[float, float, float]:
    """Derive plausible ``(A, B, C)`` angles (degrees) from alignment vectors.

    A documented, simple convention, not real tool-head kinematics (see the
    module docstring): ``A`` is the tilt of ``primary`` from the default
    "straight up" orientation (:data:`_DEFAULT_ALIGNMENT`), ``B`` is
    ``primary``'s azimuth around that axis, and ``C`` - only meaningful once
    a ``secondary`` vector has been given - is its own azimuth the same way.
    """
    i, j, _k = primary
    a = math.degrees(math.acos(max(-1.0, min(1.0, dot(primary, _DEFAULT_ALIGNMENT)))))
    b = math.degrees(math.atan2(j, i))
    c = math.degrees(math.atan2(secondary[1], secondary[0])) if secondary is not None else 0.0
    return a, b, c


def alignment_from_angles(a: float, b: float) -> Vec3:
    """Invert :func:`tool_angles` for a tool with the two rotation axes ``A`` and ``B``."""
    ra, rb = math.radians(a), math.radians(b)
    return (math.sin(ra) * math.cos(rb), math.sin(ra) * math.sin(rb), math.cos(ra))


def tool_axis_value(ctx: Ctx, name: str, cause: str) -> float:
    """Return the angle ``Tool.A``/``Tool.B``/``Tool.C`` (or the ``FoundTool`` one) names (6.20.2).

    Raises ``1503`` when ``FoundTool`` has not been set by ``FindTool`` yet.
    """
    namespace, _dot, axis = name.partition(".")
    tool_name = _named_tool_name(ctx, namespace)
    if tool_name is None or tool_name not in TOOL_CATALOG:
        raise ServerError(
            ErrorSeverity.CRITICAL, ErrorCode.TOOL_NOT_DEFINED, cause, "Tool not defined"
        )
    primary, secondary = tool_alignment(ctx, tool_name)
    a, b, c = tool_angles(primary, secondary)
    return {"A": a, "B": b, "C": c}[axis]


def tool_alignment_numbers(ctx: Ctx, namespace: str, cause: str) -> tuple[float, ...]:
    """Return ``Tool.Alignment`` as three numbers, or six once a secondary vector is set."""
    tool_name = _named_tool_name(ctx, namespace)
    if tool_name is None or tool_name not in TOOL_CATALOG:
        raise ServerError(
            ErrorSeverity.CRITICAL, ErrorCode.TOOL_NOT_DEFINED, cause, "Tool not defined"
        )
    primary, secondary = tool_alignment(ctx, tool_name)
    return (*primary, *(secondary or ()))


def apply_tool_orientation(
    ctx: Ctx,
    cause: str,
    *,
    a: float | None = None,
    b: float | None = None,
    c: float | None = None,
    alignment: tuple[float, ...] | None = None,
    relative: bool = False,
) -> None:
    """Move the active tool's rotation axes: ``Tool.A``/``Tool.B``/``Tool.Alignment`` (6.8.1, 6.20).

    ``relative`` adds the angles to the current ones (``Step``, 6.8.1). A tool
    that cannot be aligned raises ``1505``; ``Tool.C`` needs a tool with a
    third rotation axis (``Alignable_ABC``), which the catalog has none of.
    """
    if a is None and b is None and c is None and alignment is None:
        return
    tool_name = _require_alignable(ctx.state.tool.active_name, cause)
    if c is not None:
        raise ServerError(
            ErrorSeverity.CRITICAL,
            ErrorCode.ARGUMENT_NOT_SUPPORTED,
            cause,
            "Tool.C is not supported: the tool has only the rotation axes A and B",
        )
    primary, secondary = tool_alignment(ctx, tool_name)
    new_secondary: Vec3 | None
    if alignment is not None:
        if len(alignment) not in (3, 6):
            raise bad_argument(cause, "Tool.Alignment needs 3 or 6 numbers")
        try:
            new_primary = normalize((alignment[0], alignment[1], alignment[2]))
            new_secondary = (
                normalize((alignment[3], alignment[4], alignment[5]))
                if len(alignment) == 6
                else None
            )
        except ValueError as exc:
            raise bad_argument(cause, "Tool.Alignment vectors must be non-zero") from exc
    else:
        current_a, current_b, _current_c = tool_angles(primary, secondary)
        new_a = current_a if a is None else (current_a + a if relative else a)
        new_b = current_b if b is None else (current_b + b if relative else b)
        new_primary = alignment_from_angles(new_a, new_b)
        new_secondary = None
    ctx.state.tool.alignment[tool_name] = (new_primary, new_secondary)


#: The ``basicfunction`` (Annex G) a command needs from a tool, where the standard names one.
_COMMAND_FUNCTION: dict[str, str] = {
    CommandName.PT_MEAS: "PtMeas",
    CommandName.PT_MEAS_SELF_CENTER: "PTMeasSelfCenter",
    CommandName.PT_MEAS_SELF_CENTER_LOCKED: "PTMeasSelfCenterLocked",
    CommandName.SCAN_ON_LINE: "ScanOnLine",
    CommandName.SCAN_ON_CIRCLE: "ScanOnCircle",
    CommandName.SCAN_ON_HELIX: "ScanOnHelix",
    CommandName.SCAN_ON_CURVE: "ScanOnCurve",
    CommandName.SCAN_IN_PLANE_END_IS_SPHERE: "ScanOnLine",
    CommandName.SCAN_IN_PLANE_END_IS_PLANE: "ScanOnLine",
    CommandName.SCAN_IN_PLANE_END_IS_CYL: "ScanOnLine",
    CommandName.DATA_ACQUIRE: "DataAcquire",
}


def require_tool_function(ctx: Ctx, cause: str) -> None:
    """Raise ``2002`` if the active tool's ``basicfunction`` list lacks what ``cause`` needs.

    Only tools added with :func:`register_tool` are checked; the built-in tools
    accept every command, as the simulation always did.
    """
    name = ctx.state.tool.active_name
    function = _COMMAND_FUNCTION.get(cause)
    entry = TOOL_CATALOG.get(name)
    if name in _BUILTIN_TOOLS or function is None or entry is None:
        return
    if function not in entry[0].basic_functions:
        raise ServerError(
            ErrorSeverity.ERROR,
            ErrorCode.PROBE_TYPE_NOT_ALLOWED,
            cause,
            "Type of probe does not allow this operation",
        )


def require_measuring_tool(ctx: Ctx, cause: str) -> None:
    """Raise ``1503`` while no tool is loaded; with ``NoTool`` the machine only moves (6.10.1)."""
    if ctx.state.tool.active_name == NO_TOOL:
        raise ServerError(
            ErrorSeverity.CRITICAL, ErrorCode.TOOL_NOT_DEFINED, cause, "Tool not defined"
        )
    require_tool_function(ctx, cause)


#: ``Tool.<X>``/``FoundTool.<X>`` properties that ``GetProp`` answers (6.10.3, 6.20.2).
_NAMED_TOOL_PROPERTIES = (
    "Name",
    "Collection",
    "LastQualified",
    "Id",
    "Alignment",
    "AvrOffsets",
    "CollisionVolume",
    "AlignmentVolume",
)


def _try_get_named_tool_property(ctx: Ctx, arg: NamedValue) -> NamedValue | None:
    """Resolve ``Tool.<X>``/``FoundTool.<X>`` for :data:`_NAMED_TOOL_PROPERTIES`."""
    namespace, dot, leaf = arg.name.partition(".")
    if not dot or arg.args or leaf not in _NAMED_TOOL_PROPERTIES:
        return None
    maybe_tool_name = _named_tool_name(ctx, namespace)
    if leaf == "Name":
        # 6.10.3's own General Remarks: "can also be UnDefTool or NoTool" -
        # this always resolves, unlike the other four below, which need a
        # real catalog entry to answer from.
        return NamedValue(
            arg.name, (String(maybe_tool_name if maybe_tool_name is not None else UNDEF_TOOL),)
        )
    if maybe_tool_name is None or maybe_tool_name not in TOOL_CATALOG:
        if leaf == "Id":
            # Table 72's own error table: "1503 Tool not defined".
            raise ServerError(
                ErrorSeverity.CRITICAL, ErrorCode.TOOL_NOT_DEFINED, arg.name, "Tool not defined"
            )
        return None  # Collection/LastQualified/Alignment: e.g. FoundTool.* before FindTool()
    tool_name = maybe_tool_name
    if leaf == "Collection":
        return NamedValue(
            arg.name, (String(ctx.state.tool.open_collection or DEFAULT_TOOL_COLLECTION),)
        )
    if leaf == "LastQualified":
        when = ctx.state.tool.last_qualified.get(tool_name, _NEVER_QUALIFIED)
        return NamedValue(arg.name, (String(when),))
    if leaf == "Alignment":
        # Table 114: "can only be invoked as an argument of a
        # GetProp(Tool.Alignment())" - the setter form (Table 115) is not
        # modeled, see the module docstring.
        _require_alignable(tool_name, arg.name)
        primary, secondary = tool_alignment(ctx, tool_name)
        numbers = (*primary, *(secondary or ()))
        return NamedValue(arg.name, tuple(Number.of(c) for c in numbers))
    if leaf == "AvrOffsets":
        # "Relative to an arbitrary reference point which changes from server to server"
        # (Table 116): this simulation has no offset model, so zero.
        return NamedValue(arg.name, tuple(Number.of(0.0) for _ in range(3)))
    if leaf == "AlignmentVolume":
        # Figures 52/53: a sphere about the head's pivot that holds the tool in every alignment;
        # the centre is the vector from the tool's reference point, in the active CSY.
        volume = ctx.tool_handler.alignment_volume(tool_name) if ctx.tool_handler else None
        if volume is None:
            return NamedValue(arg.name, ())
        centre, radius = volume
        centre = csy_context(ctx).direction_to_client(centre)
        return NamedValue(arg.name, tuple(Number.of(v) for v in (*centre, radius)))
    if leaf == "CollisionVolume":
        # Figures 49-51: oriented bounding boxes that cover the tool, each as ``OBB`` and 15
        # numbers, in the active CSY (a rotated tool gets rotated boxes).
        boxes = ctx.tool_handler.collision_volume(tool_name) if ctx.tool_handler else None
        if boxes is None:
            return NamedValue(arg.name, ())
        context = csy_context(ctx)
        arguments: list[Argument] = []
        for box in boxes:
            arguments.append(BasicName(OBB_TOKEN))
            arguments.extend(
                Number.of(v) for v in box.mapped(context.direction_to_client).numbers()
            )
        return NamedValue(arg.name, tuple(arguments))
    # Only "Id" remains at this point.
    tool_id, _offset = TOOL_CATALOG[tool_name]
    return NamedValue(arg.name, (), Xml(tool_id_to_xml(tool_id)))


def _try_get_tool_param(ctx: Ctx, arg: NamedValue) -> NamedValue | None:
    named_tool_property = _try_get_named_tool_property(ctx, arg)
    if named_tool_property is not None:
        return named_tool_property
    resolved = _find_tool_parameter(ctx.state.tool.parameters, arg.name)
    if resolved is None:
        return None
    parameter, field_name = resolved
    value = {
        ParameterField.MIN: parameter.minimum,
        ParameterField.MAX: parameter.maximum,
        ParameterField.DEF: parameter.default,
        ParameterField.ACT: parameter.value,
    }[field_name]
    return NamedValue(arg.name, (Number.of(value),))


def _tool_property_children(ctx: Ctx, reference: str) -> tuple[tuple[str, str], ...] | None:
    """Answer ``EnumProp``/``EnumAllProp`` for ``Tool``/``Tool.<Block>``/``Tool.<Block>.<Param>``.

    Bounded to the object tree this simulation actually models (6.10.4's
    parameter blocks); a reference outside that - any other ``Tool``
    property, or any other class's object tree entirely - is left to the
    next resolver (see :func:`pyippdme.server.classes.server_class._children`).
    """
    if reference in (CommandName.TOOL, CommandName.FOUND_TOOL):
        fixed: tuple[tuple[str, str], ...] = (
            ("Id", PropertyKind.STRING),
            ("Name", PropertyKind.STRING),
            ("Collection", PropertyKind.STRING),
            ("LastQualified", PropertyKind.STRING),
            ("AvrOffsets", PropertyKind.NUMBER),
            ("CollisionVolume", PropertyKind.NUMBER),
            ("AlignmentVolume", PropertyKind.NUMBER),
        )
        if reference == CommandName.TOOL:
            fixed += (
                ("GoToPar", PropertyKind.PROPERTY),
                ("PtMeasPar", PropertyKind.PROPERTY),
                ("ScanPar", PropertyKind.PROPERTY),
                ("AlignMode", PropertyKind.PROPERTY),
            )
        tool_name = _named_tool_name(ctx, reference)
        if tool_name in _ALIGNABLE_TOOLS:
            fixed += (
                ("Alignment", PropertyKind.NUMBER),
                ("A", PropertyKind.NUMBER),
                ("B", PropertyKind.NUMBER),
                ("C", PropertyKind.NUMBER),
            )
        return fixed
    parts = reference.split(".")
    if parts == [CommandName.TOOL, "AlignMode"]:
        return ()  # 6.18.2, 6.19.2: the alignment classes have no properties
    if len(parts) == 2 and parts[0] == CommandName.TOOL:
        block = ctx.state.tool.parameters.block(parts[1])
        if block is None:
            return None
        return tuple((name, PropertyKind.PROPERTY) for name in block)
    if len(parts) == 3 and parts[0] == CommandName.TOOL:
        block = ctx.state.tool.parameters.block(parts[1])
        if block is None or parts[2] not in block:
            return None
        return tuple((field, PropertyKind.NUMBER) for field in PARAMETER_FIELDS)
    return None


async def _tool(ctx: Ctx, _args: tuple[Argument, ...]) -> HandlerResult:
    return builders.name_value(ctx.state.tool.active_name)  # Table 64: the pointer to the tool


async def _is_alignable(ctx: Ctx, _args: tuple[Argument, ...]) -> HandlerResult:
    return builders.named_boolean(
        CommandName.IS_ALIGNABLE, ctx.state.tool.active_name in _ALIGNABLE_TOOLS
    )


async def _re_qualify(ctx: Ctx, _args: tuple[Argument, ...]) -> HandlerResult:
    # 6.10.2's own "General Remarks" under ReQualify(): "does not explicitly respond a
    # successful requalification... if the method terminates without
    # sending an error, the tool is requalified successfully" - without a tool
    # handler there is no calibration artefact to measure against, so this
    # always succeeds and just records when it happened.
    if ctx.tool_handler is not None:
        await ctx.tool_handler.requalify(ctx.state.tool.active_name, ctx.cancel)
    ctx.state.tool.last_qualified[ctx.state.tool.active_name] = datetime.now(UTC).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    return None


_ALIGN_TOOL_PARAMS = (
    Parameter("i1", DataType.FLOAT, positional=True),
    Parameter("j1", DataType.FLOAT, positional=True),
    Parameter("k1", DataType.FLOAT, positional=True),
    Parameter("i2", DataType.FLOAT, mandatory=False, positional=True),
    Parameter("j2", DataType.FLOAT, mandatory=False, positional=True),
    Parameter("k2", DataType.FLOAT, mandatory=False, positional=True),
    Parameter("alpha", DataType.FLOAT, positional=True),
    Parameter("beta", DataType.FLOAT, mandatory=False, positional=True),
)


async def _align_tool(ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
    """``AlignTool(i1,j1,k1,[i2,j2,k2,]alpha,[beta])`` (Table 105).

    ``i2,j2,k2``/``alpha``/``beta`` sit in the middle of the wire argument
    order, not trailing it, so unlike every other positional command in
    this project the valid shapes aren't a contiguous count range - only
    exactly 4 (``i1,j1,k1,alpha``) or 8 (all eight) numbers are accepted.
    Since this simulation has no real physical alignment to fall short of,
    the tool always reaches exactly the requested orientation - ``alpha``/
    ``beta`` are validated (must be non-negative) but can never actually
    trigger ``1507 Tool not alignable to given orientation``.
    """
    _require_alignable(ctx.state.tool.active_name, CommandName.ALIGN_TOOL)
    if len(args) not in (4, 8) or not all(isinstance(a, Number) for a in args):
        raise bad_argument(CommandName.ALIGN_TOOL, "Expected 4 or 8 plain numbers")
    values = tuple(a.value for a in args if isinstance(a, Number))
    secondary_raw: Vec3 | None
    beta: float | None
    if len(values) == 4:
        i1, j1, k1, alpha = values
        secondary_raw = None
        beta = None
    else:
        i1, j1, k1, i2, j2, k2, alpha, beta = values
        secondary_raw = (i2, j2, k2)
    if alpha < 0 or (beta is not None and beta < 0):
        raise bad_argument(CommandName.ALIGN_TOOL, "alpha/beta must not be negative")
    try:
        primary = normalize((i1, j1, k1))
        secondary = normalize(secondary_raw) if secondary_raw is not None else None
    except ValueError as exc:
        raise bad_argument(CommandName.ALIGN_TOOL, "i1,j1,k1/i2,j2,k2 must be non-zero") from exc
    ctx.state.tool.alignment[ctx.state.tool.active_name] = (primary, secondary)
    # AlignTool() moves the machine, so (6.7.1) it implicitly executes DisableUser().
    ctx.state.mover.user_enabled = False
    # Table 105: the reached vectors are returned unnamed ("Kind U").
    return NumericData(tuple(Number.of(c) for c in (*primary, *(secondary or ()))))


async def _avr_radius(_ctx: Ctx, _args: tuple[Argument, ...]) -> HandlerResult:
    # 6.20.1's own remark: zero "in all other cases" than a sphere/cylinder
    # tip qualified with an effective-tool-radius algorithm - this
    # simulation has no such qualification, so always zero.
    return builders.named_numbers(AvrRadius=0.0)


def _resolve_alignment_namespace(arg_name: str, cause: str) -> str:
    namespace = arg_name.split(".", 1)[0]
    if namespace not in (CommandName.TOOL, CommandName.FOUND_TOOL):
        raise bad_argument(cause, f"Expected a Tool.*/FoundTool.* argument, got {arg_name!r}")
    return namespace


async def _calc_tool_alignment(ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
    """``CalcToolAlignment(Axes)`` (Table 107).

    ``Axes`` names one of ``Tool.A()``/``Tool.B()``/``Tool.C()`` (or the
    ``FoundTool`` equivalents) - only its ``Tool``/``FoundTool`` namespace
    prefix matters here, since the response is always named
    ``"Tool.Alignment"``/``"FoundTool.Alignment"`` regardless of which axis
    letter was actually given (6.20.1's own wording never distinguishes
    them further).
    """
    if len(args) != 1 or not isinstance(args[0], NamedValue) or args[0].args:
        raise bad_argument(CommandName.CALC_TOOL_ALIGNMENT, "Expected a single Tool.A()-style axis")
    namespace = _resolve_alignment_namespace(args[0].name, CommandName.CALC_TOOL_ALIGNMENT)
    tool_name = _require_alignable(
        _named_tool_name(ctx, namespace), CommandName.CALC_TOOL_ALIGNMENT
    )
    primary, secondary = tool_alignment(ctx, tool_name)
    numbers = (*primary, *(secondary or ()))
    return Items((NamedValue(f"{namespace}.Alignment", tuple(Number.of(c) for c in numbers)),))


async def _calc_tool_angles(ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
    """``CalcToolAngles(Alignment)`` (Table 108): the inverse of ``CalcToolAlignment``.

    Does not move the tool or touch stored alignment state - a pure
    calculation from the given vectors, via the same
    :func:`tool_angles` convention ``A()``/``B()``/``C()`` use.
    """
    if len(args) != 1 or not isinstance(args[0], NamedValue):
        raise bad_argument(CommandName.CALC_TOOL_ANGLES, "Expected a single Tool.Alignment(...)")
    arg = args[0]
    namespace = _resolve_alignment_namespace(arg.name, CommandName.CALC_TOOL_ANGLES)
    _require_alignable(_named_tool_name(ctx, namespace), CommandName.CALC_TOOL_ANGLES)
    if len(arg.args) not in (3, 6) or not all(isinstance(a, Number) for a in arg.args):
        raise bad_argument(CommandName.CALC_TOOL_ANGLES, "Expected 3 or 6 plain numbers")
    numbers = tuple(a.value for a in arg.args if isinstance(a, Number))
    primary: Vec3 = (numbers[0], numbers[1], numbers[2])
    secondary: Vec3 | None = (numbers[3], numbers[4], numbers[5]) if len(numbers) == 6 else None
    a, b, c = tool_angles(primary, secondary)
    fields = [
        NamedValue(f"{namespace}.A", (Number.of(a),)),
        NamedValue(f"{namespace}.B", (Number.of(b),)),
    ]
    if secondary is not None:
        fields.append(NamedValue(f"{namespace}.C", (Number.of(c),)))
    return Items(tuple(fields))


_USE_SMALLEST_ANGLE_PARAMS = (Parameter("Flag", DataType.BOOL, positional=True),)


async def _use_smallest_angle_to_align_tool(ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
    if len(args) != 1 or not isinstance(args[0], Number):
        raise bad_argument(
            CommandName.USE_SMALLEST_ANGLE_TO_ALIGN_TOOL, "Expected a single boolean"
        )
    # Stored but inert: genuinely detecting "would rotate more than 180
    # degrees" needs a real per-axis kinematic travel model this simulation
    # doesn't have - see the module docstring.
    ctx.state.tool.use_smallest_angle = args[0].value != 0
    return None


async def _enable_optimize(ctx: Ctx, _args: tuple[Argument, ...]) -> HandlerResult:
    ctx.state.tool.optimize_enabled = True
    return None


async def _disable_optimize(ctx: Ctx, _args: tuple[Argument, ...]) -> HandlerResult:
    ctx.state.tool.optimize_enabled = False
    return None


async def _is_optimize_enabled(ctx: Ctx, _args: tuple[Argument, ...]) -> HandlerResult:
    # Table 112 names the value "IsOptimizedEnabled" (Kind N: named after the variable).
    return builders.named_boolean("IsOptimizedEnabled", ctx.state.tool.optimize_enabled)


def _pointer(name: str) -> CommandHandler:
    """Build the handler of a pointer command such as ``PtMeasPar()`` (Table 77)."""

    async def pointer(_ctx: Ctx, _args: tuple[Argument, ...]) -> HandlerResult:
        return builders.name_value(f"{CommandName.TOOL}.{name}")

    return pointer


async def _opt_par(_ctx: Ctx, _args: tuple[Argument, ...]) -> HandlerResult:
    # OptPar "only exists for tools of class Optical" (6.10.4); the catalog has none.
    raise ServerError(
        ErrorSeverity.WARNING,
        ErrorCode.TOOL_PROPERTY_NOT_APPLICABLE,
        CommandName.OPT_PAR,
        "Tool property not applicable",
    )


_CALC_TOOL_ALIGNMENT_PARAMS = (Parameter("Axes", DataType.ENUM),)
_CALC_TOOL_ANGLES_PARAMS = (Parameter("Alignment", DataType.ENUM),)


def _reset_on_start_session(ctx: Ctx) -> None:
    # Alignable_AB 6.20.1's own defaults: "StartSession() establishes the
    # default value UseSmallestAngleToAlignTool(0)" / "StartSession() by
    # default sets DisableOptimize()".
    ctx.state.tool.use_smallest_angle = False
    ctx.state.tool.optimize_enabled = False


def _reset_optimize_on_clear_errors(ctx: Ctx) -> None:
    # Alignable_AB 6.20.1: "ClearAllErrors() sets DisableOptimize()".
    ctx.state.tool.optimize_enabled = False


def register(registry: CommandRegistry) -> None:
    registry.register(CommandName.TOOL, _tool, arguments=())
    registry.register(CommandName.IS_ALIGNABLE, _is_alignable, arguments=())
    registry.register(CommandName.RE_QUALIFY, _re_qualify, arguments=())
    registry.register(CommandName.ALIGN_TOOL, _align_tool, arguments=_ALIGN_TOOL_PARAMS)
    registry.register(CommandName.AVR_RADIUS, _avr_radius, arguments=())
    registry.register(
        CommandName.CALC_TOOL_ALIGNMENT,
        _calc_tool_alignment,
        arguments=_CALC_TOOL_ALIGNMENT_PARAMS,
    )
    registry.register(
        CommandName.CALC_TOOL_ANGLES, _calc_tool_angles, arguments=_CALC_TOOL_ANGLES_PARAMS
    )
    registry.register(
        CommandName.USE_SMALLEST_ANGLE_TO_ALIGN_TOOL,
        _use_smallest_angle_to_align_tool,
        arguments=_USE_SMALLEST_ANGLE_PARAMS,
    )
    registry.register(CommandName.ENABLE_OPTIMIZE, _enable_optimize, arguments=())
    registry.register(CommandName.DISABLE_OPTIMIZE, _disable_optimize, arguments=())
    registry.register(CommandName.IS_OPTIMIZE_ENABLED, _is_optimize_enabled, arguments=())
    registry.register(CommandName.GO_TO_PAR, _pointer("GoToPar"), arguments=())
    registry.register(CommandName.PT_MEAS_PAR, _pointer("PtMeasPar"), arguments=())
    registry.register(CommandName.SCAN_PAR, _pointer("ScanPar"), arguments=())
    registry.register(CommandName.OPT_PAR, _opt_par, arguments=())
    registry.register_property_resolver(setter=_try_set_tool_param, getter=_try_get_tool_param)
    registry.register_property_children(_tool_property_children)
    registry.register_session_start_hook(_reset_on_start_session)
    registry.register_clear_errors_hook(_reset_optimize_on_clear_errors)
