# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Shared argument-parsing/error/property-store helpers for command handlers.

Used by the mandatory :mod:`pyippdme.server.classes.server_class` (6.3)
directly, and by every :mod:`pyippdme.simulation.classes` handler.
"""

from __future__ import annotations

import re
from typing import Any

from pyippdme.protocol.ast import Argument, BasicName, Items, NamedValue, Number
from pyippdme.protocol.commands import CommandName
from pyippdme.protocol.errors import ErrorCode, ErrorSeverity, ServerError
from pyippdme.server.registry import CommandContext, HandlerResult
from pyippdme.types.vec3 import Vec3, dot, normalize

#: How far from perpendicular (as |cos(angle)|) :func:`check_orthogonal` still
#: accepts, absorbing floating-point error in the client-supplied vectors.
ORTHOGONALITY_TOLERANCE = 1e-3


def bad_argument(cause: str, text: str = "Bad argument") -> ServerError:
    return ServerError(ErrorSeverity.CRITICAL, ErrorCode.BAD_ARGUMENT, cause, text)


def incorrect_arguments(cause: str, text: str) -> ServerError:
    return ServerError(ErrorSeverity.CRITICAL, ErrorCode.INCORRECT_ARGUMENTS, cause, text)


def check_orthogonal(cause: str, normal: Vec3, along: Vec3, text: str) -> None:
    """Raise ``0502`` unless ``normal`` is (near enough) perpendicular to ``along``.

    Shared by every command that requires a direction vector to lie in a
    plane a normal vector defines (``ScanOnLine``/``ScanOnCircle``/
    ``ScanInPlaneEndIsSphere`` in :mod:`pyippdme.simulation.classes.scanning_class`,
    ``GoToOnCircle`` in :mod:`pyippdme.simulation.classes.cartcmm_class`).
    """
    try:
        if abs(dot(normalize(normal), normalize(along))) > ORTHOGONALITY_TOLERANCE:
            raise incorrect_arguments(cause, text)
    except ValueError as exc:
        raise incorrect_arguments(cause, "Direction and normal vectors must be non-zero") from exc


def named_number(args: tuple[Argument, ...], name: str) -> float | None:
    """Return the single numeric value of a named argument like ``X(5)``, if present."""
    for arg in args:
        if isinstance(arg, NamedValue) and arg.name == name:
            if len(arg.args) != 1 or not isinstance(arg.args[0], Number):
                raise bad_argument(name, f"Expected {name}(<number>)")
            return arg.args[0].value
    return None


def named_vector(args: tuple[Argument, ...], name: str) -> Vec3 | None:
    """Return the ``(x, y, z)`` of a named 3-number argument like ``Center(1,2,3)``, if present."""
    for arg in args:
        if isinstance(arg, NamedValue) and arg.name == name:
            if len(arg.args) != 3 or not all(isinstance(a, Number) for a in arg.args):
                raise bad_argument(name, f"Expected {name}(<x>,<y>,<z>)")
            x, y, z = (a.value for a in arg.args if isinstance(a, Number))
            return (x, y, z)
    return None


def single_basic_name(args: tuple[Argument, ...], cause: str) -> str:
    """Require exactly one bare-name argument (e.g. an enum value) and return it."""
    if len(args) != 1 or not isinstance(args[0], BasicName):
        raise bad_argument(cause, "Expected a single enumerated value")
    return args[0].value


#: A proprietary property name starts with a two-letter company namespace (6.1), e.g. ``XXMyValue``.
_PROPRIETARY_NAME_RE = re.compile(r"^[A-Z]{2}[A-Za-z0-9]+(\.|$)")


def _require_proprietary(name: str, cause: str) -> None:
    """Raise ``0505`` unless ``name`` could be a vendor extension (6.1).

    A property the standard does not define must carry a two-letter
    namespace prefix, so any other unresolved name is simply not a property.
    """
    if _PROPRIETARY_NAME_RE.match(name) is None:
        raise ServerError(
            ErrorSeverity.CRITICAL,
            ErrorCode.ARGUMENT_NOT_RECOGNIZED,
            cause,
            f"Argument {name} not recognized",
        )


def generic_set_prop(
    ctx: CommandContext[Any], args: tuple[Argument, ...], cause: str = CommandName.SET_PROP
) -> HandlerResult:
    """Apply the fallback ``SetProp`` behaviour: a per-connection store for proprietary properties.

    Shared by :mod:`pyippdme.server.classes.server_class` (the plain, generic
    ``SetProp``) and :mod:`pyippdme.simulation.classes.tool_class` (which special-cases
    ``Tool.GoToPar``/``PtMeasPar``/``ScanPar`` paths and falls back to this
    for everything else), so both stay consistent without one importing the
    other's private handler.
    """
    for arg in args:
        if not isinstance(arg, NamedValue):
            raise ServerError(ErrorSeverity.CRITICAL, ErrorCode.BAD_PROPERTY, cause, "Bad property")
        _require_proprietary(arg.name, cause)
        ctx.state.properties[arg.name] = arg.args
    return None


def generic_get_prop(
    ctx: CommandContext[Any], args: tuple[Argument, ...], cause: str = CommandName.GET_PROP
) -> HandlerResult:
    results: list[NamedValue] = []
    for arg in args:
        if not isinstance(arg, NamedValue):
            raise ServerError(ErrorSeverity.CRITICAL, ErrorCode.BAD_PROPERTY, cause, "Bad property")
        _require_proprietary(arg.name, cause)
        stored = ctx.state.properties.get(arg.name)
        if stored is None:
            raise ServerError(
                ErrorSeverity.CRITICAL,
                ErrorCode.ARGUMENT_NOT_SUPPORTED,
                cause,
                f"Property {arg.name} is not set",
            )
        results.append(NamedValue(arg.name, stored))
    return Items(tuple(results))


def positional_numbers(
    args: tuple[Argument, ...], cause: str, min_count: int, max_count: int
) -> tuple[float, ...]:
    """Require ``min_count``-``max_count`` bare (unnamed) numeric arguments, in order.

    Several commands (notably the ``Scanning`` class's known-contour scans)
    take a fixed-order list of plain numbers rather than named properties,
    e.g. ``ScanOnLine(Sx, Sy, Sz, Ex, Ey, Ez, i, j, k, StepW, RT)``.
    """
    if not (min_count <= len(args) <= max_count) or not all(isinstance(a, Number) for a in args):
        raise bad_argument(cause, f"Expected {min_count}-{max_count} plain numbers")
    return tuple(a.value for a in args if isinstance(a, Number))
