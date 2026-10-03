# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""The ``Part`` command class (6.24).

``Part`` has no commands of its own (6.24.1), only properties (6.24.2):
``Temperature``, ``XpanCoefficient``, ``Approach``, ``Search``, and
``Retract``, all plain read/write floats with a fixed
``StartSession()``-established default. Registers a
:meth:`~pyippdme.server.registry.CommandRegistry.register_property_resolver`
resolver (see :mod:`pyippdme.simulation.classes.tool_class`'s module docstring for why
that mechanism exists, rather than this module overriding
``SetProp``/``GetProp`` outright) instead of a dedicated command, plus a
:meth:`~pyippdme.server.registry.CommandRegistry.register_property_children`
resolver so ``Server``'s ``EnumProp``/``EnumAllProp`` (6.3.1.1) can list
these five properties for the ``Part`` reference.
"""

from __future__ import annotations

from pyippdme.protocol.ast import NamedValue, Number
from pyippdme.protocol.commands import CommandName
from pyippdme.protocol.errors import ErrorCode, ErrorSeverity, ServerError
from pyippdme.server.registry import CommandRegistry, PropertyKind
from pyippdme.simulation.context import Ctx

#: 6.24.2's stated ``StartSession()`` defaults ("The default value is 20°C
#: after a StartSession()", "...Approach(0.0) will be established by
#: StartSession()", etc.); ``Retract`` states no explicit default, so 0.0 is
#: used for consistency with the others.
_DEFAULTS = {
    "Part.Temperature": 20.0,
    "Part.XpanCoefficient": 0.0,
    "Part.Approach": 0.0,
    "Part.Search": 0.0,
    "Part.Retract": 0.0,
}


def _try_set(ctx: Ctx, arg: NamedValue) -> bool:
    if arg.name not in _DEFAULTS:
        return False
    if len(arg.args) != 1 or not isinstance(arg.args[0], Number):
        raise ServerError(
            ErrorSeverity.CRITICAL,
            ErrorCode.BAD_ARGUMENT,
            CommandName.SET_PROP,
            f"Expected {arg.name}(<number>)",
        )
    ctx.state.part.properties[arg.name] = arg.args[0].value
    return True


def _try_get(ctx: Ctx, arg: NamedValue) -> NamedValue | None:
    default = _DEFAULTS.get(arg.name)
    if default is None:
        return None
    value = ctx.state.part.properties.get(arg.name, default)
    return NamedValue(arg.name, (Number.of(value),))


def _property_children(ctx: Ctx, reference: str) -> tuple[tuple[str, str], ...] | None:
    del ctx  # Part's properties are the same fixed set for every connection.
    if reference != "Part":
        return None
    return tuple((name.removeprefix("Part."), PropertyKind.NUMBER) for name in _DEFAULTS)


def _reset_on_start_session(ctx: Ctx) -> None:
    # 6.24.2: each property has a documented default "after a StartSession()".
    ctx.state.part.properties.clear()


def register(registry: CommandRegistry) -> None:
    registry.register_session_start_hook(_reset_on_start_session)
    registry.register_property_resolver(setter=_try_set, getter=_try_get)
    registry.register_property_children(_property_children)
