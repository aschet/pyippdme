# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""The concrete :class:`~pyippdme.server.registry.CommandContext` this package's handlers use.

Every handler in :mod:`pyippdme.simulation.classes` reads/writes
:class:`~pyippdme.simulation.state.SimulationState`-specific fields
(``ctx.state.cart_cmm``, ``ctx.state.tool``, ...), so they're typed against
``Ctx`` here rather than the bare, unparameterized
:class:`~pyippdme.server.registry.CommandContext` - mypy's strict mode
requires every use of a generic type to specify its parameter, and this is
the one, single place that pins it to
:class:`~pyippdme.simulation.state.SimulationState` for the whole package.
"""

from __future__ import annotations

from typing import TypeAlias

from pyippdme.server.registry import CommandContext
from pyippdme.types.csy import CsyContext
from pyippdme.simulation.state import SimulationState

Ctx: TypeAlias = CommandContext[SimulationState]


def csy_context(ctx: Ctx) -> CsyContext:
    """The client's coordinate system chain: the active CSY and the transformations it set.

    A motion model that knows more than the commands set (the rotary table's angle) supplies the
    complete context itself.
    """
    if ctx.motion is not None:
        provided = ctx.motion.csy_context()
        if provided is not None:
            return provided
    cart = ctx.state.cart_cmm
    return CsyContext(cart.active_csy, cart.csy_transformations)
