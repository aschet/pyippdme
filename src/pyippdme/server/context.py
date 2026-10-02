# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""The concrete :class:`~pyippdme.server.registry.CommandContext` the mandatory classes use.

:mod:`pyippdme.server.classes`' ``Server``/``DME`` handlers only ever read
the base :class:`~pyippdme.server.registry.MachineState` fields (session,
error, machine identity), so they're typed against ``Ctx`` here rather than
the bare, unparameterized :class:`~pyippdme.server.registry.CommandContext` -
mypy's strict mode requires every use of a generic type to specify its
parameter. Compare :mod:`pyippdme.simulation.context`'s own ``Ctx``, pinned
to :class:`~pyippdme.simulation.state.SimulationState` instead, for that
package's handlers.
"""

from __future__ import annotations

from typing import TypeAlias

from pyippdme.server.registry import CommandContext, MachineState

Ctx: TypeAlias = CommandContext[MachineState]
