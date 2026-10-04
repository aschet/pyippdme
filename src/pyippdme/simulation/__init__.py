# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""This library's own bundled reference simulation, built on the core :mod:`pyippdme.server`.

A bare :class:`~pyippdme.server.IppDmeServer` registers nothing and knows
nothing about simulated machine behavior (see its own docstring); this
package is where that behavior actually lives - fake position tracking, a
fake tool catalog, fake scan geometry, and the rest of what
:class:`~pyippdme.simulation.virtual_cmm.VirtualCMM` bundles together so a
client implementer has something to develop and test against without real
hardware. :mod:`pyippdme.simulation.state`'s ``SimulationState`` extends the
core's minimal ``MachineState`` with exactly that fake state, and
:mod:`pyippdme.simulation.classes` is where the command handlers that own it
live.
"""

from collections.abc import Callable, Sequence

from pyippdme.server.registry import CommandRegistry
from pyippdme.simulation.classes import (
    register_cartcmm_class,
    register_dme_class,
    register_feature_class,
    register_formtester_class,
    register_mover_class,
    register_part_class,
    register_rawdata_class,
    register_rotarytable_class,
    register_scanning_class,
    register_server_class,
    register_tool_class,
    register_toolchanger_class,
)

#: Every command class this library's own simulation bundles, in registration
#: order - what :class:`~pyippdme.simulation.virtual_cmm.VirtualCMM` registers
#: by default, and the reference set :mod:`pyippdme.server.dispatch` reads a
#: standard command's argument schema from.
DEFAULT_COMMAND_CLASSES: Sequence[Callable[[CommandRegistry], None]] = (
    register_server_class,
    register_dme_class,
    register_cartcmm_class,
    register_tool_class,
    register_toolchanger_class,
    register_scanning_class,
    register_formtester_class,
    register_mover_class,
    register_rotarytable_class,
    register_part_class,
    register_rawdata_class,
    register_feature_class,
)

__all__ = ["DEFAULT_COMMAND_CLASSES"]
