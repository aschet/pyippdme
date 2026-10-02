# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""The command catalog of this library's own bundled simulated classes.

See :mod:`pyippdme.server.catalog` for the generic catalog-building function
this is built from - call it yourself for any other ``command_classes``
selection; this module just applies it once, to
:data:`~pyippdme.simulation.DEFAULT_COMMAND_CLASSES`, and freezes the result.
"""

from __future__ import annotations

from collections.abc import Mapping

from pyippdme.server.catalog import CommandInfo, build_command_catalog
from pyippdme.simulation import DEFAULT_COMMAND_CLASSES

#: The catalog of every command :data:`~pyippdme.simulation.DEFAULT_COMMAND_CLASSES`
#: (every built-in simulated class) registers, keyed by command name.
BUILTIN_COMMANDS: Mapping[str, CommandInfo] = build_command_catalog(DEFAULT_COMMAND_CLASSES)
