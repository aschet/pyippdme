# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Proprietary extension namespaces (6.1).

Any command, property, parameter, or attribute not defined by the standard
must be prefixed with a two-character company-specific namespace when added
by an implementation:

    Example: Let a fictional company be abbreviated by XX. Proprietary
    commands are then named as: XXProprietaryCommandName(...)
    Properties, attributes, and parameters are named analogously, e.g.:
    GoTo(XXProprietaryParameter, ...)

Nothing about a proprietary name is special at the wire-grammar level - it
is an ordinary :class:`~pyippdme.protocol.ast.BasicName`, so
:func:`proprietary_name` is purely a naming-convention helper (and a place
to catch the most common mistake, an empty or wrong-length namespace),
not something the parser or server distinguishes from any other name.
Use it both for command names (with
:meth:`~pyippdme.server.registry.CommandRegistry.register_proprietary`/
:meth:`~pyippdme.server.registry.CommandRegistry.command_proprietary`) and, since
the convention applies "analogously" to properties and parameters, when
building a :class:`~pyippdme.protocol.ast.NamedValue` for one of your own::

    NamedValue(proprietary_name("XX", "ProprietaryParameter"), (Number.of(5),))
"""

from __future__ import annotations

import re

_NAMESPACE_RE = re.compile(r"^[A-Za-z]{2}$")


def proprietary_name(namespace: str, name: str) -> str:
    """Build a spec-compliant proprietary command/property/parameter name.

    ``namespace`` must be exactly two letters (6.1: "a company
    specific abbreviation having two characters"); ``name`` is the part
    specific to your extension, e.g.
    ``proprietary_name("XX", "ProprietaryCommandName")`` ->
    ``"XXProprietaryCommandName"``.
    """
    if not _NAMESPACE_RE.match(namespace):
        raise ValueError(f"Proprietary namespace must be exactly two letters, got {namespace!r}")
    if not name:
        raise ValueError("Proprietary name must not be empty")
    return f"{namespace}{name}"
