# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Per-command argument schemas backing ``GetSupportedArguments(CommandName)`` (6.4.1).

Each built-in command class declares a :class:`Parameter` tuple per
command, matching that command's own "Parameters" table in the standard
(one entry per top-level parameter - an enum parameter like ``GoTo``'s
``Positions`` is one entry, not one per possible item inside it). The
registry stores these alongside handlers
(:meth:`~pyippdme.server.registry.CommandRegistry.register`'s ``arguments``
keyword) so ``GetSupportedArguments`` can answer for real instead of
always reporting "not supported".

For the common case of a command whose arguments are a fixed, ordered list
of plain numbers (the standard's "Kind U" positional style, e.g.
``ScanOnLine(Sx, Sy, Sz, Ex, Ey, Ez, i, j, k, StepW, RT)``),
:func:`positional_float_parameters` builds that :class:`Parameter` tuple
*and* the ``(min_count, max_count)`` bounds
:func:`~pyippdme.server._util.positional_numbers` needs to validate the
call - from the same list of names, so the advertised schema and the
enforced argument count cannot drift apart. Commands whose arguments are
richer (an enum, a named property lookup, ...) still validate by hand in
their own handler, since their per-item structure is beyond what the six
datatypes (Table 2: ``bool``, ``int``, ``float``, ``string``,
``enum``, ``name``) can express at this granularity - :func:`GetSupportedArguments`
only ever describes top-level parameters, not what belongs inside an enum.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class DataType(StrEnum):
    """Table 2's six wire datatypes for a command/property argument."""

    BOOL = "bool"
    INT = "int"
    FLOAT = "float"
    STRING = "string"
    ENUM = "enum"
    NAME = "name"


@dataclass(frozen=True, slots=True)
class Parameter:
    """One top-level parameter of a command, as the standard's own tables define it.

    ``positional`` is not part of that wire schema at all (``GetSupportedArguments``
    only ever reports ``name``/``datatype`` - see
    :func:`~pyippdme.server.classes.dme_class._get_supported_arguments`) - it is
    local, descriptive metadata for tooling built on this same catalog (the TUI's
    argument completion and ``.help``; see :mod:`pyippdme.cli.tui`), marking
    whether the *call itself* writes this argument as a bare value rather than
    as a ``Name(value)`` pair like ``GoTo(X(10), Sync(1))``. Two distinct
    shapes share this flag: a fixed, ordered list of several bare values
    (the standard's "Kind U" style, e.g. ``ScanOnLine(1, 2, 3, ...)`` - see
    :func:`positional_float_parameters`), and a single command whose lone
    argument just isn't named in its call at all (e.g.
    ``GetErrorInfo(506)``, not ``GetErrorInfo(ErrorNumber(506))``). Neither
    is derivable from ``datatype``/parameter count alone - whether a given
    command uses this shape has to be read off its handler; see
    ``tests/test_supported_arguments.py``'s
    ``test_every_positional_command_matches_this_audit`` for the full,
    by-hand-audited list this flag encodes.
    """

    name: str
    datatype: DataType
    mandatory: bool = True
    positional: bool = False


def positional_float_parameters(
    *names: str, optional: tuple[str, ...] = ()
) -> tuple[Parameter, ...]:
    """Build the ``Parameter`` tuple for a fixed, ordered list of float arguments.

    ``optional`` names (if any) must be the trailing entries of ``names``,
    matching how every such command in this project (``ScanOnLine``'s
    ``RT``, etc.) puts its one optional argument last. Every ``Parameter``
    this builds has ``positional=True`` - see :class:`Parameter`.
    """
    return tuple(
        Parameter(name, DataType.FLOAT, mandatory=name not in optional, positional=True)
        for name in names
    )


def argument_count_bounds(parameters: tuple[Parameter, ...]) -> tuple[int, int]:
    """Return the ``(min_count, max_count)`` bounds for ``positional_numbers`` to enforce.

    See :func:`~pyippdme.server._util.positional_numbers`.
    """
    mandatory = sum(1 for p in parameters if p.mandatory)
    return mandatory, len(parameters)
