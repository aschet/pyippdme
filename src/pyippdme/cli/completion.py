# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""What can be typed next in a command line: the part of completion that needs no terminal.

Both interactive front ends use it: the full-screen :mod:`pyippdme.cli.tui` (a dropdown) and the
simple shell of :mod:`pyippdme.cli.client` (``readline`` Tab cycling). It reads the catalog of
the built-in commands, so only the bundled command classes are known.
"""

from __future__ import annotations

import re

from pyippdme.protocol.parameters import ParameterName
from pyippdme.protocol.signature import DataType
from pyippdme.simulation.catalog import BUILTIN_COMMANDS

__all__ = ["argument_names", "candidate_names", "completion_context", "paren_depth"]

#: Offered whenever a command has a ``DataType.ENUM`` top-level argument (``GoTo``'s ``Positions``,
#: ``Get``'s ``Axes``, ...). The standard's ``GetSupportedArguments`` stops at that one schema
#: label - what belongs inside an enum is not machine-readable - and nobody types the label
#: itself. What is typed inside one is, across the built-in commands, drawn from this small set of
#: axis names, so they stand in for a generic solution.
ENUM_ARGUMENT_FALLBACK = (
    ParameterName.X,
    ParameterName.Y,
    ParameterName.Z,
    ParameterName.R,
)

_IDENTIFIER_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*$")


def completion_context(text_before_cursor: str) -> tuple[str | None, str]:
    """Parse text up to the cursor into ``(command whose args we're inside, partial word)``.

    ``command`` is ``None`` when the cursor isn't inside any command's parentheses (suggest
    command names); otherwise it is the *outermost* open command, even when the cursor is nested
    deeper (e.g. inside ``ScanOnCurve(Format(``): nested groups are rare in this protocol, so
    reusing the outer command's argument names is an acceptable simplification.
    """
    depth = 0
    command_name: str | None = None
    for index, char in enumerate(text_before_cursor):
        if char == "(":
            if depth == 0:
                match = _IDENTIFIER_RE.search(text_before_cursor[:index])
                command_name = match.group(0) if match else None
            depth += 1
        elif char == ")" and depth > 0:
            depth -= 1
            if depth == 0:
                command_name = None
    partial_match = _IDENTIFIER_RE.search(text_before_cursor)
    partial = partial_match.group(0) if partial_match else ""
    return (command_name if depth >= 1 else None), partial


def paren_depth(text: str) -> int:
    """Return how many parentheses are still open at the end of ``text``."""
    depth = 0
    for char in text:
        if char == "(":
            depth += 1
        elif char == ")" and depth > 0:
            depth -= 1
    return depth


def argument_names(command_name: str) -> tuple[str, ...]:
    """Return the argument names that can be typed inside ``command_name(``.

    Empty for an unknown command and for commands whose arguments are positional values (such as
    ``ScanOnLine``): their parameter names are never typed.
    """
    info = BUILTIN_COMMANDS.get(command_name)
    arguments = info.arguments if info is not None else None
    if not arguments or all(p.positional for p in arguments):
        return ()
    named = tuple(p.name for p in arguments if p.datatype != DataType.ENUM)
    if any(p.datatype == DataType.ENUM for p in arguments):
        return (*named, *ENUM_ARGUMENT_FALLBACK)
    return named


def candidate_names(text_before_cursor: str, *, top_level_only: bool = False) -> list[str]:
    """Return what can complete the word at the end of ``text_before_cursor``, sorted as listed.

    Outside any parentheses these are command names; inside a command's, its argument names.
    ``top_level_only`` offers argument names only directly inside the command's own brackets, not
    inside a nested ``X(``.
    """
    command_name, partial = completion_context(text_before_cursor)
    if command_name is None:
        names: tuple[str, ...] = tuple(BUILTIN_COMMANDS)
    elif top_level_only and paren_depth(text_before_cursor) > 1:
        names = ()
    else:
        names = argument_names(command_name)
    needle = partial.casefold()
    return [name for name in names if name.casefold().startswith(needle)]
