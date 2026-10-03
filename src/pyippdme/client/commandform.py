# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Build command lines from a form: what a front end needs to offer every command, without a GUI.

A command's parameters come from the catalog of the bundled server classes
(:data:`~pyippdme.simulation.catalog.BUILTIN_COMMANDS`) or from any catalog you build with
:func:`~pyippdme.server.catalog.build_command_catalog`. :func:`command_fields` describes one
command as an ordered list of :class:`FormField`, and :func:`build_command_line` turns the values a
user typed into the text :func:`~pyippdme.protocol.parser.parse_method` accepts::

    fields = command_fields("GoTo")                      # one enum field: "Positions"
    build_command_line("GoTo", {"Positions": "X(10),Y(20)"})   # 'GoTo(X(10),Y(20))'
    build_command_line("ScanOnLine", {"Sx": "0", ...})   # positional commands: bare values
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass

from pyippdme.protocol.signature import DataType
from pyippdme.simulation.catalog import BUILTIN_COMMANDS

__all__ = [
    "FormField",
    "build_command_line",
    "command_fields",
    "command_names",
    "result_rows",
]


@dataclass(frozen=True, slots=True)
class FormField:
    """One input of a command form."""

    name: str
    datatype: DataType
    mandatory: bool
    #: Written as a bare value in call order, not as ``Name(value)``.
    positional: bool

    @property
    def hint(self) -> str:
        """A short description for a tooltip or placeholder."""
        if self.datatype is DataType.ENUM:
            return "list of items, e.g. X(10),Y(20)"
        kind = {
            DataType.BOOL: "0 or 1",
            DataType.INT: "integer",
            DataType.FLOAT: "number",
            DataType.STRING: "text",
            DataType.NAME: "name",
        }[self.datatype]
        return kind if self.mandatory else f"{kind} (optional)"


def command_names() -> list[str]:
    """All commands of the bundled catalog, sorted."""
    return sorted(BUILTIN_COMMANDS)


def command_fields(name: str) -> list[FormField]:
    """The inputs of ``name``; empty for a command without arguments (or without a schema)."""
    info = BUILTIN_COMMANDS.get(name)
    if info is None or not info.arguments:
        return []
    return [FormField(p.name, p.datatype, p.mandatory, p.positional) for p in info.arguments]


def _quote(text: str) -> str:
    return '"' + text.replace('"', "") + '"'


def _render(field: FormField, text: str) -> str:
    if field.datatype is DataType.STRING:
        return _quote(text[1:-1] if len(text) > 1 and text[0] == text[-1] == '"' else text)
    return text


def build_command_line(
    name: str, values: Mapping[str, str], fields: list[FormField] | None = None
) -> str:
    """Turn typed values into a command line; empty values are left out.

    Positional commands write their values in order and stop at the first empty optional one
    (a gap in the middle is an error, as the protocol has no way to skip an argument).
    Raises :class:`ValueError` when a mandatory value is missing.
    """
    fields = command_fields(name) if fields is None else fields
    parts: list[str] = []
    gap: str | None = None
    for field in fields:
        text = values.get(field.name, "").strip()
        if not text:
            if field.mandatory:
                raise ValueError(f"{field.name} is required")
            gap = gap or field.name
            continue
        if field.positional:
            if gap is not None:
                raise ValueError(f"{field.name} is given but {gap} before it is empty")
            parts.append(_render(field, text))
        elif field.datatype is DataType.ENUM:
            parts.append(text)
        else:
            parts.append(f"{field.name}({_render(field, text)})")
    return f"{name}({','.join(parts)})"


def result_rows(report: Mapping[str, object]) -> Iterator[tuple[str, str]]:
    """Name/value pairs of a decoded response, for a table."""
    for key, value in report.items():
        if isinstance(value, tuple):
            yield key, ", ".join(f"{v:g}" for v in value)
        elif isinstance(value, float):
            yield key, f"{value:g}"
        else:
            yield key, "NULL" if value is None else str(value)
