# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Look up what a command is, what its arguments are, and what it returns.

:class:`CommandReference` is the one place a front end asks: the simple shell prints it
(:class:`MetaCommands`: ``.cmds``, ``.man``), and a GUI can show the same
:class:`CommandDoc` in its own widgets::

    reference = CommandReference()
    doc = reference.find("gOtO")            # case does not matter
    doc.summary, doc.signature, doc.returns
    for argument in doc.arguments:          # name, datatype, mandatory, positional, description
        ...
    print(reference.manual("GoTo"))         # the same as plain text

The command catalog (names, argument types, whether an argument is optional or a bare value)
comes from the bundled command classes. The descriptions are written for this package
(:mod:`pyippdme.client.reference_text`), not taken from the standard; for a command of your own
pass its description with :meth:`CommandReference.describe`.
"""

from __future__ import annotations

import textwrap
from collections.abc import Mapping
from dataclasses import dataclass, field

from pyippdme.client import commandform, reference_text
from pyippdme.protocol.signature import Parameter
from pyippdme.server.catalog import CommandInfo
from pyippdme.simulation.catalog import BUILTIN_COMMANDS

__all__ = ["ArgumentDoc", "CommandDoc", "CommandReference", "MetaCommands", "MetaResult"]


@dataclass(frozen=True, slots=True)
class ArgumentDoc:
    """One argument of a command."""

    name: str
    #: The wire type: ``float``, ``int``, ``bool``, ``string``, ``name`` or ``enum``.
    datatype: str
    mandatory: bool
    #: Written as a bare value in a fixed order, not as ``Name(value)``.
    positional: bool
    description: str


@dataclass(frozen=True, slots=True)
class CommandDoc:
    """Everything known about one command."""

    name: str
    group: str
    summary: str
    #: Longer remarks; empty when there are none.
    notes: str
    #: A typical call; the signature when no better example is known.
    example: str
    #: What the command answers with; empty when this is not known.
    returns: str
    arguments: tuple[ArgumentDoc, ...] = field(default_factory=tuple)

    @property
    def signature(self) -> str:
        """The call with its argument names; optional ones are marked with ``*``."""
        if not self.arguments:
            return f"{self.name}()"
        inner = ", ".join(a.name + ("" if a.mandatory else "*") for a in self.arguments)
        return f"{self.name}({inner})"


def _group_of(name: str) -> str:
    for group, names in commandform.COMMAND_GROUPS.items():
        if name in names:
            return group
    return "Other"


class CommandReference:
    """The commands of a catalog with their descriptions."""

    def __init__(
        self,
        catalog: Mapping[str, CommandInfo] | None = None,
        *,
        summaries: Mapping[str, str] | None = None,
    ) -> None:
        self.catalog: Mapping[str, CommandInfo] = (
            catalog if catalog is not None else BUILTIN_COMMANDS
        )
        self._summaries = {**reference_text.SUMMARIES, **(summaries or {})}
        self._notes = dict(reference_text.NOTES)
        self._returns = dict(reference_text.RETURNS)
        self._examples = dict(reference_text.EXAMPLES)

    def describe(
        self,
        name: str,
        summary: str,
        *,
        notes: str = "",
        returns: str = "",
        example: str = "",
    ) -> None:
        """Add or replace the description of a command, e.g. one of your own."""
        self._summaries[name] = summary
        if notes:
            self._notes[name] = notes
        if returns:
            self._returns[name] = returns
        if example:
            self._examples[name] = example

    # -- lookup ---------------------------------------------------------------------------

    def names(self) -> list[str]:
        """Return every command name, sorted."""
        return sorted(self.catalog)

    def groups(self) -> dict[str, list[str]]:
        """Return the commands by task (session, move, scan, ...); every command is in one."""
        groups = {
            group: [n for n in names if n in self.catalog]
            for group, names in commandform.COMMAND_GROUPS.items()
        }
        rest = [n for n in self.names() if not any(n in names for names in groups.values())]
        if rest:
            groups["Other"] = rest
        return {group: names for group, names in groups.items() if names}

    def resolve(self, name: str) -> str | None:
        """Return the catalog's spelling of ``name`` (ignoring case), or ``None``."""
        if name in self.catalog:
            return name
        lowered = name.lower()
        return next((n for n in self.catalog if n.lower() == lowered), None)

    def search(self, text: str) -> list[str]:
        """Return the commands whose name or summary contains ``text`` (ignoring case)."""
        needle = text.lower()
        return [
            n
            for n in self.names()
            if needle in n.lower() or needle in self._summaries.get(n, "").lower()
        ]

    def similar(self, name: str, limit: int = 5) -> list[str]:
        """Return command names that start like ``name`` or contain it, for a "did you mean"."""
        lowered = name.lower()
        starts = [n for n in self.names() if n.lower().startswith(lowered)]
        inside = [n for n in self.names() if lowered in n.lower() and n not in starts]
        return (starts + inside)[:limit]

    def find(self, name: str) -> CommandDoc | None:
        """Return the description of ``name`` (case does not matter), or ``None``."""
        resolved = self.resolve(name)
        if resolved is None:
            return None
        info = self.catalog[resolved]
        arguments = tuple(self._argument(resolved, p) for p in (info.arguments or ()))
        doc = CommandDoc(
            resolved,
            _group_of(resolved),
            self._summaries.get(resolved, "No description yet."),
            self._notes.get(resolved, ""),
            "",
            self._returns.get(resolved, ""),
            arguments,
        )
        example = self._examples.get(resolved)
        if example is None:
            example = doc.signature
        return CommandDoc(
            doc.name, doc.group, doc.summary, doc.notes, example, doc.returns, arguments
        )

    @staticmethod
    def _argument(command: str, parameter: Parameter) -> ArgumentDoc:
        name = str(parameter.name)
        description = reference_text.ARGUMENT_OVERRIDES.get(
            (command, name), reference_text.ARGUMENTS.get(name, "")
        )
        return ArgumentDoc(
            name, str(parameter.datatype), parameter.mandatory, parameter.positional, description
        )

    # -- text -----------------------------------------------------------------------------

    def listing(self, group: str | None = None) -> str:
        """Return the commands by task as text; ``group`` (a part of its name) picks some."""
        lines: list[str] = []
        for title, names in self.groups().items():
            if group and group.lower() not in title.lower():
                continue
            lines.append(f"{title}:")
            lines.extend(self._wrapped(names))
        if not lines:
            return f"No group matches {group!r}. Groups: {', '.join(self.groups())}."
        return "\n".join(lines)

    @staticmethod
    def _wrapped(names: list[str], width: int = 78) -> list[str]:
        lines: list[str] = []
        line = "  "
        for name in names:
            if len(line) + len(name) + 1 > width and line.strip():
                lines.append(line.rstrip())
                line = "  "
            line += name + "  "
        if line.strip():
            lines.append(line.rstrip())
        return lines

    def manual(self, name: str) -> str:
        """Return the manual page of ``name`` as plain text."""
        doc = self.find(name)
        if doc is None:
            close = self.similar(name)
            hint = f" Did you mean: {', '.join(close)}?" if close else " .cmds lists the commands."
            return f"No command {name!r}.{hint}"
        lines = [f"{doc.name} - {doc.summary}", "", f"  {doc.signature}", ""]
        if doc.arguments:
            width = max(len(a.name) + (0 if a.mandatory else 1) for a in doc.arguments)
            lines.append("Arguments:")
            for a in doc.arguments:
                label = a.name + ("" if a.mandatory else "*")
                kind = a.datatype + (", a bare value" if a.positional else "")
                text = f"  {label:<{width}}  [{kind}]"
                lines.append(f"{text}  {a.description}".rstrip())
            if any(not a.mandatory for a in doc.arguments):
                lines.append("  * optional")
            if any(a.positional for a in doc.arguments):
                lines.append("  Bare values are written in the order listed, without their names.")
            lines.append("")
        else:
            lines += ["No arguments.", ""]
        if doc.returns:
            lines += [textwrap.fill(f"Returns: {doc.returns}", 78, subsequent_indent="  "), ""]
        if doc.notes:
            lines += [textwrap.fill(doc.notes, 78), ""]
        lines.append(f"Example: {doc.example}")
        lines.append(f"Group: {doc.group}")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class MetaResult:
    """What a meta command produced."""

    text: str = ""
    #: The user asked to leave.
    quit: bool = False


class MetaCommands:
    """The dot commands of an interactive shell: ``.help``, ``.cmds``, ``.man`` and ``.quit``.

    ``run`` returns the text to print, so any front end can use it::

        meta = MetaCommands()
        if meta.is_meta(line):
            result = meta.run(line)
            print(result.text)
            if result.quit: ...
    """

    PREFIX = "."
    #: Name, argument hint and what it does; the order is that of ``.help``.
    COMMANDS = (
        ("help", "", "show this list"),
        ("cmds", "[group]", "list the commands by task, or only one group"),
        ("man", "<command>", "describe a command: arguments, return value, an example"),
        ("quit", "", "leave the shell (Ctrl+D does too)"),
    )

    def __init__(self, reference: CommandReference | None = None) -> None:
        self.reference = reference or CommandReference()

    @classmethod
    def names(cls) -> list[str]:
        """Return the meta commands with their dot, e.g. ``.help``."""
        return [cls.PREFIX + name for name, _, _ in cls.COMMANDS]

    @classmethod
    def is_meta(cls, line: str) -> bool:
        """Return whether ``line`` is a meta command and not a protocol command."""
        return line.lstrip().startswith(cls.PREFIX)

    def run(self, line: str) -> MetaResult:
        """Carry out a meta command line and return what to show."""
        parts = line.strip().lstrip(self.PREFIX).split()
        name = parts[0].lower() if parts else "help"
        args = parts[1:]
        if name in ("quit", "exit", "q"):
            return MetaResult("", True)
        if name == "help":
            return MetaResult(self.help_text())
        if name == "cmds":
            return MetaResult(self.reference.listing(args[0] if args else None))
        if name == "man":
            if not args:
                return MetaResult("Usage: .man <command>   (.cmds lists the commands)")
            return MetaResult(self.reference.manual(args[0]))
        return MetaResult(f"Unknown meta command .{name}. Try .help")

    def help_text(self) -> str:
        """Return the list of meta commands."""
        lines = ["Meta commands (the rest of the line is sent to the server as a command):"]
        for name, hint, text in self.COMMANDS:
            lines.append(f"  {self.PREFIX + name + ' ' + hint:<20} {text}")
        lines.append("")
        lines.append("Commands are written like GoTo(X(10),Y(20)). Tab completes names.")
        return "\n".join(lines)
