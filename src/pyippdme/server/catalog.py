# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Build a static, offline catalog of what a set of command classes would register.

Distinct from a live server's actual capabilities - calling
``GetSupportedCommands``/``GetSupportedArguments(CommandName)`` (6.4.1)
over a real connection is the authoritative answer for *that specific*
server, which may restrict or extend what any particular
``command_classes`` sequence would otherwise register. A catalog built here
needs no server instance, no connection, and no network at all: it is
exactly what a fresh :class:`~pyippdme.server.registry.CommandRegistry`
would end up with after every given class's own ``register(registry)``
runs, computed once and frozen, so a package user building tooling around
this library - REPL/TUI completion, a GUI's command palette, a linter for
hand-written scripts - can iterate command names and their
:class:`~pyippdme.protocol.signature.Parameter` schemas directly.

See :data:`pyippdme.simulation.catalog.BUILTIN_COMMANDS` for the catalog of
this library's own bundled simulated classes; call :func:`build_command_catalog`
yourself for any other ``command_classes`` selection, e.g. your own real
integration's, or a subset a particular :class:`~pyippdme.server.IppDmeServer`
instance was configured with::

    from pyippdme.server.catalog import build_command_catalog

    info = build_command_catalog(my_command_classes)["GoTo"]
    print([p.name for p in info.arguments])  # ['Positions', 'Sync']
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from pyippdme.protocol.signature import Parameter
from pyippdme.server.registry import CommandRegistry


@dataclass(frozen=True, slots=True)
class CommandInfo:
    """One command's static schema: its name and its declared arguments, if any.

    ``arguments`` mirrors :meth:`~pyippdme.server.registry.CommandRegistry.arguments`:
    ``None`` means no schema is on record (the command exists but its
    argument shape isn't declared - rare among built-in commands; see
    :meth:`~pyippdme.server.registry.CommandRegistry.register`), ``()``
    means it is declared and takes no arguments at all.
    """

    name: str
    arguments: tuple[Parameter, ...] | None


def build_command_catalog(
    command_classes: Sequence[Callable[[CommandRegistry], None]],
) -> Mapping[str, CommandInfo]:
    """Build a static catalog from ``command_classes``, without a live server or connection.

    Runs each ``register(registry)`` function - the single source of truth
    for every command's schema - against a throwaway
    :class:`~pyippdme.server.registry.CommandRegistry` that is discarded
    immediately after; no handler is ever invoked and no server/network
    state is created.
    """
    registry = CommandRegistry()
    for register in command_classes:
        register(registry)
    return MappingProxyType(
        {name: CommandInfo(name, registry.arguments(name)) for name in registry.names()}
    )
