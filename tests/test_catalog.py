# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for pyippdme.simulation.catalog: the static, offline command/argument catalog."""

from __future__ import annotations

import pytest

from pyippdme.protocol.commands import CommandName
from pyippdme.protocol.signature import DataType, Parameter
from pyippdme.server.catalog import CommandInfo, build_command_catalog
from pyippdme.server.classes import register_dme_class, register_server_class
from pyippdme.server.registry import CommandRegistry
from pyippdme.simulation import DEFAULT_COMMAND_CLASSES
from pyippdme.simulation.catalog import BUILTIN_COMMANDS


def test_builtin_commands_matches_a_freshly_built_registry() -> None:
    registry = CommandRegistry()
    for register in DEFAULT_COMMAND_CLASSES:
        register(registry)
    assert set(BUILTIN_COMMANDS) == set(registry.names())
    for name, info in BUILTIN_COMMANDS.items():
        assert info.name == name
        assert info.arguments == registry.arguments(name)


def test_builtin_commands_describes_go_to_s_arguments() -> None:
    info = BUILTIN_COMMANDS[CommandName.GO_TO]
    assert info.arguments is not None
    assert [p.name for p in info.arguments] == ["Positions", "Sync"]
    assert [p.datatype for p in info.arguments] == [DataType.ENUM, DataType.INT]


def test_builtin_commands_distinguishes_no_schema_from_empty_schema() -> None:
    # StartSession() is a real, fully-described zero-argument command.
    assert BUILTIN_COMMANDS[CommandName.START_SESSION].arguments == ()


def test_build_command_catalog_matches_a_restricted_command_class_subset() -> None:
    catalog = build_command_catalog([register_server_class, register_dme_class])
    assert CommandName.GET_DME_VERSION in catalog
    assert CommandName.GO_TO not in catalog


def test_build_command_catalog_is_read_only() -> None:
    catalog = build_command_catalog(DEFAULT_COMMAND_CLASSES)
    with pytest.raises(TypeError):
        catalog["Bogus"] = CommandInfo("Bogus", (Parameter("X", DataType.INT),))  # type: ignore[index]
