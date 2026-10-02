# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for choosing/restricting which command classes a server supports.

A real machine rarely implements every command in the standard (e.g. no
full-surface/unknown-contour scanning support); VDMA 8722 already has a
mechanism for this (0501 "Unsupported command"), and pyippdme's job is just
to make it easy to build a server that only registers what it actually
supports.
"""

from __future__ import annotations

import pytest

from pyippdme import IppDmeClient, IppDmeServer
from pyippdme.exceptions import IppDmeServerError
from pyippdme.protocol.commands import CommandName
from pyippdme.server.classes import register_dme_class, register_server_class
from pyippdme.server.registry import CommandRegistry, MachineState, commands_of, register_subset
from pyippdme.simulation.classes import register_scanning_class


async def test_restricted_command_classes_omit_other_classes_commands() -> None:
    server: IppDmeServer[MachineState] = IppDmeServer(
        command_classes=[register_server_class, register_dme_class]
    )
    port = await server.start("127.0.0.1", 0)

    client = await IppDmeClient.connect("127.0.0.1", port)
    try:
        await client.start_session()
        (version,) = await client.call(CommandName.GET_DME_VERSION)  # DME class: present
        assert version is not None

        with pytest.raises(IppDmeServerError) as excinfo:
            await client.call(CommandName.GO_TO, *[])  # CartCMM class: intentionally not registered
        assert excinfo.value.error.number == "0501"
        await client.clear_all_errors()

        with pytest.raises(IppDmeServerError) as excinfo:
            await client.call(
                CommandName.SCAN_ON_LINE, *[]
            )  # Scanning class: intentionally not registered
        assert excinfo.value.error.number == "0501"
    finally:
        await client.close()
        await server.close()


async def test_get_supported_commands_reflects_the_restricted_set() -> None:
    server: IppDmeServer[MachineState] = IppDmeServer(
        command_classes=[register_server_class, register_dme_class]
    )
    port = await server.start("127.0.0.1", 0)

    client = await IppDmeClient.connect("127.0.0.1", port)
    try:
        await client.start_session()
        results = await client.call(CommandName.GET_SUPPORTED_COMMANDS)
        names = set()
        for r in results:
            names.add(r.value.value)  # type: ignore[union-attr]
        assert "GetDMEVersion" in names
        assert "GoTo" not in names
        assert "ScanOnLine" not in names
    finally:
        await client.close()
        await server.close()


def test_registry_unregister_removes_a_command() -> None:
    registry = CommandRegistry()

    async def _handler(ctx: object, args: object) -> None:  # pragma: no cover - never called
        return None

    registry.register("Foo", _handler)
    assert "Foo" in registry.names()
    registry.unregister("Foo")
    assert "Foo" not in registry.names()
    assert registry.get("Foo") is None


def test_registry_unregister_of_unknown_command_is_a_no_op() -> None:
    registry = CommandRegistry()
    registry.unregister("DoesNotExist")  # must not raise


def test_commands_of_matches_what_the_class_actually_registers() -> None:
    names = commands_of(register_scanning_class)
    assert CommandName.SCAN_ON_LINE in names
    assert CommandName.SCAN_ON_CIRCLE in names
    assert CommandName.SCAN_ON_HELIX in names
    assert CommandName.GO_TO not in names  # CartCMM, not Scanning


def test_commands_of_does_not_touch_a_real_registry() -> None:
    registry = CommandRegistry()
    commands_of(register_scanning_class)
    assert registry.names() == ()


def test_register_subset_keeps_only_the_requested_commands() -> None:
    registry = CommandRegistry()
    keep = {CommandName.SCAN_ON_LINE, CommandName.ON_SCAN_REPORT}
    register_subset(registry, register_scanning_class, keep=keep)
    assert CommandName.SCAN_ON_LINE in registry.names()
    assert CommandName.ON_SCAN_REPORT in registry.names()
    assert CommandName.SCAN_ON_CIRCLE not in registry.names()
    assert CommandName.SCAN_ON_HELIX not in registry.names()


def test_register_subset_leaves_other_already_registered_commands_alone() -> None:
    registry = CommandRegistry()
    register_server_class(registry)
    before = set(registry.names())
    register_subset(registry, register_scanning_class, keep={CommandName.SCAN_ON_LINE})
    assert before <= set(registry.names())  # nothing from Server got dropped


async def test_register_subset_reflects_correctly_in_get_supported_commands() -> None:
    server: IppDmeServer[MachineState] = IppDmeServer(
        command_classes=[register_server_class, register_dme_class]
    )
    register_subset(server.registry, register_scanning_class, keep={CommandName.SCAN_ON_LINE})
    port = await server.start("127.0.0.1", 0)

    client = await IppDmeClient.connect("127.0.0.1", port)
    try:
        await client.start_session()
        results = await client.call(CommandName.GET_SUPPORTED_COMMANDS)
        names = {r.value.value for r in results}  # type: ignore[union-attr]
        assert "ScanOnLine" in names
        assert "ScanOnCircle" not in names
        assert "ScanOnHelix" not in names

        with pytest.raises(IppDmeServerError) as excinfo:
            await client.call(CommandName.SCAN_ON_CIRCLE, *[])
        assert excinfo.value.error.number == "0501"
    finally:
        await client.close()
        await server.close()
