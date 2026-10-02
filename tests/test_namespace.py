# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for the proprietary-extension namespace helper and registry integration (VDMA 8722 6.1)."""

from __future__ import annotations

import pytest

from pyippdme import IppDmeClient, IppDmeServer, proprietary_name
from pyippdme.protocol.ast import Argument, Number, PropertyData, String, StringValue
from pyippdme.protocol.signature import DataType, Parameter
from pyippdme.server.classes import register_dme_class, register_server_class
from pyippdme.server.context import Ctx
from pyippdme.server.registry import CommandRegistry, HandlerResult, MachineState


def test_proprietary_name_builds_the_prefixed_name() -> None:
    assert proprietary_name("XX", "ProprietaryCommandName") == "XXProprietaryCommandName"


@pytest.mark.parametrize("namespace", ["", "X", "XXX", "X1", "1X", "  "])
def test_proprietary_name_rejects_non_two_letter_namespaces(namespace: str) -> None:
    with pytest.raises(ValueError, match="two letters"):
        proprietary_name(namespace, "Foo")


def test_proprietary_name_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        proprietary_name("XX", "")


def test_registry_register_proprietary_uses_the_prefixed_name() -> None:
    registry = CommandRegistry()

    async def _handler(ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
        return None

    registry.register_proprietary("XX", "Foo", _handler)
    assert "XXFoo" in registry.names()
    assert registry.get("XXFoo") is _handler


def test_registry_command_proprietary_decorator() -> None:
    registry = CommandRegistry()

    @registry.command_proprietary("XX", "Foo")
    async def _handler(ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
        return None

    assert "XXFoo" in registry.names()


def test_registry_command_decorator_accepts_arguments() -> None:
    """The decorator forms accept the same ``arguments`` schema ``register()`` does."""
    registry = CommandRegistry()
    params = (Parameter("Value", DataType.INT),)

    @registry.command("Foo", arguments=params)
    async def _handler(ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
        return None

    assert registry.arguments("Foo") == params


def test_registry_command_proprietary_decorator_accepts_arguments() -> None:
    registry = CommandRegistry()
    params = (Parameter("Value", DataType.INT),)

    @registry.command_proprietary("XX", "Foo", arguments=params)
    async def _handler(ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
        return None

    assert registry.arguments("XXFoo") == params


async def test_proprietary_command_is_callable_over_the_wire() -> None:
    server: IppDmeServer[MachineState] = IppDmeServer(
        command_classes=[register_server_class, register_dme_class]
    )

    @server.registry.command_proprietary(
        "XX", "Echo", arguments=(Parameter("Value", DataType.INT),)
    )
    async def _echo(ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
        del ctx
        (value,) = args
        assert isinstance(value, Number)
        return StringValue(String(f"got {value.to_wire()}"))

    port = await server.start("127.0.0.1", 0)

    client = await IppDmeClient.connect("127.0.0.1", port)
    try:
        await client.start_session()
        (data,) = await client.call("XXEcho", Number.of(42))
        assert isinstance(data, StringValue)
        assert data.value.value == "got 42"

        # GetSupportedArguments describes it for real, not "0506 not supported".
        args_data = await client.call("GetSupportedArguments", String("XXEcho"))
        pairs = []
        for item in args_data:
            assert isinstance(item, PropertyData)
            pairs.append((item.first.value, item.second.value))
        assert pairs == [("Value", "int")]
    finally:
        await client.close()
        await server.close()
