# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for pyippdme.server.dispatch: typed, self-describing command handlers."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from pyippdme import CommandName, IppDmeClient, IppDmeServer
from pyippdme.exceptions import IppDmeServerError
from pyippdme.protocol.ast import Argument, BasicName, NamedValue, Number, NumericData, String
from pyippdme.protocol.errors import ErrorCode
from pyippdme.protocol.signature import DataType, Parameter, positional_float_parameters
from pyippdme.server import builders
from pyippdme.server.classes import register_dme_class, register_server_class
from pyippdme.server.dispatch import (
    _python_name,
    bind_arguments,
    command,
    command_proprietary,
    register_object,
)
from pyippdme.server.registry import CommandContext, CommandRegistry, HandlerResult, MachineState


@pytest.mark.parametrize(
    ("wire_name", "expected"),
    [
        ("StepW", "step_w"),
        ("RT", "rt"),
        ("Dis", "dis"),
        ("Angle", "angle"),
        ("AngleBaseLength", "angle_base_length"),
        ("X", "x"),
    ],
)
def test_python_name_matches_real_parameter_names(wire_name: str, expected: str) -> None:
    assert _python_name(wire_name) == expected


def test_command_forbids_arguments_for_a_standard_command() -> None:
    with pytest.raises(TypeError, match="not yours to redeclare"):
        command(CommandName.GET_DME_VERSION, arguments=())


def test_command_requires_arguments_for_a_proprietary_name() -> None:
    with pytest.raises(TypeError, match="pass `arguments="):
        command("XXMyCommand")  # type: ignore[call-overload]


def test_command_derives_the_standard_schema_automatically() -> None:
    class Obj:
        @command(CommandName.SCAN_UNKNOWN_DENSITY)
        async def scan_unknown_density(
            self,
            ctx: CommandContext[Any],
            dis: float | None,
            angle: float | None,
            angle_base_length: float | None,
        ) -> HandlerResult:
            return None

    registry = CommandRegistry()
    register_object(registry, Obj())
    assert registry.arguments(CommandName.SCAN_UNKNOWN_DENSITY) == (
        Parameter("Dis", DataType.FLOAT, mandatory=False),
        Parameter("Angle", DataType.FLOAT, mandatory=False),
        Parameter("AngleBaseLength", DataType.FLOAT, mandatory=False),
    )


def test_register_object_only_registers_decorated_methods() -> None:
    class Partial:
        def __init__(self) -> None:
            self.calls: list[str] = []

        @command(CommandName.GET_DME_VERSION)
        async def get_dme_version(self, ctx: CommandContext[Any]) -> HandlerResult:
            return builders.get_dme_version("9.9")

        async def not_decorated(self, ctx: CommandContext[Any]) -> None:  # pragma: no cover
            self.calls.append("should never be found")

    registry = CommandRegistry()
    register_object(registry, Partial())
    assert registry.names() == (CommandName.GET_DME_VERSION,)


def test_registration_rejects_a_typed_method_with_the_wrong_parameter_type() -> None:
    class BadScanning:
        @command(CommandName.SCAN_ON_LINE)  # type: ignore[arg-type]
        async def scan_on_line(
            self,
            ctx: CommandContext[Any],
            sx: int,  # wrong: should be float
            sy: float,
            sz: float,
            ex: float,
            ey: float,
            ez: float,
            i: float,
            j: float,
            k: float,
            step_w: float,
            rt: bool | None,
        ) -> AsyncIterator[NumericData]:
            yield NumericData(())

    with pytest.raises(TypeError, match="expected"):
        register_object(CommandRegistry(), BadScanning())


def test_registration_rejects_typed_parameters_not_matching_the_schema_names() -> None:
    class WrongNames:
        @command(CommandName.SCAN_UNKNOWN_DENSITY)  # type: ignore[arg-type]
        async def scan_unknown_density(
            self, ctx: CommandContext[Any], wrong_name: float | None
        ) -> HandlerResult:
            return None

    with pytest.raises(TypeError, match="don't match"):
        register_object(CommandRegistry(), WrongNames())


def test_raw_handler_style_is_registered_unconverted() -> None:
    """A command whose only parameter is `args: tuple[Argument, ...]` bypasses binding."""

    class RawGet:
        @command(CommandName.GET)
        async def get(self, ctx: CommandContext[Any], args: tuple[Argument, ...]) -> HandlerResult:
            return builders.bare_names(*(a.name for a in args if hasattr(a, "name")))

    registry = CommandRegistry()
    register_object(registry, RawGet())
    assert registry.names() == (CommandName.GET,)
    # GET's real schema (dynamic arity, ENUM) is still derived automatically.
    assert registry.arguments(CommandName.GET) is not None


def test_registration_rejects_a_typed_parameter_for_an_enum_datatype() -> None:
    """ENUM means "a nested group", which bind_arguments() has no scalar to convert to."""

    class BadEnum:
        @command(CommandName.GO_TO)  # type: ignore[arg-type]  # ENUM ("Positions") parameter
        async def go_to(
            self, ctx: CommandContext[Any], positions: str, sync: int | None
        ) -> HandlerResult:
            return None

    with pytest.raises(TypeError, match="not a scalar type"):
        register_object(CommandRegistry(), BadEnum())


def test_bind_arguments_with_no_parameters_and_no_args() -> None:
    assert bind_arguments((), (), "Test") == {}


def test_bind_arguments_with_no_parameters_but_extra_args_raises() -> None:
    with pytest.raises(Exception, match="Expected no arguments"):
        bind_arguments((), (Number.of(1),), "Test")


def test_bind_positional_rejects_too_few_arguments() -> None:
    parameters = positional_float_parameters("Sx", "Sy", "Sz")
    with pytest.raises(Exception, match="Expected 3-3 arguments"):
        bind_arguments(parameters, (Number.of(1), Number.of(2)), "Test")


def test_bind_positional_rejects_too_many_arguments() -> None:
    parameters = positional_float_parameters("Sx", "Sy")
    with pytest.raises(Exception, match="Expected 2-2 arguments"):
        bind_arguments(parameters, (Number.of(1), Number.of(2), Number.of(3)), "Test")


def test_bind_positional_converts_int_and_bool_datatypes() -> None:
    parameters = (
        Parameter("Sync", DataType.INT, positional=True),
        Parameter("Closed", DataType.BOOL, positional=True),
    )
    bound = bind_arguments(parameters, (Number.of(5), Number.of(1)), "Test")
    assert bound == {"sync": 5, "closed": True}
    assert isinstance(bound["sync"], int)


def test_bind_positional_rejects_wrong_wire_type() -> None:
    parameters = positional_float_parameters("Sx")
    with pytest.raises(Exception, match="Expected a number"):
        bind_arguments(parameters, (String("not a number"),), "Test")


def test_bind_named_rejects_a_bare_positional_argument() -> None:
    parameters = (Parameter("Dis", DataType.FLOAT, mandatory=False),)
    with pytest.raises(Exception, match="Expected named arguments"):
        bind_arguments(parameters, (Number.of(1),), "Test")


def test_bind_named_rejects_a_wrong_argument_count_inside_a_group() -> None:
    parameters = (Parameter("Dis", DataType.FLOAT, mandatory=False),)
    with pytest.raises(Exception, match=r"Expected Dis\(<value>\)"):
        bind_arguments(parameters, (NamedValue("Dis", (Number.of(1), Number.of(2))),), "Test")


def test_bind_named_converts_string_and_name_datatypes() -> None:
    parameters = (
        Parameter("Text", DataType.STRING, mandatory=False),
        Parameter("Kind", DataType.NAME, mandatory=False),
    )
    bound = bind_arguments(
        parameters,
        (NamedValue("Text", (String("hi"),)), NamedValue("Kind", (BasicName("Foo"),))),
        "Test",
    )
    assert bound == {"text": "hi", "kind": "Foo"}


def test_bind_named_rejects_a_missing_mandatory_argument() -> None:
    parameters = (Parameter("Text", DataType.STRING, mandatory=True),)
    with pytest.raises(Exception, match="Missing mandatory argument Text"):
        bind_arguments(parameters, (), "Test")


def test_bind_named_rejects_wrong_wire_type_for_string_and_name() -> None:
    parameters = (Parameter("Text", DataType.STRING, mandatory=False),)
    with pytest.raises(Exception, match="Expected a string"):
        bind_arguments(parameters, (NamedValue("Text", (Number.of(1),)),), "Test")

    name_parameters = (Parameter("Kind", DataType.NAME, mandatory=False),)
    with pytest.raises(Exception, match="Expected a name"):
        bind_arguments(name_parameters, (NamedValue("Kind", (Number.of(1),)),), "Test")


async def test_bad_wire_arguments_surface_as_a_clean_protocol_error() -> None:
    """A ServerError raised from bind_arguments() surfaces as a clean protocol error."""

    class Bad:
        @command(CommandName.SCAN_UNKNOWN_DENSITY)
        async def scan_unknown_density(
            self,
            ctx: CommandContext[Any],
            dis: float | None,
            angle: float | None,
            angle_base_length: float | None,
        ) -> HandlerResult:
            return None

    server, client = await _connected(Bad())
    try:
        with pytest.raises(IppDmeServerError) as excinfo:
            await client.call(CommandName.SCAN_UNKNOWN_DENSITY, Number.of(1))  # bare, not named
        assert excinfo.value.error.number == ErrorCode.BAD_ARGUMENT
    finally:
        await _teardown(client, server)


class _Scanning:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    @command(CommandName.GET_DME_VERSION)
    async def get_dme_version(self, ctx: CommandContext[Any]) -> HandlerResult:
        return builders.get_dme_version("9.9")

    @command(CommandName.SCAN_ON_LINE)
    async def scan_on_line(
        self,
        ctx: CommandContext[Any],
        sx: float,
        sy: float,
        sz: float,
        ex: float,
        ey: float,
        ez: float,
        i: float,
        j: float,
        k: float,
        step_w: float,
        rt: bool | None,
    ) -> AsyncIterator[NumericData]:
        self.calls.append(("scan_on_line", sx, sy, sz, ex, ey, ez, step_w, rt))
        yield NumericData((Number.of(sx), Number.of(sy), Number.of(sz)))
        yield NumericData((Number.of(ex), Number.of(ey), Number.of(ez)))

    @command(CommandName.SCAN_UNKNOWN_DENSITY)
    async def scan_unknown_density(
        self,
        ctx: CommandContext[Any],
        dis: float | None,
        angle: float | None,
        angle_base_length: float | None,
    ) -> HandlerResult:
        self.calls.append(("scan_unknown_density", dis, angle, angle_base_length))
        return None

    # A proprietary command's `arguments=` is itself a runtime value, so
    # mypy can't statically verify it matches this method's typed
    # parameters the way the specific CommandName overloads can for a
    # standard command - see the dispatch module docstring. This still
    # works correctly at runtime (verified below); only the static check
    # is unavailable here, and only here.
    @command_proprietary(  # type: ignore[arg-type]
        "XX", "GetSensorTemp", arguments=(Parameter("SensorId", DataType.INT, positional=True),)
    )
    async def get_sensor_temp(self, ctx: CommandContext[Any], sensor_id: int) -> HandlerResult:
        return builders.number(sensor_id * 1.5)


async def _connected(obj: object) -> tuple[IppDmeServer[MachineState], IppDmeClient]:
    server: IppDmeServer[MachineState] = IppDmeServer(
        command_classes=[register_server_class, register_dme_class]
    )
    register_object(server.registry, obj)
    port = await server.start("127.0.0.1", 0)
    client = await IppDmeClient.connect("127.0.0.1", port)
    await client.start_session()
    return server, client


async def _teardown(client: IppDmeClient, server: IppDmeServer[MachineState]) -> None:
    await client.close()
    await server.close()


async def test_get_supported_commands_matches_the_registered_object() -> None:
    scanning = _Scanning()
    server, client = await _connected(scanning)
    try:
        results = await client.call(CommandName.GET_SUPPORTED_COMMANDS)
        names = {r.value.value for r in results}  # type: ignore[union-attr]
        assert {
            CommandName.GET_DME_VERSION,
            CommandName.SCAN_ON_LINE,
            CommandName.SCAN_UNKNOWN_DENSITY,
            "XXGetSensorTemp",
        } <= names
        assert CommandName.SCAN_ON_CIRCLE not in names  # never registered by _Scanning
    finally:
        await _teardown(client, server)


async def test_positional_binding_end_to_end() -> None:
    scanning = _Scanning()
    server, client = await _connected(scanning)
    try:
        data = await client.call(
            CommandName.SCAN_ON_LINE, *(Number.of(v) for v in (0, 0, 0, 10, 0, 0, 0, 0, 1, 5))
        )
        assert data == (
            NumericData((Number.of(0), Number.of(0), Number.of(0))),
            NumericData((Number.of(10), Number.of(0), Number.of(0))),
        )
        assert scanning.calls == [("scan_on_line", 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 5.0, None)]
    finally:
        await _teardown(client, server)


async def test_named_binding_with_some_optional_arguments_omitted() -> None:
    scanning = _Scanning()
    server, client = await _connected(scanning)
    try:
        await client.call(CommandName.SCAN_UNKNOWN_DENSITY)
        assert scanning.calls == [("scan_unknown_density", None, None, None)]
    finally:
        await _teardown(client, server)


async def test_proprietary_command_end_to_end() -> None:
    server, client = await _connected(_Scanning())
    try:
        (temperature,) = await client.call("XXGetSensorTemp", Number.of(3))
        assert temperature == NumericData((Number.of(4.5),))
    finally:
        await _teardown(client, server)


async def test_a_command_not_on_the_object_is_unsupported() -> None:
    server, client = await _connected(_Scanning())
    try:
        with pytest.raises(IppDmeServerError) as excinfo:
            await client.call(CommandName.SCAN_ON_CIRCLE, *[])
        assert excinfo.value.error.number == "0501"
    finally:
        await _teardown(client, server)


async def test_unsupported_named_argument_is_rejected() -> None:
    server, client = await _connected(_Scanning())
    try:
        with pytest.raises(IppDmeServerError) as excinfo:
            await client.call(
                CommandName.SCAN_UNKNOWN_DENSITY, NamedValue("NotARealField", (Number.of(1),))
            )
        assert excinfo.value.error.number == "0506"
    finally:
        await _teardown(client, server)
