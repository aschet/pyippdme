# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for GetSupportedArguments(CommandName) (VDMA 8722 6.4.1) and its schema plumbing."""

from __future__ import annotations

import pytest

from pyippdme import IppDmeClient
from pyippdme.exceptions import IppDmeServerError
from pyippdme.protocol.ast import PropertyData, String
from pyippdme.protocol.commands import CommandName
from pyippdme.protocol.signature import (
    DataType,
    Parameter,
    argument_count_bounds,
    positional_float_parameters,
)
from pyippdme.server.registry import CommandRegistry
from pyippdme.simulation.catalog import BUILTIN_COMMANDS


def _pairs(data: object) -> list[tuple[str, str]]:
    result = []
    for item in data:  # type: ignore[attr-defined]
        assert isinstance(item, PropertyData)
        result.append((item.first.value, item.second.value))
    return result


async def test_get_supported_arguments_for_scan_on_line(started_client: IppDmeClient) -> None:
    data = await started_client.call(
        CommandName.GET_SUPPORTED_ARGUMENTS, String(CommandName.SCAN_ON_LINE)
    )
    assert _pairs(data) == [
        ("Sx", "float"),
        ("Sy", "float"),
        ("Sz", "float"),
        ("Ex", "float"),
        ("Ey", "float"),
        ("Ez", "float"),
        ("i", "float"),
        ("j", "float"),
        ("k", "float"),
        ("StepW", "float"),
        ("RT", "bool"),
    ]


async def test_get_supported_arguments_for_go_to(started_client: IppDmeClient) -> None:
    data = await started_client.call(CommandName.GET_SUPPORTED_ARGUMENTS, String(CommandName.GO_TO))
    assert _pairs(data) == [("Positions", "enum"), ("Sync", "int")]


async def test_get_supported_arguments_for_set_coord_system(started_client: IppDmeClient) -> None:
    data = await started_client.call(
        CommandName.GET_SUPPORTED_ARGUMENTS, String(CommandName.SET_COORD_SYSTEM)
    )
    assert _pairs(data) == [("Csy", "name")]


async def test_get_supported_arguments_for_get_error_info(started_client: IppDmeClient) -> None:
    data = await started_client.call(
        CommandName.GET_SUPPORTED_ARGUMENTS, String(CommandName.GET_ERROR_INFO)
    )
    assert _pairs(data) == [("ErrorNumber", "int")]


async def test_get_supported_arguments_empty_for_a_real_zero_argument_command(
    started_client: IppDmeClient,
) -> None:
    # GetCoordSystem() is a real, registered command that takes no arguments
    # at all - GetSupportedArguments must say so (an empty list), not "not
    # supported" (0506), which would wrongly claim the command doesn't exist.
    # (register()'s `arguments=None` vs. `arguments=()` distinction - see
    # test_registry_arguments_returns_none_when_undeclared below for the
    # equivalent check directly against CommandRegistry.)
    data = await started_client.call(
        CommandName.GET_SUPPORTED_ARGUMENTS, String(CommandName.GET_COORD_SYSTEM)
    )
    assert data == ()


async def test_get_supported_arguments_unknown_command(started_client: IppDmeClient) -> None:
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(
            CommandName.GET_SUPPORTED_ARGUMENTS, String("ThisCommandDoesNotExist")
        )
    assert excinfo.value.error.number == "0506"


async def test_get_supported_arguments_requires_a_single_string(
    started_client: IppDmeClient,
) -> None:
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(CommandName.GET_SUPPORTED_ARGUMENTS)
    assert excinfo.value.error.number == "0509"


def test_argument_count_bounds_all_mandatory() -> None:
    params = positional_float_parameters("A", "B", "C")
    assert argument_count_bounds(params) == (3, 3)


def test_argument_count_bounds_with_trailing_optional() -> None:
    params = positional_float_parameters("A", "B", "C", optional=("C",))
    assert argument_count_bounds(params) == (2, 3)
    assert [p.mandatory for p in params] == [True, True, False]


def test_positional_float_parameters_marks_every_parameter_positional() -> None:
    """Kind U style, e.g. ScanOnLine(1, 2, 3, ...) - see Parameter.positional's docstring."""
    params = positional_float_parameters("A", "B", "C")
    assert all(p.positional for p in params)


def test_parameter_defaults_to_not_positional() -> None:
    assert Parameter("X", DataType.FLOAT).positional is False


def test_scan_on_line_s_hand_appended_rt_is_also_marked_positional() -> None:
    """RT is spliced onto ScanOnLine's schema by hand, not via positional_float_parameters.

    It isn't a float, but is still written as a bare value like the rest -
    it must carry the same ``positional=True`` or TUI argument completion
    would wrongly suggest it.
    """
    params = BUILTIN_COMMANDS[CommandName.SCAN_ON_LINE].arguments
    assert params is not None
    assert all(p.positional for p in params)


def test_scan_on_curve_is_not_positional() -> None:
    """Unlike its ScanOn... siblings, ScanOnCurve takes named arguments.

    ``Closed(...)``, ``Format(...)``, ``RT(...)``, ``Data(...)`` - not bare
    positional values.
    """
    params = BUILTIN_COMMANDS[CommandName.SCAN_ON_CURVE].arguments
    assert params is not None
    assert not any(p.positional for p in params)


def test_registry_arguments_returns_none_when_undeclared() -> None:
    registry = CommandRegistry()

    async def _handler(ctx: object, args: object) -> None:  # pragma: no cover - never called
        return None

    registry.register("Foo", _handler)
    assert registry.arguments("Foo") is None


def test_registry_arguments_returns_empty_tuple_for_known_zero_argument_command() -> None:
    """``arguments=()`` (a known, empty schema) must stay distinct from omitting it entirely."""
    registry = CommandRegistry()

    async def _handler(ctx: object, args: object) -> None:  # pragma: no cover - never called
        return None

    registry.register("Foo", _handler, arguments=())
    assert registry.arguments("Foo") == ()
    assert registry.arguments("Foo") is not None


def test_registry_arguments_returns_declared_schema() -> None:
    registry = CommandRegistry()
    params = (Parameter("X", DataType.FLOAT),)

    async def _handler(ctx: object, args: object) -> None:  # pragma: no cover - never called
        return None

    registry.register("Foo", _handler, arguments=params)
    assert registry.arguments("Foo") == params


def test_registry_unregister_also_clears_signature() -> None:
    registry = CommandRegistry()
    params = (Parameter("X", DataType.FLOAT),)

    async def _handler(ctx: object, args: object) -> None:  # pragma: no cover - never called
        return None

    registry.register("Foo", _handler, arguments=params)
    registry.unregister("Foo")
    assert registry.arguments("Foo") is None


#: Every built-in command whose argument(s) are written as bare values, not
#: Name(value) pairs - by hand, one handler at a time (see Parameter.positional's
#: docstring for why this can't be derived from datatype/count alone). Spans two
#: shapes: a fixed, ordered list of several bare values (positional_float_parameters,
#: e.g. ScanOnLine) and a single command whose lone argument isn't named at all
#: in its call (e.g. GetErrorInfo(506)).
_EXPECTED_POSITIONAL_COMMANDS = frozenset(
    {
        CommandName.SCAN_ON_CIRCLE_HINT,
        CommandName.SCAN_ON_LINE_HINT,
        CommandName.SCAN_ON_CURVE_HINT,
        CommandName.SCAN_UNKNOWN_HINT,
        CommandName.SCAN_ON_CIRCLE,
        CommandName.SCAN_ON_LINE,
        CommandName.SCAN_ON_HELIX,
        CommandName.SCAN_IN_PLANE_END_IS_SPHERE,
        CommandName.SCAN_IN_PLANE_END_IS_PLANE,
        CommandName.SCAN_IN_PLANE_END_IS_CYL,
        CommandName.SCAN_IN_CYL_END_IS_SPHERE,
        CommandName.SCAN_IN_CYL_END_IS_PLANE,
        CommandName.ALIGN_TOOL,
        CommandName.USE_SMALLEST_ANGLE_TO_ALIGN_TOOL,
        "UseSmallestAngleToAlignTool",  # the other spelling in the standard
        CommandName.CENTER_PART,
        CommandName.TILT_PART,
        CommandName.TILT_CENTER_PART,
        CommandName.ALIGN_PART,
        CommandName.SET_COORD_SYSTEM,
        CommandName.SET_CSY_TRANSFORMATION,
        CommandName.GET_CSY_TRANSFORMATION,
        CommandName.SAVE_NAMED_CSY_TRANSFORMATION,
        CommandName.GET_NAMED_CSY_TRANSFORMATION,
        CommandName.SAVE_ACTIVE_COORD_SYSTEM,
        CommandName.LOAD_COORD_SYSTEM,
        CommandName.DELETE_COORD_SYSTEM,
        CommandName.READ_TEMPERATURE_SENSOR,
        CommandName.GET_SUPPORTED_ARGUMENTS,
        CommandName.DATA_ACQUIRE,
        CommandName.RAW_DATA_BIN_SETUP,
        CommandName.GET_RAW_DATA_BIN,
        CommandName.GET_RAW_DATA_SHA_MEM,
        CommandName.RELEASE_SHA_MEM,
        CommandName.GET_RAW_DATA_FILE,
        CommandName.DEL_RAW_DATA_FILE,
        CommandName.DELETE_ACQUISITION,
        CommandName.ENABLE_ROTARY_TABLE_VAR_CSY,
        CommandName.STOP_DAEMON,
        CommandName.GET_ERROR_INFO,
        CommandName.CHANGE_TOOL,
        CommandName.FIND_TOOL,
        CommandName.SET_TOOL,
        CommandName.GET_CHANGE_TOOL_ACTION,
        CommandName.ENUM_TOOL_COLLECTION,
        CommandName.ENUM_ALL_TOOL_COLLECTIONS,
        CommandName.OPEN_TOOL_COLLECTION,
    }
)


def test_every_positional_command_matches_this_audit() -> None:
    """Locks in the by-hand handler audit behind Parameter.positional.

    Every command with arguments is either fully positional (every
    Parameter marked ``positional=True``) or fully named (none are) -
    no built-in command currently mixes the two conventions within one
    call. A command moving from one group to the other silently would
    break TUI argument completion (it would start suggesting, or stop
    suggesting, argument names that don't/do actually work) without any
    other test catching it, which is exactly what this guards against.
    """
    for name, info in BUILTIN_COMMANDS.items():
        if not info.arguments:
            continue
        positional_flags = {p.positional for p in info.arguments}
        assert len(positional_flags) == 1, (
            f"{name} mixes positional and named parameters: "
            f"{[(p.name, p.positional) for p in info.arguments]}"
        )
        is_positional = info.arguments[0].positional
        assert is_positional == (name in _EXPECTED_POSITIONAL_COMMANDS), (
            f"{name}: positional={is_positional}, "
            f"but audit expects {name in _EXPECTED_POSITIONAL_COMMANDS}"
        )
