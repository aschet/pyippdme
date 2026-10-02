# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""The ``DME`` command class (section 6.4), mandatory for all servers."""

from __future__ import annotations

from pyippdme.protocol.ast import Argument, String
from pyippdme.protocol.commands import CommandName
from pyippdme.protocol.errors import ErrorCode, ErrorSeverity, ServerError
from pyippdme.protocol.signature import DataType, Parameter
from pyippdme.server import builders
from pyippdme.server.context import Ctx
from pyippdme.server.registry import CommandRegistry, HandlerResult

_GET_SUPPORTED_ARGUMENTS_PARAMS = (Parameter("CommandName", DataType.STRING, positional=True),)


async def _get_dme_version(ctx: Ctx, _args: tuple[Argument, ...]) -> HandlerResult:
    return builders.get_dme_version(ctx.state.dme_version)


async def _get_supported_commands(ctx: Ctx, _args: tuple[Argument, ...]) -> HandlerResult:
    return builders.string_list(ctx.registry.names())


async def _get_supported_arguments(ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
    if len(args) != 1 or not isinstance(args[0], String):
        raise ServerError(
            ErrorSeverity.CRITICAL,
            ErrorCode.BAD_ARGUMENT,
            CommandName.GET_SUPPORTED_ARGUMENTS,
            "Expected a single command name",
        )
    command_name = args[0].value
    parameters = ctx.registry.arguments(command_name)
    if parameters is None:
        raise ServerError(
            ErrorSeverity.CRITICAL,
            ErrorCode.ARGUMENT_NOT_SUPPORTED,
            CommandName.GET_SUPPORTED_ARGUMENTS,
            "Argument not supported",
        )
    return builders.supported_arguments(parameters)


async def _get_machine_class(ctx: Ctx, _args: tuple[Argument, ...]) -> HandlerResult:
    return builders.string_value(ctx.state.machine_class)


async def _home(ctx: Ctx, _args: tuple[Argument, ...]) -> HandlerResult:
    ctx.state.homed = True
    # No further state of its own - a class with a documented "Home()
    # triggers this" side effect (Mover 6.7.1's implicit DisableUser())
    # registers its own hook instead.
    for hook in ctx.registry.home_hooks():
        hook(ctx)
    return None


async def _is_homed(ctx: Ctx, _args: tuple[Argument, ...]) -> HandlerResult:
    return builders.named_boolean(CommandName.IS_HOMED, ctx.state.homed)


def register(registry: CommandRegistry) -> None:
    registry.register(CommandName.GET_DME_VERSION, _get_dme_version, arguments=())
    registry.register(CommandName.GET_SUPPORTED_COMMANDS, _get_supported_commands, arguments=())
    registry.register(
        CommandName.GET_SUPPORTED_ARGUMENTS,
        _get_supported_arguments,
        arguments=_GET_SUPPORTED_ARGUMENTS_PARAMS,
    )
    registry.register(CommandName.GET_MACHINE_CLASS, _get_machine_class, arguments=())
    registry.register(CommandName.HOME, _home, arguments=())
    registry.register(CommandName.IS_HOMED, _is_homed, arguments=())
