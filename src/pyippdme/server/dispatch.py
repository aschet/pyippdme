# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Typed, self-describing command handlers - an alternative to plain registration.

Every built-in command class (:mod:`pyippdme.server.classes`) is a module of free
functions taking the raw ``tuple[Argument, ...]`` and parsing it by hand.
That's fine for this library's own, already-tested handlers, but it means
anyone else implementing a command has to touch
:mod:`pyippdme.protocol.ast` directly, and separately has to keep whatever they
register in sync with what ``GetSupportedCommands()``/``GetSupportedArguments()``
(6.4.1) report about it.

This module is a second, additive way to write a handler, for exactly that
case:

* :func:`command`/:func:`command_proprietary` mark an ordinary method as the
  handler for a command name; :func:`register_object` finds every marked
  method on a plain object (no base class required - structural, like
  :class:`~pyippdme.server.backend.MachineBackend`) and registers exactly
  those, so ``GetSupportedCommands()`` is always derived from what you
  actually implemented, never a separately-maintained list.
* A method's own parameters can be plain typed values (``sx: float, ...``)
  instead of a raw ``tuple[Argument, ...]`` - :func:`bind_arguments` converts
  incoming wire arguments into them, using the same
  :class:`~pyippdme.protocol.signature.Parameter` schema already required for
  ``GetSupportedArguments``, so there is exactly one declared schema, not
  two. A method whose only parameter (besides ``ctx``) is still
  ``args: tuple[Argument, ...]`` is registered unconverted - for the
  handful of commands (dynamic arity, an ``enum``-typed argument, ...)
  whose shape doesn't fit a fixed, typed parameter list; see this module's
  own commands - ``GetDMEVersion``, ``ScanOnLine``, ``ScanUnknownDensity``
  are typed, ``GoTo`` (whose real schema is one opaque ``Positions`` enum,
  not fixed ``X``/``Y``/``Z`` parameters) is not, and never could be.
* For a standard :class:`~pyippdme.protocol.commands.CommandName`, ``arguments``
  is derived automatically from the schema this library already declares
  for it - the standard fixes that shape, it is not yours to redeclare, and
  :func:`command` raises if you try. A handful of standard commands
  (:data:`ScanOnLineHandler`, :data:`ScanUnknownDensityHandler`,
  :data:`GetDmeVersionHandler`) additionally get a statically-typed
  ``@overload`` of :func:`command`, so a wrong parameter type is a
  ``mypy``/editor error where you write it, not something only caught at
  server startup; every other standard command still works (registered,
  schema-derived, runtime-validated) through the generic
  ``CommandName`` overload, just without that extra static check yet.
  ``arguments`` is required for a proprietary name (:func:`command_proprietary`),
  since nothing could know its shape but you.

None of this touches the 9 existing built-in command classes - it is a
second way to write a *new* class (built-in or your own), not a
replacement for how the existing ones work.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, get_origin, get_type_hints, overload

from pyippdme.protocol.ast import Argument, BasicName, DataPayload, NamedValue, Number, String
from pyippdme.protocol.commands import CommandName
from pyippdme.protocol.errors import ErrorCode, ErrorSeverity, ServerError
from pyippdme.protocol.namespace import proprietary_name
from pyippdme.protocol.signature import DataType, Parameter
from pyippdme.server.registry import CommandContext, CommandHandler, CommandRegistry, HandlerResult

_MARKER = "__ippdme_command__"

#: Cache for :func:`_standard_arguments`, computed lazily on first use.
_standard_arguments_cache: CommandRegistry | None = None


def _standard_arguments() -> CommandRegistry:
    """Build (once, then cache) a registry of every built-in command, to read its ``arguments``.

    Not a second, hand-maintained schema: each ``register_*_class`` function's
    own body already is the schema (its ``arguments=`` keyword at each
    :meth:`~pyippdme.server.registry.CommandRegistry.register` call); this just
    reads it, the same way :func:`~pyippdme.server.registry.commands_of` reads
    a class's command names. The schema of a standard
    :class:`~pyippdme.protocol.commands.CommandName` is declared by the
    registration calls in :mod:`pyippdme.simulation`, so this imports it on
    first use rather than when this module is imported.
    """
    global _standard_arguments_cache
    if _standard_arguments_cache is None:
        from pyippdme.simulation import DEFAULT_COMMAND_CLASSES

        registry = CommandRegistry()
        for register_fn in DEFAULT_COMMAND_CLASSES:
            register_fn(registry)
        _standard_arguments_cache = registry
    return _standard_arguments_cache


_CAMEL_BOUNDARY_1 = re.compile(r"(.)([A-Z][a-z]+)")
_CAMEL_BOUNDARY_2 = re.compile(r"([a-z0-9])([A-Z])")


def _python_name(wire_name: str) -> str:
    """Convert a wire parameter name (``StepW``) to a Python one (``step_w``).

    A run of uppercase letters with nothing lowercase following (``RT``) is
    treated as one acronym, not split apart - matching this library's own
    parameter names (see :mod:`pyippdme.protocol.parameters`), which never mix
    an acronym mid-word the way this rule wouldn't handle.
    """
    with_boundaries = _CAMEL_BOUNDARY_1.sub(r"\1_\2", wire_name)
    return _CAMEL_BOUNDARY_2.sub(r"\1_\2", with_boundaries).lower()


def _bad_argument(cause: str, text: str = "Bad argument") -> ServerError:
    return ServerError(ErrorSeverity.CRITICAL, ErrorCode.BAD_ARGUMENT, cause, text)


def _incorrect_arguments(cause: str, text: str) -> ServerError:
    return ServerError(ErrorSeverity.CRITICAL, ErrorCode.INCORRECT_ARGUMENTS, cause, text)


_SCALAR_TYPES: dict[DataType, type] = {
    DataType.FLOAT: float,
    DataType.INT: int,
    DataType.BOOL: bool,
    DataType.STRING: str,
    DataType.NAME: str,
}


def _scalar_value(arg: Argument, datatype: DataType, cause: str) -> object:
    if datatype in (DataType.FLOAT, DataType.INT, DataType.BOOL):
        if not isinstance(arg, Number):
            raise _bad_argument(cause, "Expected a number")
        if datatype is DataType.INT:
            return int(arg.value)
        if datatype is DataType.BOOL:
            return arg.value != 0
        return arg.value
    if datatype is DataType.STRING:
        if not isinstance(arg, String):
            raise _bad_argument(cause, "Expected a string")
        return arg.value
    if datatype is DataType.NAME:
        if not isinstance(arg, BasicName):
            raise _bad_argument(cause, "Expected a name")
        return arg.value
    raise _bad_argument(cause, f"{datatype} arguments cannot be bound to a typed parameter")


def bind_arguments(
    parameters: tuple[Parameter, ...], args: tuple[Argument, ...], cause: str
) -> dict[str, object]:
    """Convert wire ``args`` into ``{python_name: value}`` per ``parameters``' own schema.

    ``parameters`` must not include an :attr:`~pyippdme.protocol.signature.DataType.ENUM`
    entry - that datatype means "a nested group, not a single scalar value"
    (see :class:`~pyippdme.protocol.signature.Parameter`'s docstring), which has no
    single Python type to bind to; a command with one keeps using the raw
    ``args: tuple[Argument, ...]`` handler style instead.
    """
    if not parameters:
        if args:
            raise _incorrect_arguments(cause, "Expected no arguments")
        return {}
    if parameters[0].positional:
        return _bind_positional(parameters, args, cause)
    return _bind_named(parameters, args, cause)


def _bind_positional(
    parameters: tuple[Parameter, ...], args: tuple[Argument, ...], cause: str
) -> dict[str, object]:
    mandatory = sum(1 for p in parameters if p.mandatory)
    if not (mandatory <= len(args) <= len(parameters)):
        raise _bad_argument(cause, f"Expected {mandatory}-{len(parameters)} arguments")
    bound: dict[str, object] = {}
    for p, arg in zip(parameters, args, strict=False):
        bound[_python_name(p.name)] = _scalar_value(arg, p.datatype, cause)
    for p in parameters[len(args) :]:
        bound[_python_name(p.name)] = None
    return bound


def _bind_named(
    parameters: tuple[Parameter, ...], args: tuple[Argument, ...], cause: str
) -> dict[str, object]:
    by_name: dict[str, NamedValue] = {}
    for arg in args:
        if not isinstance(arg, NamedValue):
            raise _bad_argument(cause, "Expected named arguments")
        by_name[arg.name] = arg
    bound: dict[str, object] = {}
    for p in parameters:
        matched = by_name.pop(p.name, None)
        if matched is None:
            if p.mandatory:
                raise _incorrect_arguments(cause, f"Missing mandatory argument {p.name}")
            bound[_python_name(p.name)] = None
            continue
        if len(matched.args) != 1:
            raise _bad_argument(cause, f"Expected {p.name}(<value>)")
        bound[_python_name(p.name)] = _scalar_value(matched.args[0], p.datatype, cause)
    if by_name:
        raise ServerError(
            ErrorSeverity.CRITICAL,
            ErrorCode.ARGUMENT_NOT_SUPPORTED,
            cause,
            f"Unsupported argument(s): {sorted(by_name)}",
        )
    return bound


@dataclass(frozen=True, slots=True)
class _CommandMarker:
    name: str
    arguments: tuple[Parameter, ...] | None


# --- command()/command_proprietary(): the public decorators -----------------
# Each specifically-typed overload below pins the decorator's expected
# callable shape to exactly that command's real, correct argument schema, so
# a mismatched parameter type is a static (mypy/editor) error. Every other
# standard command still works through the generic `CommandName` overload -
# registered, schema-derived, and validated at registration time - just
# without that extra static check. See the module docstring for why some
# standard commands (GoTo, Get, ...) can never get one.

# `self`'s position is typed `Any` (not `object`): Callable parameter types
# are contravariant, so `Obj.method`'s actual `self: Obj` would not be
# assignable to a slot declared `self: object` - `Any` is what escapes that
# check, the standard way to spell "some instance, whichever" here.
GetDmeVersionHandler: TypeAlias = Callable[[Any, CommandContext[Any]], Awaitable[HandlerResult]]
ScanOnLineHandler: TypeAlias = Callable[
    [
        Any,
        CommandContext[Any],
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        bool | None,
    ],
    AsyncIterator[DataPayload],
]
ScanUnknownDensityHandler: TypeAlias = Callable[
    [Any, CommandContext[Any], float | None, float | None, float | None], Awaitable[HandlerResult]
]
#: The shape of an *undecorated* method as written in a class body - unlike
#: :data:`~pyippdme.server.registry.CommandHandler`, which is what
#: :func:`register_object` ultimately hands to
#: :meth:`~pyippdme.server.registry.CommandRegistry.register` once ``self`` is
#: already bound away. Used by the two overloads below that don't pin a
#: specific command's own parameter types.
MethodHandler: TypeAlias = Callable[
    [Any, CommandContext[Any], tuple[Argument, ...]], Awaitable[HandlerResult]
]


@overload
def command(  # type: ignore[overload-overlap]
    name: Literal[CommandName.GET_DME_VERSION],
) -> Callable[[GetDmeVersionHandler], GetDmeVersionHandler]: ...
@overload
def command(  # type: ignore[overload-overlap]
    name: Literal[CommandName.SCAN_ON_LINE],
) -> Callable[[ScanOnLineHandler], ScanOnLineHandler]: ...
@overload
def command(  # type: ignore[overload-overlap]
    name: Literal[CommandName.SCAN_UNKNOWN_DENSITY],
) -> Callable[[ScanUnknownDensityHandler], ScanUnknownDensityHandler]: ...
@overload
def command(name: CommandName) -> Callable[[MethodHandler], MethodHandler]: ...
@overload
def command(
    name: str, *, arguments: tuple[Parameter, ...]
) -> Callable[[MethodHandler], MethodHandler]: ...
def command(
    name: CommandName | str, *, arguments: tuple[Parameter, ...] | None = None
) -> Callable[[Any], Any]:
    """Mark a method as the handler for ``name``; :func:`register_object` finds it later.

    See the module docstring for the rule on ``arguments``: never given for
    a standard :class:`~pyippdme.protocol.commands.CommandName` (derived
    automatically instead), always required otherwise. The specific
    overloads above are the only reason to prefer this over
    :meth:`~pyippdme.server.registry.CommandRegistry.command` directly - they
    pin a handful of standard commands' expected parameter types statically;
    this implementation's own signature is deliberately untyped (``Any``),
    since it is never what a caller's type checker actually sees - only the
    overloads above are, which is where the real guarantee comes from.
    """
    if isinstance(name, CommandName):
        if arguments is not None:
            raise TypeError(
                f"{name} is a standard command - its argument schema is fixed by "
                f"the standard, not yours to redeclare; omit `arguments=`."
            )
        arguments = _standard_arguments().arguments(name)
    elif arguments is None:
        raise TypeError(f"{name!r} is not a standard command name - pass `arguments=` yourself.")

    def decorator(func: Any) -> Any:
        setattr(func, _MARKER, _CommandMarker(name, arguments))
        return func

    return decorator


def command_proprietary(
    namespace: str, name: str, *, arguments: tuple[Parameter, ...]
) -> Callable[[MethodHandler], MethodHandler]:
    """Mark a method as the handler for a proprietary command (6.1).

    ``arguments`` is required - a proprietary command's shape is entirely
    your own, there is nothing to derive it from.
    """
    return command(proprietary_name(namespace, name), arguments=arguments)


# --- register_object(): find and register every @command-marked method -----


def _is_raw_handler(method: Callable[..., object]) -> bool:
    """Whether ``method``'s own signature already is ``(ctx, args: tuple[Argument, ...])``.

    ``ctx``'s own type parameter (``CommandContext[MachineState]``, a
    project's own ``Ctx`` alias, ...) is irrelevant here - only its generic
    origin matters, since a handler for any state type is still "raw" in
    the sense this function means.
    """
    hints = get_type_hints(method)
    hints.pop("return", None)
    values = list(hints.values())
    if len(values) != 2 or values[1] != tuple[Argument, ...]:
        return False
    ctx_hint = values[0]
    return ctx_hint is CommandContext or get_origin(ctx_hint) is CommandContext


def _validate_typed_signature(
    method: Callable[..., object], parameters: tuple[Parameter, ...], cause: str
) -> None:
    hints = get_type_hints(method)
    hints.pop("return", None)
    ctx_name, *param_names = hints.keys()
    expected_names = [_python_name(p.name) for p in parameters]
    if param_names != expected_names:
        raise TypeError(
            f"{cause}: handler parameters {param_names} don't match its own "
            f"declared schema {expected_names}"
        )
    for p, py_name in zip(parameters, expected_names, strict=True):
        if p.datatype not in _SCALAR_TYPES:
            raise TypeError(
                f"{cause}: parameter {py_name!r} is {p.datatype} - not a scalar type "
                f"bind_arguments() can convert; use the raw args: tuple[Argument, ...] style"
            )
        expected = _SCALAR_TYPES[p.datatype]
        if not p.mandatory:
            expected = expected | None  # type: ignore[assignment]
        if hints[py_name] != expected:
            raise TypeError(
                f"{cause}: parameter {py_name!r} is {hints[py_name]!r}, expected {expected!r}"
            )
    del ctx_name


def _wrap(method: Callable[..., object], marker: _CommandMarker) -> CommandHandler:
    if _is_raw_handler(method):
        return method  # type: ignore[return-value]

    parameters = marker.arguments or ()
    _validate_typed_signature(method, parameters, marker.name)
    streaming = inspect.isasyncgenfunction(method)

    async def wrapper(ctx: CommandContext[Any], args: tuple[Argument, ...]) -> HandlerResult:
        bound = bind_arguments(parameters, args, marker.name)
        if streaming:
            # An async-generator method (e.g. ScanOnLine) returns its
            # AsyncIterator directly when called - awaiting that call itself
            # (rather than iterating what it returns) is a TypeError.
            return method(ctx, **bound)  # type: ignore[return-value]
        return await method(ctx, **bound)  # type: ignore[misc,no-any-return]

    return wrapper


def register_object(registry: CommandRegistry, obj: object) -> None:
    """Register every ``@command``/``@command_proprietary``-marked method found on ``obj``.

    ``GetSupportedCommands()``/``GetSupportedArguments()`` (6.4.1) reflect
    exactly what this registers - there is no separate capability
    declaration anywhere else to keep in sync (see the module docstring).
    A typed method's parameters are checked against its own declared schema
    immediately, so a mismatch fails at server startup, not at a client's
    first call.
    """
    for attr_name in dir(obj):
        if attr_name.startswith("_"):
            continue
        method = getattr(obj, attr_name, None)
        if method is None:
            continue
        marker = getattr(method, _MARKER, None)
        if not isinstance(marker, _CommandMarker):
            continue
        registry.register(marker.name, _wrap(method, marker), arguments=marker.arguments)
