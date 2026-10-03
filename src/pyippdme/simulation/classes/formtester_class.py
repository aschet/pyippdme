# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""A subset of the ``FormTester`` command class (6.6.1).

``CenterPart``/``TiltPart``/``TiltCenterPart`` are alignment commands meant
to be called iteratively by the client against a real form tester's rotary
table/spindle: each measures how far a part still is from being centered
or tilted into the rotation axis (assumed, per 6.6, to be the Z-direction)
and returns ``0`` (moved, try again) or ``1`` (already within tolerance) as
a stop criterion. There is no real mechanical part-alignment stage here, so
this reports the threshold check directly from the given points/direction
rather than actually moving anything - a server with a real rotary
table/spindle would replace these handlers to drive it.

``LockAxis``/``LockPosition`` are the interesting pair: they must make
``GoTo``/``PtMeas``/the ``Scanning`` commands silently ignore motion on
locked axes rather than erroring (6.6.1: "Using LockAxis(X(), Y())
will disable any movement of the X and Y axis... without causing an
error"), so ``MachineState.locked_axes`` is read directly by
:mod:`pyippdme.simulation.classes.cartcmm_class`'s ``GoTo``/``PtMeas`` handlers. Only
the ``X``/``Y``/``Z`` axes are enforced, since those are the only axes this
project's simulated machine moves at all; ``R``/``A``/``B``/``C`` are
accepted and stored but have no rotary table or tool orientation to lock.
Likewise, ``LockPosition``'s ``XFR``/``YFR``/``ZFR``/``RFR``/``PFR``
(RotaryTableFixCsy-relative) names are validated and stored, including the
"conflicting combination" check, but have no observable effect on motion
since this project does not implement ``RotaryTableFixCsy`` at all.

Note: 6.6.1's own command tables cite errors "1010 Unable to
move" and "1011 Bad lock combination", but Annex B (the section explicitly
marked normative) numbers these 1011 and 1012 respectively. This module
follows Annex B, consistent with how this project resolves every other
such internal inconsistency in the standard.
"""

from __future__ import annotations

import math

from pyippdme.protocol.ast import Argument, NamedValue
from pyippdme.protocol.commands import CommandName
from pyippdme.protocol.errors import ErrorCode, ErrorSeverity, ServerError
from pyippdme.protocol.signature import (
    DataType,
    Parameter,
    argument_count_bounds,
    positional_float_parameters,
)
from pyippdme.server import builders
from pyippdme.server._util import bad_argument, positional_numbers
from pyippdme.server.registry import CommandRegistry, HandlerResult
from pyippdme.simulation.context import Ctx
from pyippdme.types.vec3 import Vec3, dot, normalize

_CENTER_PART_PARAMS = positional_float_parameters("Px", "Py", "Pz", "Limit")
_TILT_PART_PARAMS = positional_float_parameters("Dx", "Dy", "Dz", "Limit")
_TILT_CENTER_PART_PARAMS = positional_float_parameters(
    "Px1", "Py1", "Pz1", "Px2", "Py2", "Pz2", "Limit"
)
_LOCK_AXIS_PARAMS = (Parameter("Axis", DataType.ENUM),)
_LOCK_POSITION_PARAMS = (Parameter("Position", DataType.ENUM),)

#: Axes LockAxis(...) accepts (6.6.1); only X/Y/Z are enforced (see module docstring).
_LOCKABLE_AXES = frozenset({"X", "Y", "Z", "R", "A", "B", "C"})
#: The axes this project's simulated CartCMM actually moves.
_ENFORCED_AXES = frozenset({"X", "Y", "Z"})
#: Positions LockPosition(...) accepts (6.6.1 example).
_LOCKABLE_POSITIONS = frozenset({"XFR", "YFR", "ZFR", "RFR", "PFR"})
_CARTESIAN_POSITIONS = frozenset({"XFR", "YFR"})
_CYLINDRICAL_POSITIONS = frozenset({"RFR", "PFR"})


async def _center_part(_ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
    px, py, _pz, limit = positional_numbers(
        args, CommandName.CENTER_PART, *argument_count_bounds(_CENTER_PART_PARAMS)
    )
    return builders.boolean(math.hypot(px, py) < limit)


async def _tilt_part(_ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
    dx, dy, dz, limit = positional_numbers(
        args, CommandName.TILT_PART, *argument_count_bounds(_TILT_PART_PARAMS)
    )
    return builders.boolean(_within_tilt_limit((dx, dy, dz), limit))


async def _tilt_center_part(_ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
    bounds = argument_count_bounds(_TILT_CENTER_PART_PARAMS)
    px1, py1, _pz1, px2, py2, _pz2, limit = positional_numbers(
        args, CommandName.TILT_CENTER_PART, *bounds
    )
    centered = math.hypot(px1, py1) < limit and math.hypot(px2, py2) < limit
    return builders.boolean(centered)


def _within_tilt_limit(direction: Vec3, limit: float) -> bool:
    """Check whether ``direction`` is within ``limit`` mm of the +Z axis over a 100mm baseline."""
    try:
        unit = normalize(direction)
    except ValueError as exc:
        raise bad_argument(CommandName.TILT_PART, "Direction vector must be non-zero") from exc
    # The rotation axis is a line, not a directed ray, so fold to [0, 90] degrees.
    cos_angle = min(1.0, abs(dot(unit, (0.0, 0.0, 1.0))))
    angle = math.acos(cos_angle)
    lateral_deviation_at_100mm = 100.0 * math.tan(angle)
    return lateral_deviation_at_100mm < limit


async def _lock_axis(ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
    locked: set[str] = set()
    for arg in args:
        if not isinstance(arg, NamedValue) or arg.args:
            raise ServerError(
                ErrorSeverity.CRITICAL,
                ErrorCode.BAD_PROPERTY,
                CommandName.LOCK_AXIS,
                "Bad property",
            )
        if arg.name not in _LOCKABLE_AXES:
            raise ServerError(
                ErrorSeverity.CRITICAL,
                ErrorCode.ARGUMENT_NOT_RECOGNIZED,
                CommandName.LOCK_AXIS,
                f"Unknown axis {arg.name}",
            )
        locked.add(arg.name)
    if locked >= _ENFORCED_AXES:
        raise ServerError(
            ErrorSeverity.ERROR, ErrorCode.UNABLE_TO_MOVE, CommandName.LOCK_AXIS, "Unable to move"
        )
    ctx.state.form_tester.locked_axes = frozenset(locked)
    # 6.6.1: "All positions are unlocked after a LockAxis with one of the (X(), Y(), Z())
    # arguments."
    if locked & _ENFORCED_AXES:
        ctx.state.form_tester.locked_positions = frozenset()
    return None


async def _lock_position(ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
    locked: set[str] = set()
    for arg in args:
        if not isinstance(arg, NamedValue) or arg.args:
            raise ServerError(
                ErrorSeverity.CRITICAL,
                ErrorCode.BAD_PROPERTY,
                CommandName.LOCK_POSITION,
                "Bad property",
            )
        if arg.name not in _LOCKABLE_POSITIONS:
            raise ServerError(
                ErrorSeverity.CRITICAL,
                ErrorCode.ARGUMENT_NOT_RECOGNIZED,
                CommandName.LOCK_POSITION,
                f"Unknown position {arg.name}",
            )
        locked.add(arg.name)
    if locked & _CARTESIAN_POSITIONS and locked & _CYLINDRICAL_POSITIONS:
        raise ServerError(
            ErrorSeverity.ERROR,
            ErrorCode.BAD_LOCK_COMBINATIONS,
            CommandName.LOCK_POSITION,
            "Bad lock combinations",
        )
    ctx.state.form_tester.locked_positions = frozenset(locked)
    # Per 6.6.1: X, Y, and Z axis locks are cleared by a LockPosition call.
    ctx.state.form_tester.locked_axes = ctx.state.form_tester.locked_axes - _ENFORCED_AXES
    return None


def register(registry: CommandRegistry) -> None:
    registry.register(CommandName.CENTER_PART, _center_part, arguments=_CENTER_PART_PARAMS)
    registry.register(CommandName.TILT_PART, _tilt_part, arguments=_TILT_PART_PARAMS)
    registry.register(
        CommandName.TILT_CENTER_PART, _tilt_center_part, arguments=_TILT_CENTER_PART_PARAMS
    )
    registry.register(CommandName.LOCK_AXIS, _lock_axis, arguments=_LOCK_AXIS_PARAMS)
    registry.register(CommandName.LOCK_POSITION, _lock_position, arguments=_LOCK_POSITION_PARAMS)
