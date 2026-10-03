# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""A subset of the ``RotaryTable`` command class (6.23).

``R()``/``R(r)`` (6.23.1) are implemented in
:mod:`pyippdme.simulation.classes.cartcmm_class`, since the standard only allows them
as arguments of ``Get``/``OnReport``/``GoTo``/``PtMeas`` - the same commands
that already handle ``X``/``Y``/``Z``, never as a standalone command name.
This module adds ``EnableRotaryTableVarCsy`` and a simplified ``AlignPart``.

Scope note on ``AlignPart``: the standard does not say which machine axis a
rotary table turns about; this simulation assumes the first table turns about
the Z axis (the table plane is the machine's XY plane), which is the
conventional CMM rotary-table configuration, and the optional second table,
which the standard says is orthogonal to the first, about the X axis.
"""

from __future__ import annotations

import math

from pyippdme.protocol.ast import Argument, Number, NumericData
from pyippdme.protocol.commands import CommandName
from pyippdme.protocol.errors import ErrorCode, ErrorSeverity, ServerError
from pyippdme.protocol.signature import DataType, Parameter, positional_float_parameters
from pyippdme.server import builders
from pyippdme.server._util import bad_argument, positional_numbers
from pyippdme.server.registry import CommandRegistry, HandlerResult
from pyippdme.simulation.context import Ctx
from pyippdme.types.vec3 import Vec3

_ENABLE_PARAMS = (Parameter("Bool", DataType.BOOL, positional=True),)
_ALIGN_PART_PARAMS = (
    *positional_float_parameters("px1", "py1", "pz1", "mx1", "my1", "mz1"),
    *(
        Parameter(name, DataType.FLOAT, mandatory=False, positional=True)
        for name in ("px2", "py2", "pz2", "mx2", "my2", "mz2")
    ),
    Parameter("alpha", DataType.FLOAT, positional=True),
    Parameter("beta", DataType.FLOAT, mandatory=False, positional=True),
)


async def _enable_rotary_table_var_csy(ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
    if len(args) != 1 or not isinstance(args[0], Number):
        raise ServerError(
            ErrorSeverity.CRITICAL,
            ErrorCode.BAD_ARGUMENT,
            CommandName.ENABLE_ROTARY_TABLE_VAR_CSY,
            "Expected a single boolean",
        )
    ctx.state.rotary_table.var_csy_enabled = args[0].value != 0
    return None


def apply_part_alignment(ctx: Ctx, numbers: tuple[float, ...], cause: str) -> NumericData:
    """Turn the rotary table(s) to the orientation ``AlignPart(...)`` asks for (6.23.1).

    ``numbers`` are ``px1..mz1, alpha`` for one table, or
    ``px1..mz1, px2..mz2, alpha, beta`` for two. The first table turns about
    the machine Z axis, the second, orthogonal one about the X axis.
    """
    if len(numbers) not in (7, 14):
        raise bad_argument(cause, "Expected 7 numbers, or 14 for a second rotary table")
    two_tables = len(numbers) == 14
    part_vec: Vec3 = (numbers[0], numbers[1], numbers[2])
    machine_vec: Vec3 = (numbers[3], numbers[4], numbers[5])
    alpha = numbers[-2] if two_tables else numbers[-1]
    part_xy = _normalize_plane(part_vec, 0, 1, cause)
    machine_xy = _normalize_plane(machine_vec, 0, 1, cause)

    _check_alignment(part_xy, machine_xy, alpha, cause)
    ctx.state.rotary_table.position = _rotation(part_xy, machine_xy) % 360.0
    if not two_tables:
        return builders.align_part(part_xy, machine_xy)

    second_part: Vec3 = (numbers[6], numbers[7], numbers[8])
    second_machine: Vec3 = (numbers[9], numbers[10], numbers[11])
    beta = numbers[13]
    part_yz = _normalize_plane(second_part, 1, 2, cause)
    machine_yz = _normalize_plane(second_machine, 1, 2, cause)
    _check_alignment(part_yz, machine_yz, beta, cause)
    ctx.state.rotary_table.second_position = _rotation(part_yz, machine_yz) % 360.0
    return builders.align_part(part_xy, machine_xy, part_yz, machine_yz)


def _rotation(part: tuple[float, float], machine: tuple[float, float]) -> float:
    return math.degrees(math.atan2(machine[1], machine[0])) - math.degrees(
        math.atan2(part[1], part[0])
    )


def _check_alignment(
    part: tuple[float, float], machine: tuple[float, float], allowed_error: float, cause: str
) -> None:
    if allowed_error == 0.0:
        return  # "In case alpha is zero no error check is performed" (6.23.1).
    # By construction the table is rotated to exactly eliminate the angle
    # between the two projected vectors, so this only ever catches
    # floating-point residue - a real (non-simulated) table would have
    # actual mechanical error to check against alpha here.
    cosine = max(-1.0, min(1.0, part[0] * machine[0] + part[1] * machine[1]))
    if math.degrees(math.acos(cosine)) > allowed_error:
        raise ServerError(
            ErrorSeverity.ERROR, ErrorCode.PART_NOT_ALIGNED, cause, "Part not aligned"
        )


async def _align_part(ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
    numbers = positional_numbers(args, CommandName.ALIGN_PART, 7, 14)
    return apply_part_alignment(ctx, numbers, CommandName.ALIGN_PART)


def _normalize_plane(vector: Vec3, first: int, second: int, cause: str) -> tuple[float, float]:
    """Project ``vector`` onto the plane of two of its axes and normalize it."""
    u, v = vector[first], vector[second]
    length = math.hypot(u, v)
    if length == 0.0:
        raise ServerError(
            ErrorSeverity.ERROR,
            ErrorCode.VECTOR_HAS_NO_NORM,
            cause,
            "Vector has no norm after projection",
        )
    return u / length, v / length


def register(registry: CommandRegistry) -> None:
    registry.register(
        CommandName.ENABLE_ROTARY_TABLE_VAR_CSY,
        _enable_rotary_table_var_csy,
        arguments=_ENABLE_PARAMS,
    )
    registry.register(CommandName.ALIGN_PART, _align_part, arguments=_ALIGN_PART_PARAMS)
