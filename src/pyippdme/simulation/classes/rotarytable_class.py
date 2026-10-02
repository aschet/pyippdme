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
(single) rotary table turns about; this simulation assumes it is the Z axis
(the table plane is the machine's XY plane), which is the conventional CMM
rotary-table configuration. The optional second, orthogonal rotary table
(``px2``...``beta``) is not modeled - a request that includes it is rejected
with ``0506`` rather than silently ignored.
"""

from __future__ import annotations

import math

from pyippdme.protocol.ast import Argument, Number
from pyippdme.protocol.commands import CommandName
from pyippdme.protocol.errors import ErrorCode, ErrorSeverity, ServerError
from pyippdme.protocol.signature import DataType, Parameter, positional_float_parameters
from pyippdme.server import builders
from pyippdme.server._util import positional_numbers
from pyippdme.server.registry import CommandRegistry, HandlerResult
from pyippdme.simulation.context import Ctx
from pyippdme.types.vec3 import Vec3

_ENABLE_PARAMS = (Parameter("Bool", DataType.BOOL, positional=True),)
_ALIGN_PART_NAMES = ("px1", "py1", "pz1", "mx1", "my1", "mz1", "alpha")
_ALIGN_PART_PARAMS = positional_float_parameters(*_ALIGN_PART_NAMES)


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


async def _align_part(ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
    if len(args) != 7:
        raise ServerError(
            ErrorSeverity.CRITICAL,
            ErrorCode.ARGUMENT_NOT_SUPPORTED,
            CommandName.ALIGN_PART,
            "A second, orthogonal rotary table is not supported",
        )
    px1, py1, pz1, mx1, my1, mz1, alpha = positional_numbers(args, CommandName.ALIGN_PART, 7, 7)
    part_vec: Vec3 = (px1, py1, pz1)
    machine_vec: Vec3 = (mx1, my1, mz1)
    part_xy = _normalize_xy(part_vec, CommandName.ALIGN_PART)
    machine_xy = _normalize_xy(machine_vec, CommandName.ALIGN_PART)

    rotation = math.degrees(math.atan2(machine_xy[1], machine_xy[0])) - math.degrees(
        math.atan2(part_xy[1], part_xy[0])
    )
    ctx.state.rotary_table.position = rotation % 360.0

    if alpha != 0.0:
        # By construction the table is rotated to exactly eliminate the angle
        # between the two projected vectors, so this only ever catches
        # floating-point residue - a real (non-simulated) table would have
        # actual mechanical error to check against alpha here.
        cosine = max(-1.0, min(1.0, part_xy[0] * machine_xy[0] + part_xy[1] * machine_xy[1]))
        residual = math.degrees(math.acos(cosine))
        if residual > alpha:
            raise ServerError(
                ErrorSeverity.ERROR,
                ErrorCode.PART_NOT_ALIGNED,
                CommandName.ALIGN_PART,
                "Part not aligned",
            )

    return builders.align_part(part_xy, machine_xy)


def _normalize_xy(vector: Vec3, cause: str) -> tuple[float, float]:
    """Project ``vector`` onto the XY (table) plane and normalize it."""
    x, y, _z = vector
    length = math.hypot(x, y)
    if length == 0.0:
        raise ServerError(
            ErrorSeverity.ERROR,
            ErrorCode.VECTOR_HAS_NO_NORM,
            cause,
            "Vector has no norm after projection",
        )
    return x / length, y / length


def register(registry: CommandRegistry) -> None:
    registry.register(
        CommandName.ENABLE_ROTARY_TABLE_VAR_CSY,
        _enable_rotary_table_var_csy,
        arguments=_ENABLE_PARAMS,
    )
    registry.register(CommandName.ALIGN_PART, _align_part, arguments=_ALIGN_PART_PARAMS)
