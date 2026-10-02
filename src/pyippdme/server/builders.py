# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Typed helpers that build response payloads for command handlers.

Mirrors :mod:`pyippdme.client.builders` (which builds the ``Argument`` tuple a
command's *request* needs) on the response side: each function here builds
exactly the :data:`~pyippdme.server.registry.HandlerResult` a specific
command's response needs, so a handler - whether one of this project's own,
or a user's own server implementing a command from scratch (see
:class:`~pyippdme.server.IppDmeServer`'s module docstring,
:meth:`~pyippdme.server.registry.CommandRegistry.register`) - doesn't need to
know that command's exact wire shape (which :data:`~pyippdme.protocol.ast.DataPayload`
variant, how many ``NamedValue``s, what they're named, in what order) to
answer it correctly.

Every built-in command class whose response has a fixed, well-known shape
(:mod:`pyippdme.server.classes`) uses these same functions internally instead
of building its own response locally - matching
:mod:`pyippdme.client.builders`'s own reason for existing: exactly one place
knows each shape, not several private, near-identical copies of the same
handful of patterns (a boolean as a bare number, a boolean as one named
field, a list of bare names, ...) re-implemented per file.

Not every command gets a *dedicated* function here: one whose response shape
is fixed by its *request* rather than by the command itself (``Get``/
``PtMeas``/``SetScaleTemperatures`` echo back only as many fields as were
asked for; ``GetProp``/``EnumProp`` enumerate however many properties/
children happen to exist) has nothing further to fix in advance -
:func:`named_numbers`/:func:`bare_names` below already build whatever shape
those turn out to need, one call at a time, same as they would for a
genuinely fixed-shape command.
"""

from __future__ import annotations

from collections.abc import Iterable

from pyippdme.protocol.ast import (
    BasicName,
    Items,
    NamedValue,
    NameValue,
    Number,
    NumericData,
    PropertyData,
    String,
    StringValue,
)
from pyippdme.protocol.commands import CommandName
from pyippdme.protocol.parameters import ParameterName
from pyippdme.protocol.signature import Parameter
from pyippdme.types.csy import CoordinateTransform

# ---------------------------------------------------------------------------
# Generic building blocks - not tied to any one command, and the ones a
# command-specific function below can't be written for (see module
# docstring) still need.
# ---------------------------------------------------------------------------


def bare_names(*names: str) -> Items:
    """Build one empty-argument ``NamedValue`` per name.

    E.g. ``EnumerateMoverAxes()`` -> ``X(), Y(), Z()``.
    """
    return Items(tuple(NamedValue(n, ()) for n in names))


def named_numbers(**values: float) -> Items:
    """Build one ``NamedValue(name, (Number,))`` per keyword.

    E.g. ``Get(X(10), Y(20))``'s response shape.
    """
    return Items(tuple(NamedValue(name, (Number.of(v),)) for name, v in values.items()))


def number(value: float) -> NumericData:
    """Build a single bare ``NumericData``, e.g. ``ReadTemperatureSensor(...)``'s response shape."""
    return NumericData((Number.of(value),))


def boolean(value: bool) -> NumericData:
    """Build a single bare ``0``/``1`` ``NumericData`` - a common yes/no response shape."""
    return number(1 if value else 0)


def named_boolean(name: str, value: bool) -> Items:
    """Build a single named ``0``/``1`` field, e.g. ``IsHomed()``/``IsUserEnabled()``'s shape."""
    return Items((NamedValue(name, (Number.of(1 if value else 0),)),))


def name_value(value: str) -> NameValue:
    """Build a single bare name, e.g. ``GetCoordSystem()``/``Tool()``/``FoundTool()``'s shape."""
    return NameValue(value)


def string_value(value: str) -> StringValue:
    """Build a single bare (quoted) string, e.g. ``GetMachineClass()``'s response shape."""
    return StringValue(String(value))


def string_list(values: Iterable[str]) -> list[StringValue]:
    """Build one bare-string message per value, e.g. ``GetSupportedCommands()``'s response shape."""
    return [StringValue(String(v)) for v in values]


def name_list(values: Iterable[str]) -> list[NameValue]:
    """Build one bare-name message per value, e.g. ``EnumTools()``'s response shape."""
    return [NameValue(v) for v in values]


def supported_arguments(parameters: Iterable[Parameter]) -> list[PropertyData]:
    """Build ``GetSupportedArguments(CommandName)``'s response shape (6.4.1): one pair each."""
    return [property_entry(p.name, p.datatype) for p in parameters]


def property_entry(name: str, kind: str) -> PropertyData:
    """Build a single name/kind pair, e.g. ``EnumProp(...)``'s per-entry response shape."""
    return PropertyData(String(name), String(kind))


# ---------------------------------------------------------------------------
# Per-command builders, for a shape specific to one command.
# ---------------------------------------------------------------------------


def get_dme_version(version: str) -> Items:
    """``GetDMEVersion()``'s response shape (6.4.1)."""
    return Items((NamedValue(ParameterName.DME_VERSION, (String(version),)),))


def get_xtd_err_status(*, homed: bool, error: tuple[int, int] | None = None) -> list[Items]:
    """``GetXtdErrStatus()``'s response shape (6.3.1): ``IsHomed``, plus an active error if any.

    ``error`` is ``(number, severity)``.
    """
    lines = [named_boolean(CommandName.IS_HOMED, homed)]
    if error is not None:
        error_number, severity = error
        lines.append(
            Items(
                (
                    NamedValue(ParameterName.ACTIVE_ERROR, (Number.of(error_number),)),
                    NamedValue(ParameterName.SEVERITY, (Number.of(severity),)),
                )
            )
        )
    return lines


def csy_transformation(transform: CoordinateTransform) -> Items:
    """``GetCsyTransformation``/``GetNamedCsyTransformation``'s response shape (6.5.1/6.5.2)."""
    return Items(
        (
            NamedValue(ParameterName.X0, (Number.of(transform.x0),)),
            NamedValue(ParameterName.Y0, (Number.of(transform.y0),)),
            NamedValue(ParameterName.Z0, (Number.of(transform.z0),)),
            NamedValue(ParameterName.THETA, (Number.of(transform.theta),)),
            NamedValue(ParameterName.PSI, (Number.of(transform.psi),)),
            NamedValue(ParameterName.PHI, (Number.of(transform.phi),)),
        )
    )


def temperature_sensor(
    name: str, kind: str, *, cmm_temp_correction: bool, scale_axis: str | None = None
) -> Items:
    """``GetTemperatureSensors()``'s per-sensor response shape (6.5.2)."""
    fields: list[NamedValue] = [
        NamedValue(ParameterName.NAME, (String(name),)),
        NamedValue(ParameterName.KIND, (BasicName(kind),)),
        NamedValue(
            ParameterName.CMM_TEMP_CORRECTION, (Number.of(1 if cmm_temp_correction else 0),)
        ),
    ]
    if scale_axis is not None:
        fields.append(NamedValue(ParameterName.SCALE, (String(scale_axis),)))
    return Items(tuple(fields))


def temperature_reading(name: str, temperature: float) -> Items:
    """``ReadAllTemperatures()``'s per-sensor response shape (6.5.2)."""
    return Items(
        (
            NamedValue(ParameterName.NAME, (String(name),)),
            NamedValue(ParameterName.TEMPERATURE, (Number.of(temperature),)),
        )
    )


def align_part(part_xy: tuple[float, float], machine_xy: tuple[float, float]) -> NumericData:
    """``AlignPart(...)``'s response shape (6.23): the two projected, normalized XY vectors.

    Each is reported as a full 3-number vector with a ``0.0`` Z, matching
    the request's own ``(px1, py1, pz1, mx1, my1, mz1, alpha)`` shape.
    """
    return NumericData(tuple(Number.of(v) for v in (*part_xy, 0.0, *machine_xy, 0.0)))


def get_change_tool_action(dx: float, dy: float, dz: float, *, action: str = "Switch") -> Items:
    """``GetChangeToolAction(...)``'s response shape (6.22.1)."""
    return Items(
        (
            NamedValue(ParameterName.ARGUMENT, (BasicName(action),)),
            NamedValue(ParameterName.X, (Number.of(dx),)),
            NamedValue(ParameterName.Y, (Number.of(dy),)),
            NamedValue(ParameterName.Z, (Number.of(dz),)),
        )
    )


def get_raw_data_sha_mem(name: str, data_segment: int, size: int) -> Items:
    """``GetRawDataShaMem(...)``'s response shape (6.15.1.1)."""
    return Items(
        (
            NamedValue(ParameterName.SHA_MEM_NAME, (String(name),)),
            NamedValue(ParameterName.DATA_SEGMENT, (Number.of(data_segment),)),
            NamedValue(ParameterName.SIZE, (Number.of(size),)),
        )
    )


def get_raw_data_file(url: str) -> Items:
    """``GetRawDataFile(...)``'s response shape (6.15.1.1)."""
    return Items((NamedValue(ParameterName.FILE_URL, (String(url),)),))
