# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""The events a server sends on its own with tag ``E0000`` (5.5.1, 5.5.3)."""

from __future__ import annotations

from dataclasses import dataclass

from pyippdme.client.report import Report, ReportValue, report_from_payload
from pyippdme.protocol.ast import (
    BasicName,
    DataPayload,
    DataResponse,
    Items,
    NamedValue,
    Number,
    String,
)


@dataclass(frozen=True, slots=True)
class KeyPress:
    """``KeyPress(NameOfKey)``: a key was pressed on the jog box.

    Besides the keys of the machine, ``Done``, ``Del`` and the soft keys ``F1`` ... ``Fn``
    are defined.
    """

    key: str


@dataclass(frozen=True, slots=True)
class ClearancePoint:
    """``GoTo(...)``: the user set a clearance or intermediate point.

    ``values`` are the fields of ``OnPtMeasReport``; vectors are ``(0, 0, 0)``.
    """

    values: Report


@dataclass(frozen=True, slots=True)
class ManualPoint:
    """``PtMeas(...)``: the user picked a point by hand; ``values`` as of ``OnPtMeasReport``."""

    values: Report


@dataclass(frozen=True, slots=True)
class ToolChanged:
    """``ChangeTool(ToolName)``: the tool was changed without a command from the client."""

    tool_name: str


@dataclass(frozen=True, slots=True)
class PropertyChanged:
    """``SetProp(...)``: a property was changed without a command from the client."""

    name: str
    value: ReportValue


@dataclass(frozen=True, slots=True)
class ToolCollectionOpened:
    """``OpenToolCollection(ToolCollectionPath)``: the server changed the tool collection."""

    path: str


@dataclass(frozen=True, slots=True)
class DataAcquireRequested:
    """``DataAcquire(...)``: the user took a measurement at the machine (6.15, use case 2).

    The client usually inserts the command into its inspection plan. ``positions`` are the
    ``(position, primary, secondary)`` triples; ``step_width`` and ``rotary_table`` are the
    optional last numbers when the server sent them.
    """

    acq_name: str
    acquisition_type: str
    settings_name: str
    positions: tuple[tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]], ...]
    step_width: float | None = None
    rotary_table: bool | None = None


@dataclass(frozen=True, slots=True)
class UnknownEvent:
    """An event the standard does not define, for example a proprietary one."""

    payload: DataPayload


ServerEvent = (
    KeyPress
    | ClearancePoint
    | ManualPoint
    | ToolChanged
    | PropertyChanged
    | ToolCollectionOpened
    | DataAcquireRequested
    | UnknownEvent
)


def _string(named: NamedValue) -> str:
    (value,) = named.args
    if not isinstance(value, String):
        raise TypeError(f"Expected a string argument for {named.name}")
    return value.value


def parse_event(response: DataResponse) -> ServerEvent:
    """Read the pre-defined event (5.5.3) a data response with tag ``E0000`` carries."""
    payload = response.data
    if not isinstance(payload, Items) or len(payload.values) != 1:
        return UnknownEvent(payload)
    (named,) = payload.values
    if named.name == "DataAcquire":
        parsed = _data_acquire(named)
        if parsed is not None:
            return parsed
    if named.name == "KeyPress":
        return KeyPress(_string(named))
    if named.name == "ChangeTool":
        return ToolChanged(_string(named))
    if named.name == "OpenToolCollection":
        return ToolCollectionOpened(_string(named))
    if named.name in ("GoTo", "PtMeas"):
        fields = tuple(arg for arg in named.args if isinstance(arg, NamedValue))
        values = report_from_payload(Items(fields))
        return ClearancePoint(values) if named.name == "GoTo" else ManualPoint(values)
    if named.name == "SetProp" and len(named.args) == 1 and isinstance(named.args[0], NamedValue):
        prop = named.args[0]
        if all(isinstance(a, Number) for a in prop.args):
            numbers = tuple(a.value for a in prop.args if isinstance(a, Number))
            return PropertyChanged(prop.name, numbers[0] if len(numbers) == 1 else numbers)
        return PropertyChanged(prop.name, _string(prop))
    return UnknownEvent(payload)


def _data_acquire(named: NamedValue) -> DataAcquireRequested | None:
    """Read the positional arguments of ``DataAcquire`` (Table 95); ``None`` if they do not fit."""
    args = named.args
    if len(args) < 4 or not isinstance(args[0], String) or not isinstance(args[2], String):
        return None
    kind, count = args[1], args[3]
    if not isinstance(kind, BasicName) or not isinstance(count, Number):
        return None
    n = int(count.value)
    numbers = [a.value for a in args[4:] if isinstance(a, Number)]
    if len(numbers) != len(args) - 4 or len(numbers) < 9 * n:
        return None
    positions = tuple(
        (
            tuple(numbers[9 * i : 9 * i + 3]),
            tuple(numbers[9 * i + 3 : 9 * i + 6]),
            tuple(numbers[9 * i + 6 : 9 * i + 9]),
        )
        for i in range(n)
    )
    rest = numbers[9 * n :]
    return DataAcquireRequested(
        args[0].value,
        kind.value,
        args[2].value,
        positions,
        rest[0] if rest else None,
        bool(rest[1]) if len(rest) > 1 else None,
    )
