# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Build command lines from a form: what a front end needs to offer every command, without a GUI.

A command's parameters come from the catalog of the bundled server classes
(:data:`~pyippdme.simulation.catalog.BUILTIN_COMMANDS`) or from any catalog you build with
:func:`~pyippdme.server.catalog.build_command_catalog`. :func:`command_fields` describes one
command as an ordered list of :class:`FormField`, and :func:`build_command_line` turns the values a
user typed into the text :func:`~pyippdme.protocol.parser.parse_method` accepts::

    fields = command_fields("GoTo")                      # one enum field: "Positions"
    build_command_line("GoTo", {"Positions": "X(10),Y(20)"})   # 'GoTo(X(10),Y(20))'
    build_command_line("ScanOnLine", {"Sx": "0", ...})   # positional commands: bare values
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass

from pyippdme.protocol.signature import DataType
from pyippdme.server.catalog import CommandInfo
from pyippdme.simulation.catalog import BUILTIN_COMMANDS

__all__ = [
    "COMMAND_GROUPS",
    "FormField",
    "build_command_line",
    "command_fields",
    "command_names",
    "grouped_commands",
    "result_rows",
]

#: The commands by the class of the object model that defines them (Figure 1, chapter 6).
COMMAND_GROUPS: dict[str, tuple[str, ...]] = {
    "Session and errors": (
        "StartSession",
        "EndSession",
        "AbortE",
        "ClearAllErrors",
        "GetErrStatusE",
        "GetXtdErrStatus",
        "GetErrorInfo",
        "StopDaemon",
        "StopAllDaemons",
        "EnumNameSpaces",
    ),
    "Properties": ("GetProp", "GetPropE", "SetProp", "EnumProp", "EnumAllProp"),
    "Machine": (
        "GetDMEVersion",
        "GetSupportedCommands",
        "GetSupportedArguments",
        "GetMachineClass",
        "Home",
        "IsHomed",
        "EnableUser",
        "DisableUser",
        "IsUserEnabled",
        "EnumerateMoverAxes",
    ),
    "Move": (
        "GoTo",
        "Step",
        "GoToOnCircle",
        "GoToOnSpiral",
        "Get",
        "OnMoveReport",
        "OnMoveReportE",
        "LockAxis",
        "LockPosition",
    ),
    "Coordinate systems": (
        "SetCoordSystem",
        "GetCoordSystem",
        "SetCsyTransformation",
        "GetCsyTransformation",
        "SaveNamedCsyTransformation",
        "GetNamedCsyTransformation",
        "SaveActiveCoordSystem",
        "LoadCoordSystem",
        "DeleteCoordSystem",
        "EnumCoordSystems",
        "EnableRotaryTableVarCsy",
        "AlignPart",
    ),
    "Temperature": (
        "GetTemperatureSensors",
        "ReadTemperatureSensor",
        "ReadAllTemperatures",
        "UpdateScaleTemperatures",
        "SetScaleTemperatures",
        "GetScaleTemperatures",
        "SetTemperatureCompensationOrigin",
    ),
    "Measure a point": (
        "OnPtMeasReport",
        "PtMeas",
        "PtMeasPar",
        "PtMeasSelfCenter",
        "PtMeasSelfCenterLocked",
    ),
    "Scan": (
        "OnScanReport",
        "ScanPar",
        "ScanOnLine",
        "ScanOnCircle",
        "ScanOnHelix",
        "ScanOnCurve",
        "ScanOnLineHint",
        "ScanOnCircleHint",
        "ScanOnCurveHint",
        "ScanOnCurveDensity",
        "ScanUnknownHint",
        "ScanUnknownDensity",
        "ScanInPlaneEndIsSphere",
        "ScanInPlaneEndIsPlane",
        "ScanInPlaneEndIsCyl",
        "ScanInCylEndIsSphere",
        "ScanInCylEndIsPlane",
    ),
    "Tool": (
        "Tool",
        "GoToPar",
        "OptPar",
        "IsAlignable",
        "AlignTool",
        "AvrRadius",
        "CalcToolAlignment",
        "CalcToolAngles",
        "UseSmallestAngletoAlignTool",
        "ReQualify",
        "EnableOptimize",
        "DisableOptimize",
        "IsOptimizeEnabled",
    ),
    "Tool changer": (
        "EnumTools",
        "ChangeTool",
        "FindTool",
        "FoundTool",
        "SetTool",
        "GetChangeToolAction",
        "EnumToolCollection",
        "EnumAllToolCollections",
        "OpenToolCollection",
    ),
    "Optical and raw data": (
        "DataAcquire",
        "DeleteAcquistion",
        "DeleteAllAcquisitions",
        "AdvDataStruct",
        "RawDataBinSetup",
        "GetRawDataBin",
        "GetRawDataShaMem",
        "ReleaseShaMem",
        "GetRawDataFile",
        "DelRawDataFile",
    ),
    "Form tester": ("CenterPart", "TiltPart", "TiltCenterPart"),
}


def grouped_commands(
    catalog: Mapping[str, CommandInfo] = BUILTIN_COMMANDS,
) -> dict[str, list[str]]:
    """Return the commands of ``catalog`` by group; those not in a group go to ``Other``."""
    available = set(catalog)
    groups: dict[str, list[str]] = {}
    placed: set[str] = set()
    for group, names in COMMAND_GROUPS.items():
        found = [n for n in names if n in available]
        if found:
            groups[group] = found
            placed.update(found)
    rest = sorted(available - placed)
    if rest:
        groups["Other"] = rest
    return groups


@dataclass(frozen=True, slots=True)
class FormField:
    """One input of a command form."""

    name: str
    datatype: DataType
    mandatory: bool
    #: Written as a bare value in call order, not as ``Name(value)``.
    positional: bool

    @property
    def hint(self) -> str:
        """A short description for a tooltip or placeholder."""
        if self.datatype is DataType.ENUM:
            return "list of items, e.g. X(10),Y(20)"
        kind = {
            DataType.BOOL: "0 or 1",
            DataType.INT: "integer",
            DataType.FLOAT: "number",
            DataType.STRING: "text",
            DataType.NAME: "name",
        }[self.datatype]
        return kind if self.mandatory else f"{kind} (optional)"


def command_names(catalog: Mapping[str, CommandInfo] = BUILTIN_COMMANDS) -> list[str]:
    """All commands of ``catalog`` (default: the bundled classes), sorted."""
    return sorted(catalog)


def command_fields(
    name: str, catalog: Mapping[str, CommandInfo] = BUILTIN_COMMANDS
) -> list[FormField]:
    """Return the inputs of ``name``; empty without arguments (or without a schema).

    ``catalog`` can come from :func:`~pyippdme.server.catalog.build_command_catalog` for your own
    command classes.
    """
    info = catalog.get(name)
    if info is None or not info.arguments:
        return []
    return [FormField(p.name, p.datatype, p.mandatory, p.positional) for p in info.arguments]


def _quote(text: str) -> str:
    return '"' + text.replace('"', "") + '"'


def _render(field: FormField, text: str) -> str:
    if field.datatype is DataType.STRING:
        return _quote(text[1:-1] if len(text) > 1 and text[0] == text[-1] == '"' else text)
    return text


def build_command_line(
    name: str,
    values: Mapping[str, str],
    fields: list[FormField] | None = None,
    catalog: Mapping[str, CommandInfo] = BUILTIN_COMMANDS,
) -> str:
    """Turn typed values into a command line; empty values are left out.

    Positional commands write their values in order and stop at the first empty optional one
    (a gap in the middle is an error, as the protocol has no way to skip an argument).
    Raises :class:`ValueError` when a mandatory value is missing.
    """
    fields = command_fields(name, catalog) if fields is None else fields
    parts: list[str] = []
    gap: str | None = None
    for field in fields:
        text = values.get(field.name, "").strip()
        if not text:
            if field.mandatory:
                raise ValueError(f"{field.name} is required")
            gap = gap or field.name
            continue
        if field.positional:
            if gap is not None:
                raise ValueError(f"{field.name} is given but {gap} before it is empty")
            parts.append(_render(field, text))
        elif field.datatype is DataType.ENUM:
            parts.append(text)
        else:
            parts.append(f"{field.name}({_render(field, text)})")
    return f"{name}({','.join(parts)})"


def result_rows(report: Mapping[str, object]) -> Iterator[tuple[str, str]]:
    """Name/value pairs of a decoded response, for a table."""
    for key, value in report.items():
        if isinstance(value, tuple):
            yield key, ", ".join(f"{v:g}" for v in value)
        elif isinstance(value, float):
            yield key, f"{value:g}"
        else:
            yield key, "NULL" if value is None else str(value)
