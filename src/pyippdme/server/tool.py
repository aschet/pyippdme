# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""The ``Tool`` parameter-block mechanism (6.10.4 "Parameter sets of class Tool").

``GoToPar``/``PtMeasPar``/``ScanPar`` are property blocks addressed through
``Tool`` (e.g. ``GetProp(Tool.PtMeasPar.Retract())``) even though, per 6.10.4,
they conceptually belong to other classes (``Tool``, ``TouchTrigger``,
``Scanning`` respectively). Each individual parameter has four values -
``Min``, ``Max``, ``Def`` and ``Act`` - of which only ``Act`` is client
settable, clamped into ``[Min, Max]`` with warning ``0504`` if the client's
value falls outside it (6.10.4). ``.Act()`` may be omitted, i.e.
``Tool.PtMeasPar.Retract()`` and ``Tool.PtMeasPar.Retract.Act()`` refer to
the same value.

This module holds only the mechanism itself - a bare :class:`ToolParameters`
starts with empty blocks, no ``Min``/``Max``/``Def`` numbers of its own,
since the standard leaves those to each tool's qualification. See
:mod:`pyippdme.simulation.tool` for this library's own bundled simulation's
plausible defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class ParameterField(StrEnum):
    """Sub-property suffixes accepted after a parameter name, e.g. ``Retract.Min()``."""

    MIN = "Min"
    MAX = "Max"
    DEF = "Def"
    ACT = "Act"


#: Every :class:`ParameterField`, in the order 6.10.4 lists them.
PARAMETER_FIELDS = tuple(ParameterField)


@dataclass(slots=True)
class ToolParameter:
    """One physical value of a parameter set, with the Min/Max/Def/Act model."""

    minimum: float
    maximum: float
    default: float
    act: float | None = None

    @property
    def value(self) -> float:
        """The effective (``Act``, or ``Def`` if ``Act`` was never set) value."""
        return self.default if self.act is None else self.act

    def set_act(self, value: float) -> bool:
        """Clamp ``value`` into ``[minimum, maximum]`` and store it as ``Act``.

        Returns whether clamping changed the value (the caller should then
        raise warning ``0504`` "Argument out of range").
        """
        clamped = min(max(value, self.minimum), self.maximum)
        self.act = clamped
        return clamped != value


ToolParameterSet = dict[str, ToolParameter]


@dataclass(slots=True)
class ToolParameters:
    """All ``GoToPar``/``PtMeasPar``/``ScanPar`` parameter blocks of the active tool.

    Starts with every block empty - see the module docstring for why no
    ``Min``/``Max``/``Def`` numbers are built in here.
    """

    go_to_par: ToolParameterSet = field(default_factory=dict)
    pt_meas_par: ToolParameterSet = field(default_factory=dict)
    scan_par: ToolParameterSet = field(default_factory=dict)

    def block(self, name: str) -> ToolParameterSet | None:
        return {
            "GoToPar": self.go_to_par,
            "PtMeasPar": self.pt_meas_par,
            "ScanPar": self.scan_par,
        }.get(name)
