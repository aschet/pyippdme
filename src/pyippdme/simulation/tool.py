# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""This library's own plausible ``Tool`` parameter-block defaults (6.10.4).

The ``Min``/``Max``/``Def`` values below are this library's own simulation
defaults (plausible values for a small tactile CMM), not numbers the
standard mandates - see :mod:`pyippdme.server.tool` for the mechanism
itself (Min/Max/Def/Act clamping semantics), which has no such numbers
built in.
"""

from __future__ import annotations

from pyippdme.server.tool import ToolParameter, ToolParameters, ToolParameterSet


def _go_to_par() -> ToolParameterSet:
    return {
        "Speed": ToolParameter(0.1, 500.0, 100.0),
        "Accel": ToolParameter(0.1, 2000.0, 500.0),
    }


def _pt_meas_par() -> ToolParameterSet:
    return {
        "Speed": ToolParameter(0.1, 50.0, 5.0),
        "Accel": ToolParameter(0.1, 500.0, 100.0),
        "Approach": ToolParameter(0.0, 50.0, 5.0),
        "Search": ToolParameter(0.0, 20.0, 2.0),
        # Retract may be negative: "the server will move back to the
        # approach position" (6.12.1's remark on Tool.Retract()).
        "Retract": ToolParameter(-1.0, 50.0, 2.0),
    }


def _scan_par() -> ToolParameterSet:
    return {
        "Speed": ToolParameter(0.1, 50.0, 5.0),
        "Accel": ToolParameter(0.1, 500.0, 100.0),
        "Retract": ToolParameter(-1.0, 50.0, 2.0),
    }


def default_tool_parameters() -> ToolParameters:
    return ToolParameters(go_to_par=_go_to_par(), pt_meas_par=_pt_meas_par(), scan_par=_scan_par())
