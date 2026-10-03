# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""A digital twin of a bridge CMM that answers the I++ DME protocol, without any GUI.

Needs the ``OCP`` package (``pip install pyippdme[twin]``). See
:class:`~pyippdme.twin.twin.DigitalTwin`; the Qt front end is
:mod:`pyippdme.gui`.
"""

from pyippdme.twin.machine import MachineModel
from pyippdme.twin.objects import SceneObject, demo_sample, primitive_fixture
from pyippdme.twin.sensor import LineScanner, LineScannerSpec
from pyippdme.twin.spec import PRESETS, Accuracy, MachineSpec, ToolSpec
from pyippdme.twin.twin import DigitalTwin, DrawItem, TwinEvent, TwinSnapshot
from pyippdme.twin.view import SimulationView

__all__ = [
    "PRESETS",
    "Accuracy",
    "DigitalTwin",
    "DrawItem",
    "LineScanner",
    "LineScannerSpec",
    "MachineModel",
    "MachineSpec",
    "SceneObject",
    "SimulationView",
    "ToolSpec",
    "TwinEvent",
    "TwinSnapshot",
    "demo_sample",
    "primitive_fixture",
]
