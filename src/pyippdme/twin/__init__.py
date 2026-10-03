# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""A digital twin of a bridge CMM that answers the I++ DME protocol, without any GUI.

The simulation is split so that you can use the parts you need:

* Pure Python/numpy, importable without OpenCASCADE or Qt: :mod:`~pyippdme.twin.spec`
  (machine and tool definitions, ``machine.toml``), :mod:`~pyippdme.twin.planning`
  (velocity profiles and path planners), :mod:`~pyippdme.twin.toolmath` (tool
  kinematics), :mod:`~pyippdme.twin.optical` (optical sensor models) and
  :mod:`~pyippdme.twin.features` (nominal features and their evaluation).
* Needs the ``OCP`` package (``pip install pyippdme[twin]``): the CAD layer
  (:mod:`~pyippdme.twin.cad`), machine and tool shapes, and
  :class:`~pyippdme.twin.twin.DigitalTwin`, which puts it all behind the protocol.
* :mod:`pyippdme.gui` is one front end for it (Qt); any other can use
  :class:`~pyippdme.twin.view.SimulationView`.

Names exported here load on first use, so importing the package does not import OCP.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

_EXPORTS = {
    "Accuracy": "spec",
    "MachineSpec": "spec",
    "ToolSpec": "spec",
    "PRESETS": "spec",
    "DEFAULT_TOOLS": "spec",
    "MachineModel": "machine",
    "SceneObject": "objects",
    "demo_sample": "objects",
    "primitive_fixture": "objects",
    "build_check_artifact": "artifact",
    "build_reference_sphere": "artifact",
    "DigitalTwin": "twin",
    "DrawItem": "twin",
    "TwinEvent": "twin",
    "TwinSnapshot": "twin",
    "SimulationView": "view",
    "LineScanner": "optical",
    "LineScannerSpec": "optical",
}

__all__ = [
    "Accuracy",
    "DEFAULT_TOOLS",
    "DigitalTwin",
    "DrawItem",
    "LineScanner",
    "LineScannerSpec",
    "MachineModel",
    "MachineSpec",
    "PRESETS",
    "SceneObject",
    "SimulationView",
    "ToolSpec",
    "TwinEvent",
    "TwinSnapshot",
    "build_check_artifact",
    "build_reference_sphere",
    "demo_sample",
    "primitive_fixture",
]

if TYPE_CHECKING:
    from pyippdme.twin.artifact import build_check_artifact, build_reference_sphere  # noqa: F401
    from pyippdme.twin.machine import MachineModel  # noqa: F401
    from pyippdme.twin.objects import SceneObject, demo_sample, primitive_fixture  # noqa: F401
    from pyippdme.twin.optical import LineScanner, LineScannerSpec  # noqa: F401
    from pyippdme.twin.spec import (  # noqa: F401
        DEFAULT_TOOLS,
        PRESETS,
        Accuracy,
        MachineSpec,
        ToolSpec,
    )
    from pyippdme.twin.twin import DigitalTwin, DrawItem, TwinEvent, TwinSnapshot  # noqa: F401
    from pyippdme.twin.view import SimulationView  # noqa: F401


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'pyippdme.twin' has no attribute {name!r}")
    value = getattr(importlib.import_module(f"pyippdme.twin.{module}"), name)
    globals()[name] = value
    return value
