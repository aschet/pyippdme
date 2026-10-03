# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Client and server package for VDMA 8722:2024-04.

I++ DME - Dimensional Measurement Equipment Interface.
"""

from pyippdme.client import IppDmeClient
from pyippdme.client.model import IppDmeMachine
from pyippdme.exceptions import (
    IppDmeConnectionError,
    IppDmeError,
    IppDmeProtocolError,
    IppDmeServerError,
    IppDmeTimeoutError,
)
from pyippdme.protocol.commands import CommandName
from pyippdme.protocol.errors import ErrorSeverity, ServerError
from pyippdme.protocol.namespace import proprietary_name
from pyippdme.protocol.network import TCP_NETWORK, MemoryNetwork, Network
from pyippdme.protocol.parameters import ParameterName
from pyippdme.protocol.transport import DEFAULT_PORT
from pyippdme.server import IppDmeServer
from pyippdme.server.dispatch import command, command_proprietary, register_object
from pyippdme.server.motion import (
    MotionError,
    MotionModel,
    MotionRequest,
    ProbeRequest,
    ProbeResult,
    ToolHandler,
)
from pyippdme.server.registry import (
    CommandContext,
    CommandRegistry,
    MachineState,
    commands_of,
    component_name,
    register_subset,
)
from pyippdme.server.surface import RawSensor, SampleSurface
from pyippdme.simulation import DEFAULT_COMMAND_CLASSES
from pyippdme.simulation.surface import CylinderSurface, PlaneSurface, SphereSurface
from pyippdme.simulation.virtual_cmm import VirtualCMM
from pyippdme.spy import Spy, SpyDirection, SpyMessage
from pyippdme.types.csy import CSY_CHAIN, CoordinateTransform, CsyContext

__version__ = "0.1.0"

__all__ = [
    "CSY_CHAIN",
    "DEFAULT_COMMAND_CLASSES",
    "DEFAULT_PORT",
    "TCP_NETWORK",
    "CommandContext",
    "CommandName",
    "CommandRegistry",
    "CoordinateTransform",
    "CsyContext",
    "CylinderSurface",
    "ErrorSeverity",
    "IppDmeClient",
    "IppDmeConnectionError",
    "IppDmeError",
    "IppDmeMachine",
    "IppDmeProtocolError",
    "IppDmeServer",
    "IppDmeServerError",
    "IppDmeTimeoutError",
    "MachineState",
    "MemoryNetwork",
    "MotionError",
    "MotionModel",
    "MotionRequest",
    "Network",
    "ParameterName",
    "PlaneSurface",
    "ProbeRequest",
    "ProbeResult",
    "RawSensor",
    "SampleSurface",
    "ServerError",
    "SphereSurface",
    "Spy",
    "SpyDirection",
    "SpyMessage",
    "ToolHandler",
    "VirtualCMM",
    "__version__",
    "command",
    "command_proprietary",
    "commands_of",
    "component_name",
    "proprietary_name",
    "register_object",
    "register_subset",
]
