# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""The concrete, simulated command classes behind :class:`~pyippdme.simulation.VirtualCMM`.

Each module here exposes a single ``register(registry)`` function that adds
its commands to a :class:`~pyippdme.server.registry.CommandRegistry`, exactly
like :mod:`pyippdme.server.classes`'s two mandatory classes (``Server``/``DME``,
re-exported here too for convenience - see :data:`DEFAULT_COMMAND_CLASSES`).
Unlike those two, every class here models *simulated* machine behavior over
:class:`~pyippdme.server.registry.MachineState`'s
:class:`~pyippdme.simulation.state.SimulationState` extension - a real
integrator driving actual hardware would replace these with their own
handlers rather than reusing them, which is exactly why they live in this
separate package instead of :mod:`pyippdme.server` itself.

:mod:`cartcmm_class` (a subset of ``CartCMM``/``Cartesian``/``TouchTrigger``
covering coordinate queries, coordinate-system transformations, absolute
motion, and single-point measurement, plus ``TouchTrigger_SelfCentering``),
:mod:`tool_class` (a subset of ``Tool`` covering the
``GoToPar``/``PtMeasPar``/``ScanPar`` parameter blocks, ``Tool.Id()``, and
``Alignable_AB``/``Alignable_ABC``), :mod:`toolchanger_class` (``ToolChanger``:
``EnumTools``/``ChangeTool``/``FindTool``/``FoundTool``/``SetTool``/
``GetChangeToolAction`` over a small fixed tool catalog), :mod:`scanning_class`
(a subset of ``Scanning`` covering known- and simplified unknown-contour
scans), :mod:`formtester_class` (a subset of ``FormTester`` covering the
centering/tilting alignment commands and axis/position locking),
:mod:`mover_class` (``Mover``: user-enable state, axis enumeration, and
scale-temperature/compensation-origin commands), :mod:`rotarytable_class` (a
subset of ``RotaryTable``), :mod:`part_class` (``Part``: the
temperature/approach/search/retract properties), and :mod:`rawdata_class` (a
subset of ``Optical``/``RawDataHandling`` covering data acquisition and all
three raw-data transfer technologies). Add further classes the same way and
register them on your own :class:`~pyippdme.server.IppDmeServer` instance.
"""

from pyippdme.server.classes import register_dme_class, register_server_class
from pyippdme.simulation.classes.cartcmm_class import register as register_cartcmm_class
from pyippdme.simulation.classes.feature_class import register as register_feature_class
from pyippdme.simulation.classes.formtester_class import register as register_formtester_class
from pyippdme.simulation.classes.mover_class import register as register_mover_class
from pyippdme.simulation.classes.part_class import register as register_part_class
from pyippdme.simulation.classes.rawdata_class import register as register_rawdata_class
from pyippdme.simulation.classes.rotarytable_class import register as register_rotarytable_class
from pyippdme.simulation.classes.scanning_class import register as register_scanning_class
from pyippdme.simulation.classes.tool_class import register as register_tool_class
from pyippdme.simulation.classes.toolchanger_class import register as register_toolchanger_class

__all__ = [
    "register_cartcmm_class",
    "register_dme_class",
    "register_feature_class",
    "register_formtester_class",
    "register_mover_class",
    "register_part_class",
    "register_rawdata_class",
    "register_rotarytable_class",
    "register_scanning_class",
    "register_server_class",
    "register_tool_class",
    "register_toolchanger_class",
]
