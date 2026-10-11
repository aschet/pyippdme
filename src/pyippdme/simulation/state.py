# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Per-connection state for this library's bundled simulated command classes.

Each nested dataclass here is owned by exactly one
:mod:`pyippdme.simulation.classes` module; :class:`SimulationState` bundles
them all as an extension of the core
:class:`~pyippdme.server.registry.MachineState` (session/error/identity
only), which is all :mod:`pyippdme.server.classes`' mandatory ``Server``/
``DME`` classes need. A real integrator with their own command classes
would define their own such extension the same way, rather than reusing
this one - see :class:`~pyippdme.server.IppDmeServer`'s ``state_factory``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any

from pyippdme.protocol.ast import EventTag
from pyippdme.server.registry import MachineState
from pyippdme.server.tool import ToolParameters
from pyippdme.simulation.tool import default_tool_parameters
from pyippdme.types.csy import CoordinateTransform
from pyippdme.types.pointcloud import PointCloudSet
from pyippdme.types.vec3 import Vec3


@dataclass(slots=True)
class CartCmmState:
    """State owned by :mod:`pyippdme.simulation.classes.cartcmm_class` (6.5, 6.8, 6.12)."""

    active_csy: str = "MachineCsy"
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    #: Axis names requested by the last ``OnPtMeasReport(...)``; empty until set.
    pt_meas_report: tuple[str, ...] = ()
    #: Live (this-connection-only) transforms set by ``SetCsyTransformation``
    #: (6.5.2), keyed by one of :data:`~pyippdme.types.csy.LIVE_TRANSFORM_NAMES`.
    #: Unlike named/saved transforms (see :class:`~pyippdme.types.csy.CsyStore`),
    #: these are never persisted.
    csy_transformations: dict[str, CoordinateTransform] = field(default_factory=dict)


@dataclass(slots=True)
class ToolState:
    """State owned by ``tool_class``/``toolchanger_class`` (6.10/6.22).

    See :mod:`pyippdme.simulation.classes`.
    """

    #: ``GoToPar``/``PtMeasPar``/``ScanPar`` parameter blocks of the active
    #: tool (6.10.4); see :mod:`pyippdme.simulation.tool`.
    parameters: ToolParameters = field(default_factory=default_tool_parameters)
    #: Name of the currently active tool, changed by ``ChangeTool``/``SetTool``
    #: (6.22.1); a key into :data:`pyippdme.simulation.classes.tool_class.TOOL_CATALOG`.
    active_name: str = "RefTool"
    #: Name of the tool last resolved by ``FindTool(...)``, or ``None`` if
    #: ``FindTool`` was never called this session ("otherwise it is
    #: UnDefTool", 6.22.1's ``FoundTool()``).
    found_name: str | None = None
    #: ISO-8601 UTC timestamp of each tool's last ``ReQualify()`` (6.10.2),
    #: keyed by tool name; unset reads as ``"00000000T000000Z"`` ("never
    #: qualified", ``Tool.LastQualified()``'s own example format, 6.10.3).
    last_qualified: dict[str, str] = field(default_factory=dict)
    #: Alignment set by ``AlignTool()`` (6.20.1), keyed by tool name: the
    #: primary vector, and the secondary vector if given. A tool never
    #: aligned this session is absent - see
    #: :func:`~pyippdme.simulation.classes.tool_class.tool_alignment` for the
    #: default this falls back to.
    alignment: dict[str, tuple[Vec3, Vec3 | None]] = field(default_factory=dict)
    #: Path of the tool collection opened by ``OpenToolCollection`` (6.22, Figure 56); tools are
    #: then named by the entries of that collection. ``None``: tool names are used as they are.
    open_collection: str | None = None
    #: ``UseSmallestAngletoAlignTool()``'s modal flag (6.20.1) - reset to
    #: ``False`` by ``StartSession()``.
    use_smallest_angle: bool = False
    #: ``Enable``/``DisableOptimize()``'s modal flag (6.20.1) - reset to
    #: ``False`` by ``StartSession()`` and ``ClearAllErrors()``.
    optimize_enabled: bool = False


@dataclass(slots=True)
class ScanningState:
    """State owned by :mod:`pyippdme.simulation.classes.scanning_class` (6.13.2)."""

    #: Report field names requested by the last ``OnScanReport(...)``; empty until set.
    report: tuple[str, ...] = ()


@dataclass(slots=True)
class FormTesterState:
    """State owned by :mod:`pyippdme.simulation.classes.formtester_class` (6.6.1)."""

    #: Axes currently locked by ``LockAxis(...)``; read by ``GoTo``/``PtMeas``
    #: to silently ignore motion on these axes.
    locked_axes: frozenset[str] = frozenset()
    #: Positions currently locked by ``LockPosition(...)``; stored and
    #: validated, but not enforced (see pyippdme.simulation.classes.formtester_class).
    locked_positions: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class MoveReportDaemon:
    """An active ``OnMoveReport``/``OnMoveReportE`` daemon (6.10.2's Table 68)."""

    tag: EventTag
    #: Which ``X``/``Y``/``Z``/``R``/``Tool.A``/``Tool.B``/``Tool.C`` fields to report.
    axes: tuple[str, ...]


@dataclass(slots=True)
class MoverState:
    """State owned by :mod:`pyippdme.simulation.classes.mover_class` (6.7.1)."""

    #: Manual jog box enable state (``EnableUser``/``DisableUser``).
    user_enabled: bool = True
    #: Per-axis scale temperature (``Set``/``GetScaleTemperatures``), keyed
    #: by axis name; an axis reads as 20.0 until explicitly set.
    scale_temperatures: dict[str, float] = field(default_factory=dict)
    #: Per-axis temperature compensation origin offset
    #: (``SetTemperatureCompensationOrigin``), keyed by X/Y/Z; unset reads as
    #: 0.0 ("the zero point of the active coordinate system will be used").
    temperature_compensation_origin: dict[str, float] = field(default_factory=dict)
    #: The active ``OnMoveReport``/``OnMoveReportE`` daemon, if any; see
    #: :data:`MoveReportDaemon` and :mod:`pyippdme.simulation.classes.mover_class`.
    report_daemon: MoveReportDaemon | None = None


@dataclass(slots=True)
class RotaryTableState:
    """State owned by :mod:`pyippdme.simulation.classes.rotarytable_class` (6.23)."""

    #: Position of the rotary table in degrees (``R()``/``R(r)``), read/written
    #: as a fourth axis by ``Get``/``GoTo``/``PtMeas``.
    position: float = 0.0
    #: Position of the second, orthogonal rotary table, set by ``AlignPart`` (6.23.1).
    second_position: float = 0.0
    #: Whether rotary-table-CSY calculation is enabled (``EnableRotaryTableVarCsy``).
    var_csy_enabled: bool = False


@dataclass(slots=True)
class RawDataState:
    """State owned by :mod:`pyippdme.simulation.classes.rawdata_class` (6.15, 6.17)."""

    #: Raw-data acquisitions buffered by ``DataAcquire`` (6.15.1), keyed by
    #: ``AcqName``, retrieved via one of ``GetRawDataBin``/``GetRawDataShaMem``/
    #: ``GetRawDataFile`` (6.17.2).
    #: "EndSession() implicitly deletes all buffered acquisitions" (6.15.1).
    acquisitions: dict[str, PointCloudSet] = field(default_factory=dict)
    #: ``RawDataFormat``/``Port`` remembered from the last ``RawDataBinSetup`` (6.17.2.1).
    bin_format: str | None = None
    bin_port: int | None = None
    #: ``LiveMode`` of the last ``RawDataBinSetup``: points are sent while they are acquired.
    bin_live: bool = False
    #: Shared-memory segments opened by ``GetRawDataShaMem``, keyed by
    #: ``AcqName``, released by ``ReleaseShaMem`` (6.17.2.2).
    shared_memory_segments: dict[str, SharedMemory] = field(default_factory=dict)
    #: Files written by ``GetRawDataFile``, keyed by ``AcqName``, deleted by
    #: ``DelRawDataFile`` (6.17.2.3).
    files: dict[str, Path] = field(default_factory=dict)


@dataclass(slots=True)
class PartState:
    """State owned by :mod:`pyippdme.simulation.classes.part_class` (6.24.2).

    Keyed by dotted property name (``Part.Temperature``, ...); unset reads
    as that property's ``StartSession()`` default - see
    :mod:`pyippdme.simulation.classes.part_class`.
    """

    properties: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class FeatureState:
    """State of the deprecated ``FeatureExtraction`` class: the regions of interest by name."""

    rois: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SimulationState(MachineState):
    """Per-connection state for every bundled simulated command class.

    State that belongs to exactly one command class is grouped into its own
    nested dataclass (``state.tool.active_name`` rather than a same-level
    field), so each class's state stays discoverable without this class
    growing without bound as more simulated classes are added. See
    :class:`~pyippdme.server.registry.MachineState` for the core fields this
    extends.
    """

    cart_cmm: CartCmmState = field(default_factory=CartCmmState)
    tool: ToolState = field(default_factory=ToolState)
    scanning: ScanningState = field(default_factory=ScanningState)
    form_tester: FormTesterState = field(default_factory=FormTesterState)
    mover: MoverState = field(default_factory=MoverState)
    rotary_table: RotaryTableState = field(default_factory=RotaryTableState)
    raw_data: RawDataState = field(default_factory=RawDataState)
    part: PartState = field(default_factory=PartState)
    features: FeatureState = field(default_factory=FeatureState)

    def carry_over_from(self, previous: MachineState) -> None:
        """Keep what the machine keeps between clients (6.3.1).

        That is the active tool and its properties, the active coordinate
        system with its transformations, and where the machine stands.
        """
        super().carry_over_from(previous)
        if not isinstance(previous, SimulationState):
            return
        self.tool.active_name = previous.tool.active_name
        self.tool.parameters = previous.tool.parameters
        self.tool.last_qualified = previous.tool.last_qualified
        self.tool.alignment = previous.tool.alignment
        self.cart_cmm.active_csy = previous.cart_cmm.active_csy
        self.cart_cmm.position = previous.cart_cmm.position
        self.cart_cmm.csy_transformations = previous.cart_cmm.csy_transformations
        self.rotary_table.position = previous.rotary_table.position
        self.rotary_table.second_position = previous.rotary_table.second_position
        self.rotary_table.var_csy_enabled = previous.rotary_table.var_csy_enabled
