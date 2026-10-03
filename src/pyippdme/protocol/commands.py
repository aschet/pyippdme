# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Names of the commands this library's built-in command classes register.

Every ``registry.register("GoTo", ...)``-style call in :mod:`pyippdme.server.classes`
uses a member of :class:`CommandName` instead of a bare string literal, and a
caller of the low-level :class:`~pyippdme.client.IppDmeClient` API can do the
same - ``client.call(CommandName.GO_TO, ...)`` instead of retyping
``"GoTo"`` by hand, with a typo caught as an unknown *name* (by mypy/the
IDE) rather than surfacing later as a wire-level ``0501 Unsupported
command``. :class:`CommandName` is a :class:`~enum.StrEnum`, so it is a
plain string everywhere one is expected - passing a member to
:meth:`~pyippdme.client.IppDmeClient.call` or comparing it against wire
text both work exactly as if it were the string itself.

This only covers the built-in classes' own commands (see
:mod:`pyippdme.server.classes`); a proprietary command registered via
:meth:`~pyippdme.server.registry.CommandRegistry.command_proprietary` has no
member here; naming it well (and, ideally, exporting a constant for it too)
is the caller's own responsibility.
"""

from __future__ import annotations

from enum import StrEnum


class CommandName(StrEnum):
    """Every command name a built-in command class registers, grouped by class."""

    # Server (6.3) - session lifecycle, error handling, generic properties.
    START_SESSION = "StartSession"
    END_SESSION = "EndSession"
    STOP_DAEMON = "StopDaemon"
    STOP_ALL_DAEMONS = "StopAllDaemons"
    ABORT_E = "AbortE"
    GET_ERROR_INFO = "GetErrorInfo"
    GET_ERR_STATUS_E = "GetErrStatusE"
    GET_XTD_ERR_STATUS = "GetXtdErrStatus"
    CLEAR_ALL_ERRORS = "ClearAllErrors"
    ENUM_NAME_SPACES = "EnumNameSpaces"
    SET_PROP = "SetProp"
    GET_PROP = "GetProp"
    GET_PROP_E = "GetPropE"
    ENUM_PROP = "EnumProp"
    ENUM_ALL_PROP = "EnumAllProp"

    # DME (6.4) - mandatory machine-identity/introspection commands.
    GET_DME_VERSION = "GetDMEVersion"
    GET_SUPPORTED_COMMANDS = "GetSupportedCommands"
    GET_SUPPORTED_ARGUMENTS = "GetSupportedArguments"
    GET_MACHINE_CLASS = "GetMachineClass"
    HOME = "Home"
    IS_HOMED = "IsHomed"

    # CartCMM/Cartesian/TouchTrigger (6.5, 6.8, 6.12) - coordinate systems,
    # motion, single-point measurement.
    SET_COORD_SYSTEM = "SetCoordSystem"
    GET_COORD_SYSTEM = "GetCoordSystem"
    SET_CSY_TRANSFORMATION = "SetCsyTransformation"
    GET_CSY_TRANSFORMATION = "GetCsyTransformation"
    SAVE_NAMED_CSY_TRANSFORMATION = "SaveNamedCsyTransformation"
    GET_NAMED_CSY_TRANSFORMATION = "GetNamedCsyTransformation"
    SAVE_ACTIVE_COORD_SYSTEM = "SaveActiveCoordSystem"
    LOAD_COORD_SYSTEM = "LoadCoordSystem"
    DELETE_COORD_SYSTEM = "DeleteCoordSystem"
    ENUM_COORD_SYSTEMS = "EnumCoordSystems"
    GET = "Get"
    GO_TO = "GoTo"
    ON_PT_MEAS_REPORT = "OnPtMeasReport"
    PT_MEAS = "PtMeas"

    # TouchTrigger_SelfCentering (6.14) - self-centering point measurement.
    PT_MEAS_SELF_CENTER = "PtMeasSelfCenter"
    PT_MEAS_SELF_CENTER_LOCKED = "PtMeasSelfCenterLocked"

    STEP = "Step"
    GO_TO_ON_CIRCLE = "GoToOnCircle"
    GO_TO_ON_SPIRAL = "GoToOnSpiral"
    GET_TEMPERATURE_SENSORS = "GetTemperatureSensors"
    READ_TEMPERATURE_SENSOR = "ReadTemperatureSensor"
    READ_ALL_TEMPERATURES = "ReadAllTemperatures"

    # Tool (6.10) - the active tool's identity and parameter blocks.
    TOOL = "Tool"
    IS_ALIGNABLE = "IsAlignable"
    RE_QUALIFY = "ReQualify"

    # Alignable_AB/Alignable_ABC (6.20/6.21) - tool orientation.
    ALIGN_TOOL = "AlignTool"
    AVR_RADIUS = "AvrRadius"
    CALC_TOOL_ALIGNMENT = "CalcToolAlignment"
    CALC_TOOL_ANGLES = "CalcToolAngles"
    USE_SMALLEST_ANGLE_TO_ALIGN_TOOL = "UseSmallestAngletoAlignTool"
    ENABLE_OPTIMIZE = "EnableOptimize"
    DISABLE_OPTIMIZE = "DisableOptimize"
    IS_OPTIMIZE_ENABLED = "IsOptimizeEnabled"
    AVR_OFFSETS = "AvrOffsets"
    COLLISION_VOLUME = "CollisionVolume"
    ALIGNMENT_VOLUME = "AlignmentVolume"

    # ToolChanger (6.22) - installing/changing tools.
    ENUM_TOOLS = "EnumTools"
    CHANGE_TOOL = "ChangeTool"
    FIND_TOOL = "FindTool"
    FOUND_TOOL = "FoundTool"
    SET_TOOL = "SetTool"
    GET_CHANGE_TOOL_ACTION = "GetChangeToolAction"
    ENUM_TOOL_COLLECTION = "EnumToolCollection"
    ENUM_ALL_TOOL_COLLECTIONS = "EnumAllToolCollections"
    OPEN_TOOL_COLLECTION = "OpenToolCollection"

    # Scanning (6.13.2) - known/unknown-contour scans and their hints.
    ON_SCAN_REPORT = "OnScanReport"
    SCAN_ON_CIRCLE_HINT = "ScanOnCircleHint"
    SCAN_ON_LINE_HINT = "ScanOnLineHint"
    SCAN_ON_CURVE_HINT = "ScanOnCurveHint"
    SCAN_ON_CURVE_DENSITY = "ScanOnCurveDensity"
    SCAN_ON_CURVE = "ScanOnCurve"
    SCAN_ON_CIRCLE = "ScanOnCircle"
    SCAN_ON_LINE = "ScanOnLine"
    SCAN_ON_HELIX = "ScanOnHelix"
    SCAN_UNKNOWN_HINT = "ScanUnknownHint"
    SCAN_UNKNOWN_DENSITY = "ScanUnknownDensity"
    SCAN_IN_PLANE_END_IS_SPHERE = "ScanInPlaneEndIsSphere"
    SCAN_IN_PLANE_END_IS_PLANE = "ScanInPlaneEndIsPlane"
    SCAN_IN_PLANE_END_IS_CYL = "ScanInPlaneEndIsCyl"
    SCAN_IN_CYL_END_IS_SPHERE = "ScanInCylEndIsSphere"
    SCAN_IN_CYL_END_IS_PLANE = "ScanInCylEndIsPlane"

    # FormTester (6.6.1) - centering/tilting alignment and axis/position locking.
    CENTER_PART = "CenterPart"
    TILT_PART = "TiltPart"
    TILT_CENTER_PART = "TiltCenterPart"
    LOCK_AXIS = "LockAxis"
    LOCK_POSITION = "LockPosition"

    # Mover (6.7.1) - manual jog enable, axis enumeration, scale temperatures.
    ENABLE_USER = "EnableUser"
    DISABLE_USER = "DisableUser"
    IS_USER_ENABLED = "IsUserEnabled"
    ENUMERATE_MOVER_AXES = "EnumerateMoverAxes"
    UPDATE_SCALE_TEMPERATURES = "UpdateScaleTemperatures"
    SET_SCALE_TEMPERATURES = "SetScaleTemperatures"
    GET_SCALE_TEMPERATURES = "GetScaleTemperatures"
    SET_TEMPERATURE_COMPENSATION_ORIGIN = "SetTemperatureCompensationOrigin"
    ON_MOVE_REPORT = "OnMoveReport"
    ON_MOVE_REPORT_E = "OnMoveReportE"

    # RotaryTable (6.23) - R() is only ever an argument of Get/GoTo/PtMeas
    # (see pyippdme.simulation.classes.rotarytable_class), so it has no command name here.
    ENABLE_ROTARY_TABLE_VAR_CSY = "EnableRotaryTableVarCsy"
    ALIGN_PART = "AlignPart"

    # Optical/RawDataHandling (6.15, 6.17) - data acquisition and retrieval.
    DATA_ACQUIRE = "DataAcquire"
    # Spelled "DeleteAcquistion" (missing the second "i") in the standard
    # itself - both its own command-signature header ("DeleteAcquistion(Name)")
    # and the Annex A command index use this spelling; only the generic
    # "The X() command does Y" prose sentence under it uses the corrected
    # spelling. Treated as the standard's own typo and implemented under the
    # header/index spelling, since that is what a wire-compatible peer
    # replicating the published text would send.
    DELETE_ACQUISITION = "DeleteAcquistion"
    DELETE_ALL_ACQUISITIONS = "DeleteAllAcquisitions"
    ADV_DATA_STRUCT = "AdvDataStruct"
    RAW_DATA_BIN_SETUP = "RawDataBinSetup"
    GET_RAW_DATA_BIN = "GetRawDataBin"
    GET_RAW_DATA_SHA_MEM = "GetRawDataShaMem"
    RELEASE_SHA_MEM = "ReleaseShaMem"
    GET_RAW_DATA_FILE = "GetRawDataFile"
    DEL_RAW_DATA_FILE = "DelRawDataFile"

    # FeatureExtraction (Annex J.2, deprecated).
    ROI = "ROI"
    FEATURE_EXTRACT = "FeatureExtract"

    # Part (6.24) has no commands of its own, only properties (Temperature,
    # XpanCoefficient, Approach, Search, Retract - reached through SetProp/
    # GetProp above, see pyippdme.simulation.classes.part_class), so it adds no members.
