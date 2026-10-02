# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Error classification and the pre-defined error table (5.6, Annex B)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum


class ErrorSeverity(IntEnum):
    """F1: default error severity classification (Table 4)."""

    INFO = 0
    WARNING = 1
    ERROR = 2
    CRITICAL = 3
    FATAL = 9

    @property
    def requires_clear_all_errors(self) -> bool:
        """Whether a client must send ``ClearAllErrors()`` before continuing."""
        return self >= ErrorSeverity.ERROR


@dataclass(frozen=True, slots=True)
class ServerError(Exception):
    """A parsed ``Error(F1, F2, F3, Text)`` response (5.6.3)."""

    severity: ErrorSeverity
    number: str
    """F2: 4-digit error number identifying the error, e.g. ``"1008"``."""
    cause: str
    """F3: name of the method that reported the error."""
    text: str
    """Additional human-readable information."""

    def __str__(self) -> str:
        return f"[{self.severity.name} {self.number}] {self.cause}: {self.text}"


class ErrorCode(StrEnum):
    """4-digit error numbers pre-defined by I++ DME (Annex B).

    A :class:`str` subtype (``ErrorCode.BAD_ARGUMENT == "0509"`` is ``True``),
    so it can be passed anywhere :class:`ServerError`'s ``number`` field or a
    wire-text comparison expects a plain string - the point of naming these
    is to replace scattered raw ``"0509"``-style literals at every call site
    that raises a pre-defined error with a name that a typo in the digits
    can't silently slip past (mypy/the IDE catch an unknown *name*; nothing
    catches a wrong but well-formed 4-digit string).

    Numbers below 8000 are reserved by the specification; 8000-8999 and
    9000-9999 are free for server- and client-defined proprietary errors
    respectively (5.6.3), so are intentionally not members here - a real
    error number outside this enum is not necessarily invalid, just not
    one of these pre-defined ones.
    """

    BUFFER_FULL = "0000"
    ILLEGAL_TAG = "0001"
    NO_SPACE_AT_POSITION_6 = "0002"
    RESERVED_0003 = "0003"
    RESERVED_0004 = "0004"
    RESERVED_0005 = "0005"
    TRANSACTION_ABORTED = "0006"
    ILLEGAL_CHARACTER = "0007"
    PROTOCOL_ERROR = "0008"
    EMERGENCY_STOP = "0500"
    UNSUPPORTED_COMMAND = "0501"
    INCORRECT_ARGUMENTS = "0502"
    CONTROLLER_COMMUNICATIONS_FAILURE = "0503"
    ARGUMENT_OUT_OF_RANGE = "0504"
    ARGUMENT_NOT_RECOGNIZED = "0505"
    ARGUMENT_NOT_SUPPORTED = "0506"
    ILLEGAL_COMMAND = "0507"
    BAD_CONTEXT = "0508"
    BAD_ARGUMENT = "0509"
    BAD_PROPERTY = "0510"
    ERROR_PROCESSING_METHOD = "0511"
    NO_DAEMONS_ACTIVE = "0512"
    DAEMON_DOES_NOT_EXIST = "0513"
    USE_CLEAR_ALL_ERRORS = "0514"
    DAEMON_ALREADY_EXISTS = "0515"
    MACHINE_IN_ERROR_STATE = "1000"
    ILLEGAL_TOUCH = "1001"
    AXIS_DOES_NOT_EXIST = "1002"
    NO_TOUCH = "1003"
    ANGLES_NOT_SUPPORTED = "1004"
    ERROR_DURING_HOME = "1005"
    SURFACE_NOT_FOUND = "1006"
    THETA_OUT_OF_RANGE = "1007"
    TARGET_OUT_OF_MACHINE_VOLUME = "1008"
    AIR_PRESSURE_OUT_OF_RANGE = "1009"
    VECTOR_HAS_NO_NORM = "1010"
    UNABLE_TO_MOVE = "1011"
    BAD_LOCK_COMBINATIONS = "1012"
    COORDINATE_SYSTEM_NOT_FOUND = "1013"
    ACTIVE_INTERNAL_CORRECTION = "1014"
    FAILED_TO_RESEAT_HEAD = "1500"
    PROBE_NOT_ARMED = "1501"
    TOOL_NOT_FOUND = "1502"
    TOOL_NOT_DEFINED = "1503"
    COLLECTION_NOT_FOUND = "1504"
    TOOL_NOT_ALIGNABLE = "1505"
    TOOL_PROPERTY_NOT_APPLICABLE = "1506"
    TOOL_NOT_ALIGNABLE_TO_ORIENTATION = "1507"
    TOOL_NOT_CALIBRATED = "2000"
    HEAD_ERROR_EXCESSIVE_FORCE = "2001"
    PROBE_TYPE_NOT_ALLOWED = "2002"
    RAW_DATA_NOT_AVAILABLE = "2003"
    NO_SHARED_MEMORY_AVAILABLE = "2004"
    FILE_NOT_FOUND = "2005"
    MACHINE_LIMIT_ENCOUNTERED = "2500"
    AXIS_NOT_ACTIVE = "2501"
    AXIS_POSITION_ERROR = "2502"
    SCALE_READ_HEAD_FAILURE = "2503"
    COLLISION = "2504"
    ANGLE_OUT_OF_RANGE = "2505"
    PART_NOT_ALIGNED = "2506"
    WRONG_DATA_FORMAT = "5000"
    PORT_NOT_AVAILABLE = "5001"


#: Pre-defined errors of I++ DME (Annex B), keyed by 4-digit number.
DEFAULT_ERRORS: dict[ErrorCode, tuple[ErrorSeverity, str]] = {
    ErrorCode.BUFFER_FULL: (ErrorSeverity.INFO, "Buffer full"),
    ErrorCode.ILLEGAL_TAG: (ErrorSeverity.ERROR, "Illegal tag"),
    ErrorCode.NO_SPACE_AT_POSITION_6: (ErrorSeverity.ERROR, "No space at pos. 6"),
    ErrorCode.RESERVED_0003: (ErrorSeverity.ERROR, "Reserved"),
    ErrorCode.RESERVED_0004: (ErrorSeverity.ERROR, "Reserved"),
    ErrorCode.RESERVED_0005: (ErrorSeverity.ERROR, "Reserved"),
    ErrorCode.TRANSACTION_ABORTED: (
        ErrorSeverity.ERROR,
        "Transaction aborted (Use ClearAllErrors To Continue)",
    ),
    ErrorCode.ILLEGAL_CHARACTER: (ErrorSeverity.CRITICAL, "Illegal character"),
    ErrorCode.PROTOCOL_ERROR: (ErrorSeverity.CRITICAL, "Protocol error"),
    ErrorCode.EMERGENCY_STOP: (ErrorSeverity.CRITICAL, "Emergency stop"),
    ErrorCode.UNSUPPORTED_COMMAND: (ErrorSeverity.CRITICAL, "Unsupported command"),
    ErrorCode.INCORRECT_ARGUMENTS: (ErrorSeverity.CRITICAL, "Incorrect arguments"),
    ErrorCode.CONTROLLER_COMMUNICATIONS_FAILURE: (
        ErrorSeverity.FATAL,
        "Controller communications failure",
    ),
    ErrorCode.ARGUMENT_OUT_OF_RANGE: (ErrorSeverity.WARNING, "Argument out of range"),
    ErrorCode.ARGUMENT_NOT_RECOGNIZED: (ErrorSeverity.CRITICAL, "Argument not recognized"),
    ErrorCode.ARGUMENT_NOT_SUPPORTED: (ErrorSeverity.CRITICAL, "Argument not supported"),
    ErrorCode.ILLEGAL_COMMAND: (ErrorSeverity.CRITICAL, "Illegal command"),
    ErrorCode.BAD_CONTEXT: (ErrorSeverity.CRITICAL, "Bad context"),
    ErrorCode.BAD_ARGUMENT: (ErrorSeverity.CRITICAL, "Bad argument"),
    ErrorCode.BAD_PROPERTY: (ErrorSeverity.CRITICAL, "Bad property"),
    ErrorCode.ERROR_PROCESSING_METHOD: (ErrorSeverity.CRITICAL, "Error processing method"),
    ErrorCode.NO_DAEMONS_ACTIVE: (ErrorSeverity.WARNING, "No daemons are active"),
    ErrorCode.DAEMON_DOES_NOT_EXIST: (ErrorSeverity.ERROR, "Daemon does not exist"),
    ErrorCode.USE_CLEAR_ALL_ERRORS: (ErrorSeverity.ERROR, "Use ClearAllErrors to continue"),
    ErrorCode.DAEMON_ALREADY_EXISTS: (ErrorSeverity.ERROR, "Daemon already exists"),
    ErrorCode.MACHINE_IN_ERROR_STATE: (ErrorSeverity.CRITICAL, "Machine in error state"),
    ErrorCode.ILLEGAL_TOUCH: (ErrorSeverity.ERROR, "Illegal touch"),
    ErrorCode.AXIS_DOES_NOT_EXIST: (ErrorSeverity.FATAL, "Axis does not exist"),
    ErrorCode.NO_TOUCH: (ErrorSeverity.ERROR, "No touch"),
    ErrorCode.ANGLES_NOT_SUPPORTED: (
        ErrorSeverity.FATAL,
        "Number of angles not supported on current device",
    ),
    ErrorCode.ERROR_DURING_HOME: (ErrorSeverity.CRITICAL, "Error during home"),
    ErrorCode.SURFACE_NOT_FOUND: (ErrorSeverity.ERROR, "Surface not found"),
    ErrorCode.THETA_OUT_OF_RANGE: (ErrorSeverity.CRITICAL, "Theta out of range"),
    ErrorCode.TARGET_OUT_OF_MACHINE_VOLUME: (
        ErrorSeverity.CRITICAL,
        "Target position out of machine volume",
    ),
    ErrorCode.AIR_PRESSURE_OUT_OF_RANGE: (ErrorSeverity.CRITICAL, "Air pressure out of range"),
    ErrorCode.VECTOR_HAS_NO_NORM: (ErrorSeverity.ERROR, "Vector has no norm"),
    ErrorCode.UNABLE_TO_MOVE: (ErrorSeverity.ERROR, "Unable to move"),
    ErrorCode.BAD_LOCK_COMBINATIONS: (ErrorSeverity.ERROR, "Bad lock combinations"),
    ErrorCode.COORDINATE_SYSTEM_NOT_FOUND: (ErrorSeverity.CRITICAL, "Coordinate system not found"),
    ErrorCode.ACTIVE_INTERNAL_CORRECTION: (
        ErrorSeverity.ERROR,
        "Failed to set value due to active internal correction",
    ),
    ErrorCode.FAILED_TO_RESEAT_HEAD: (ErrorSeverity.CRITICAL, "Failed to re-seat head"),
    ErrorCode.PROBE_NOT_ARMED: (ErrorSeverity.CRITICAL, "Probe not armed"),
    ErrorCode.TOOL_NOT_FOUND: (ErrorSeverity.CRITICAL, "Tool not found"),
    ErrorCode.TOOL_NOT_DEFINED: (ErrorSeverity.CRITICAL, "Tool not defined"),
    ErrorCode.COLLECTION_NOT_FOUND: (ErrorSeverity.CRITICAL, "Collection not found"),
    ErrorCode.TOOL_NOT_ALIGNABLE: (ErrorSeverity.ERROR, "Tool not alignable"),
    ErrorCode.TOOL_PROPERTY_NOT_APPLICABLE: (
        ErrorSeverity.WARNING,
        "Tool property not applicable",
    ),
    ErrorCode.TOOL_NOT_ALIGNABLE_TO_ORIENTATION: (
        ErrorSeverity.CRITICAL,
        "Tool not alignable to given orientation",
    ),
    ErrorCode.TOOL_NOT_CALIBRATED: (ErrorSeverity.CRITICAL, "Tool not calibrated"),
    ErrorCode.HEAD_ERROR_EXCESSIVE_FORCE: (ErrorSeverity.ERROR, "Head error excessive force"),
    ErrorCode.PROBE_TYPE_NOT_ALLOWED: (
        ErrorSeverity.CRITICAL,
        "Type of probe does not allow this operation",
    ),
    ErrorCode.RAW_DATA_NOT_AVAILABLE: (
        ErrorSeverity.CRITICAL,
        "Raw data of Acquisition not available",
    ),
    ErrorCode.NO_SHARED_MEMORY_AVAILABLE: (
        ErrorSeverity.CRITICAL,
        "No shared memory space available",
    ),
    ErrorCode.FILE_NOT_FOUND: (ErrorSeverity.CRITICAL, "File not found"),
    ErrorCode.MACHINE_LIMIT_ENCOUNTERED: (
        ErrorSeverity.CRITICAL,
        "Machine limit encountered [Move Out Of Limits]",
    ),
    ErrorCode.AXIS_NOT_ACTIVE: (ErrorSeverity.CRITICAL, "Axis not active"),
    ErrorCode.AXIS_POSITION_ERROR: (ErrorSeverity.CRITICAL, "Axis position error"),
    ErrorCode.SCALE_READ_HEAD_FAILURE: (ErrorSeverity.FATAL, "Scale read head failure"),
    ErrorCode.COLLISION: (ErrorSeverity.CRITICAL, "Collision"),
    ErrorCode.ANGLE_OUT_OF_RANGE: (ErrorSeverity.ERROR, "Specified angle out of range"),
    ErrorCode.PART_NOT_ALIGNED: (ErrorSeverity.ERROR, "Part not aligned"),
    ErrorCode.WRONG_DATA_FORMAT: (ErrorSeverity.ERROR, "Wrong data format"),
    ErrorCode.PORT_NOT_AVAILABLE: (ErrorSeverity.ERROR, "Port not available"),
}


def describe(number: str) -> tuple[ErrorSeverity, str] | None:
    """Look up the default severity/text of a pre-defined error number, if known.

    ``number`` need not itself be an :class:`ErrorCode` - a plain wire-text
    string (e.g. parsed from a remote server's response, which may well be
    a proprietary 8000-9999 code this enum intentionally excludes) is
    resolved the same way, returning ``None`` for anything not pre-defined.
    """
    try:
        code = ErrorCode(number)
    except ValueError:
        return None
    return DEFAULT_ERRORS.get(code)
