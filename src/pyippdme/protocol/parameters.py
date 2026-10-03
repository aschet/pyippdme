# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Names this library's built-in command classes use as ``NamedValue``/``BasicName`` wire text.

Where :class:`~pyippdme.protocol.commands.CommandName` names the ``method`` half of a
wire line (``GoTo(...)``), :class:`ParameterName` names the argument/response
identifiers that appear *inside* the parentheses - ``X`` in ``GoTo(X(10))``,
``Center``/``IJK`` in ``GoToOnCircle(Center(...), IJK(...), ...)``, and so
on - so a caller building a request by hand (via the low-level
:class:`~pyippdme.client.IppDmeClient` API, ``NamedValue(ParameterName.X, ...)``
instead of ``NamedValue("X", ...)``) or reading a response has the same
typo-safety :class:`~pyippdme.protocol.commands.CommandName` already gives command
names. :class:`ParameterName` is a :class:`~enum.StrEnum`, so a member is a
plain string everywhere one is expected.

This only covers *atomic* identifiers: a single ``NamedValue``/``BasicName``
literal this library's built-in classes construct or parse verbatim, such as
``X``/``Center``/``Format``. It deliberately does not cover:

- Purely positional arguments (e.g. ``ScanOnLine``'s ``Sx, Sy, Sz, ...``),
  since those names never appear as wire text at all - only as descriptive
  labels in :class:`~pyippdme.protocol.signature.Parameter` for
  ``GetSupportedArguments``.
- Enum/name-kind top-level :class:`~pyippdme.protocol.signature.Parameter` names like
  ``GoTo``'s ``Positions`` or ``EnumProp``'s ``Reference``, which are also
  ``GetSupportedArguments`` schema labels only - the wire text a caller
  actually sends is the enum's *members* (``X()``, ``Y()``, ...), which are
  covered here, or an arbitrary caller-supplied name (``EnumProp("Tool")``),
  which by definition cannot be enumerated.
- Compound, dotted property paths (``Tool.Id``, ``Tool.GoToPar.Speed``,
  ``Part.Temperature``, ...), since those are compositions built from a
  parameter-block name and a leaf name, not atomic constants - see
  :mod:`pyippdme.simulation.tool` and :mod:`pyippdme.simulation.classes.part_class` for that
  namespace instead.
"""

from __future__ import annotations

from enum import StrEnum


class ParameterName(StrEnum):
    """Every atomic ``NamedValue``/``BasicName`` identifier a built-in class uses on the wire."""

    # Coordinate axes / RotaryTable (GoTo/Get/PtMeas/Step/OnPtMeasReport, 6.5/6.8/6.12/6.23).
    # ER and Q are OnScanReport/OnPtMeasReport-only report fields (6.10.2, Tables
    # 66 and 70), not settable axes.
    X = "X"
    Y = "Y"
    Z = "Z"
    R = "R"
    Q = "Q"
    ER = "ER"

    # Coordinate-system transformation (CartCMM 6.5.1: [Get/Set]CsyTransformation and friends).
    X0 = "X0"
    Y0 = "Y0"
    Z0 = "Z0"
    THETA = "Theta"
    PSI = "Psi"
    PHI = "Phi"

    # Circular/spiral/helical motion and scanning geometry (6.8.1, 6.13.2.1).
    CENTER = "Center"
    IJK = "IJK"
    # OnPtMeasReport/OnScanReport-only report field (Table 32; Tables 75/78 list it).
    IJK_ACT = "IJKAct"
    # PtMeasSelfCenterLocked's plane-constraint vector (6.14.1).
    LMN = "LMN"

    # ScanOnCurve (6.13.2.1, Table 85).
    CLOSED = "Closed"
    FORMAT = "Format"
    RT = "RT"
    DATA = "Data"
    TAG = "tag"

    # ScanOnCurveDensity/ScanUnknownDensity hints (6.13.2, Tables 84/88).
    DIS = "Dis"
    ANGLE = "Angle"
    ANGLE_BASE_LENGTH = "AngleBaseLength"
    AT_NOMINALS = "AtNominals"

    # Server (6.3.1): GetXtdErrStatus's per-error response fields.
    ACTIVE_ERROR = "ActiveError"
    SEVERITY = "Severity"

    # CartCMM temperature sensors (6.5.2): GetTemperatureSensors' per-sensor response fields.
    NAME = "Name"
    KIND = "Kind"
    SCALE = "Scale"
    CMM_TEMP_CORRECTION = "CMMTempCorrection"
    TEMPERATURE = "Temperature"

    # ToolChanger (6.22): GetChangeToolAction's response field.
    ARGUMENT = "Argument"

    # RawDataHandling (6.17): GetRawDataShaMem/GetRawDataFile response fields.
    SHA_MEM_NAME = "ShaMemName"
    DATA_SEGMENT = "DataSegment"
    SIZE = "Size"
    FILE_URL = "FileURL"
