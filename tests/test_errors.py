# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

from pyippdme.protocol.errors import DEFAULT_ERRORS, ErrorSeverity, ServerError, describe


def test_severity_requires_clear_all_errors_threshold() -> None:
    assert not ErrorSeverity.INFO.requires_clear_all_errors
    assert not ErrorSeverity.WARNING.requires_clear_all_errors
    assert ErrorSeverity.ERROR.requires_clear_all_errors
    assert ErrorSeverity.CRITICAL.requires_clear_all_errors
    assert ErrorSeverity.FATAL.requires_clear_all_errors


def test_describe_known_error() -> None:
    result = describe("1008")
    assert result == (ErrorSeverity.CRITICAL, "Target position out of machine volume")


def test_describe_unknown_error() -> None:
    assert describe("9999") is None


def test_default_errors_table_is_well_formed() -> None:
    assert len(DEFAULT_ERRORS) > 40
    for number, (severity, text) in DEFAULT_ERRORS.items():
        assert len(number) == 4
        assert number.isdigit()
        assert isinstance(severity, ErrorSeverity)
        assert text


def test_server_error_str() -> None:
    error = ServerError(
        ErrorSeverity.CRITICAL, "1008", "GoTo", "Target position out of machine volume"
    )
    assert str(error) == "[CRITICAL 1008] GoTo: Target position out of machine volume"
