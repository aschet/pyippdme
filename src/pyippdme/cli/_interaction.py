# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Moved to :mod:`pyippdme.client.interaction`; this name stays for existing imports."""

from __future__ import annotations

from pyippdme.client.interaction import (
    Acked,
    CommandEvent,
    Completed,
    ConnectionLost,
    Failed,
    ParseFailed,
    Received,
    format_error,
    run_command_line,
)

__all__ = [
    "Acked",
    "CommandEvent",
    "Completed",
    "ConnectionLost",
    "Failed",
    "ParseFailed",
    "Received",
    "format_error",
    "run_command_line",
]
