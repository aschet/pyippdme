# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Exception hierarchy used throughout pyippdme."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyippdme.protocol.errors import ServerError


class IppDmeError(Exception):
    """Base class for all errors raised by pyippdme."""


class IppDmeProtocolError(IppDmeError):
    """The wire data violates the I++ DME grammar (5.2)."""


class IppDmeConnectionError(IppDmeError):
    """The TCP connection to the peer failed or was closed unexpectedly."""


class IppDmeTimeoutError(IppDmeError):
    """A transaction did not complete within the configured timeout."""


class IppDmeServerError(IppDmeError):
    """A server reported an ``Error(...)`` response for a transaction.

    Wraps :class:`pyippdme.protocol.errors.ServerError`, which carries the
    F1/F2/F3/Text fields (5.6).
    """

    def __init__(self, error: ServerError) -> None:
        self.error = error
        super().__init__(str(error))
