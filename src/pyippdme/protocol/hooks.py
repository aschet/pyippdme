# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Shared machinery for observe-only wire-line hooks, used by both client and server.

A line hook is called with one wire line's text (tag included, no
``<CR><LF>``) as it is sent or received - observe-only: it can watch
traffic (log it, record it, ...) but never delay, alter, or block it. A
hook that raises is logged and otherwise ignored, so a bug in observation
code can't break the connection. :class:`~pyippdme.client.IppDmeClient` and
:class:`~pyippdme.server.IppDmeServer` each have their own
``on_line_sent``/``on_line_received`` parameters built on this - this
module exists only so the (tiny) exception-safety wrapper isn't duplicated
between the two.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

logger = logging.getLogger("pyippdme.hooks")

LineHook = Callable[[str], None]


def call_line_hook(hook: LineHook, text: str) -> None:
    try:
        hook(text)
    except Exception:
        logger.exception("Unhandled exception in a line hook")
