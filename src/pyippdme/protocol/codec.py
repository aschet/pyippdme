# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Line-level framing: attach/strip the mandatory ``<CR><LF>`` terminator.

Section 5.1 mandates ASCII 32-126 content terminated by a
``<CR><LF>`` pair, sent together. The standard does not define a maximum
line length; ``MAX_LINE_LENGTH`` is this library's own memory-safety bound
against a malformed or hostile peer sending an unbounded line, sized to
comfortably fit the largest payloads this library itself produces (e.g. a
``Tool.Id()``/raw point cloud XML document, see :mod:`pyippdme.types.tool_id`/
:mod:`pyippdme.types.pointcloud`), not a spec-mandated value.
"""

from __future__ import annotations

from pyippdme.exceptions import IppDmeProtocolError
from pyippdme.protocol.ast import Command, Response
from pyippdme.protocol.parser import parse_command, parse_response

TERMINATOR = "\r\n"
MAX_LINE_LENGTH = 8 * 1024 * 1024


def encode_line(node: Command | Response) -> bytes:
    """Render a :class:`Command` or :class:`Response` as a terminated wire line."""
    text = node.to_wire()
    if not text.isascii() or any(not (32 <= ord(c) <= 126) for c in text):
        raise IppDmeProtocolError(f"Line contains non-printable-ASCII content: {text!r}")
    if len(text) > MAX_LINE_LENGTH:
        raise IppDmeProtocolError(f"Line exceeds {MAX_LINE_LENGTH} characters")
    return (text + TERMINATOR).encode("ascii")


def _strip_terminator(line: str) -> str:
    if not line.endswith(TERMINATOR):
        raise IppDmeProtocolError(f"Line is missing the <CR><LF> terminator: {line!r}")
    return line[: -len(TERMINATOR)]


def decode_command_line(line: str) -> Command:
    """Parse a raw command line, including its ``<CR><LF>`` terminator."""
    return parse_command(_strip_terminator(line))


def decode_response_line(line: str) -> Response:
    """Parse a raw response line, including its ``<CR><LF>`` terminator."""
    return parse_response(_strip_terminator(line))
