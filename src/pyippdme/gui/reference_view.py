# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""A manual page of a command as rich text (from the command reference)."""

from __future__ import annotations

import html

from PySide6.QtWidgets import QTextBrowser, QWidget

from pyippdme.client.reference import CommandDoc, CommandReference

__all__ = ["ReferenceView", "render_html"]

_STYLE = (
    "<style>"
    "h3 { margin-bottom: 2px; } "
    "code { font-family: monospace; } "
    "table { border-collapse: collapse; } "
    "td, th { padding: 2px 8px 2px 0; vertical-align: top; text-align: left; } "
    ".muted { color: gray; }"
    "</style>"
)


def render_html(doc: CommandDoc) -> str:
    """Return the manual page of ``doc`` as an HTML fragment."""
    esc = html.escape
    parts = [
        _STYLE,
        f"<h3>{esc(doc.name)}</h3>",
        f"<p>{esc(doc.summary)}</p>",
        f"<p><code>{esc(doc.signature)}</code></p>",
    ]
    if doc.arguments:
        rows = []
        for a in doc.arguments:
            label = esc(a.name) + ("" if a.mandatory else "*")
            kind = esc(a.datatype) + (", bare value" if a.positional else "")
            rows.append(
                f"<tr><td><code>{label}</code></td><td class='muted'>{kind}</td>"
                f"<td>{esc(a.description)}</td></tr>"
            )
        parts.append("<p><b>Arguments</b></p><table>" + "".join(rows) + "</table>")
        notes = []
        if any(not a.mandatory for a in doc.arguments):
            notes.append("* optional")
        if any(a.positional for a in doc.arguments):
            notes.append("bare values are written in the order listed, without their names")
        if notes:
            parts.append(f"<p class='muted'>{esc('; '.join(notes))}</p>")
    else:
        parts.append("<p class='muted'>No arguments.</p>")
    if doc.returns:
        parts.append(f"<p><b>Returns</b><br>{esc(doc.returns)}</p>")
    if doc.notes:
        parts.append(f"<p>{esc(doc.notes)}</p>")
    parts.append(f"<p><b>Example</b><br><code>{esc(doc.example)}</code></p>")
    parts.append(f"<p class='muted'>Group: {esc(doc.group)}</p>")
    return "".join(parts)


class ReferenceView(QTextBrowser):
    """Shows what a command is, its arguments and what it returns."""

    def __init__(
        self, reference: CommandReference | None = None, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.reference = reference or CommandReference()
        self.setOpenLinks(False)
        self.show_hint()

    def show_hint(self) -> None:
        """Show what to do to see a command."""
        self.setHtml(
            "<p class='muted'>Pick a command in the list on the left to read what it does, "
            "what its arguments mean and what it returns.</p>"
        )

    def show_command(self, name: str) -> bool:
        """Show the page of ``name``; return whether the command is known."""
        doc = self.reference.find(name)
        if doc is None:
            self.setHtml(f"<p>No command {html.escape(name)!r}.</p>")
            return False
        self.setHtml(render_html(doc))
        return True
