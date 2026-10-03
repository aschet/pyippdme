# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Loading of the packaged SVG icons, colored from the current palette.

The files in ``icons/`` use four colour roles that are replaced by palette colours when painting,
so the icons follow light and dark themes and look disabled when they are.
"""

from __future__ import annotations

import re
from functools import cache
from pathlib import Path
from typing import NamedTuple

from PySide6.QtCore import QPoint, QRect, QRectF, QSize, Qt
from PySide6.QtGui import (
    QColor,
    QGuiApplication,
    QIcon,
    QIconEngine,
    QImage,
    QPainter,
    QPalette,
    QPixmap,
)
from PySide6.QtSvg import QSvgRenderer

__all__ = ["app_icon", "icon_names", "load_icon"]

_ICON_DIR = Path(__file__).parent / "icons"

# Color roles in the icon files: main strokes, secondary strokes, light fills and the accent.
_INK, _MID, _LIGHT, _ACCENT = "#2f6fb0", "#7fb8d8", "#dbe8f6", "#e8833a"

_MODE = QIcon.Mode


class _Colors(NamedTuple):
    ink: str
    mid: str
    light: str
    accent: str


def _mix(ink: QColor, background: QColor, share: float) -> str:
    def blend(a: int, b: int) -> int:
        return round(a * share + b * (1 - share))

    return QColor(
        blend(ink.red(), background.red()),
        blend(ink.green(), background.green()),
        blend(ink.blue(), background.blue()),
    ).name()


def _colors(mode: QIcon.Mode, palette: QPalette) -> _Colors:
    background = palette.color(QPalette.ColorRole.Window)
    if mode == _MODE.Disabled:
        ink = palette.color(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText)
        accent = ink.name()
    else:
        ink = palette.color(QPalette.ColorRole.ButtonText)
        accent = _ACCENT
    return _Colors(ink.name(), _mix(ink, background, 0.6), _mix(ink, background, 0.15), accent)


@cache
def _source(name: str) -> str:
    return (_ICON_DIR / f"{name}.svg").read_text(encoding="utf-8")


def _themed(name: str, colors: _Colors) -> bytes:
    table = {_INK: colors.ink, _MID: colors.mid, _LIGHT: colors.light, _ACCENT: colors.accent}
    return re.sub(
        r"#[0-9a-f]{6}", lambda m: table.get(m.group(), m.group()), _source(name)
    ).encode()


class _ThemedSvgEngine(QIconEngine):
    """Paints an SVG icon directly, using the application palette at paint time."""

    def __init__(self, name: str) -> None:
        super().__init__()
        self._name = name
        self._renderers: dict[_Colors, QSvgRenderer] = {}

    def paint(self, painter: QPainter, rect: QRect, mode: QIcon.Mode, state: QIcon.State) -> None:
        colors = _colors(mode, QGuiApplication.palette())
        renderer = self._renderers.get(colors)
        if renderer is None:
            renderer = QSvgRenderer(_themed(self._name, colors))
            self._renderers[colors] = renderer
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        renderer.render(painter, QRectF(rect))
        painter.restore()

    def actualSize(  # noqa: N802
        self, size: QSize, mode: QIcon.Mode, state: QIcon.State
    ) -> QSize:
        return size

    def pixmap(self, size: QSize, mode: QIcon.Mode, state: QIcon.State) -> QPixmap:
        image = QImage(size, QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(image)
        self.paint(painter, QRect(QPoint(0, 0), size), mode, state)
        painter.end()
        return QPixmap.fromImage(image)

    def clone(self) -> QIconEngine:
        return _ThemedSvgEngine(self._name)


def load_icon(name: str) -> QIcon:
    """Return the packaged icon ``name`` (file name without ``.svg``), colored from the palette."""
    return QIcon(_ThemedSvgEngine(name))


def app_icon() -> QIcon:
    """Return the application icon, which keeps its own colors."""
    return QIcon(str(_ICON_DIR / "app.svg"))


def icon_names() -> list[str]:
    """Return the names :func:`load_icon` accepts."""
    return sorted(p.stem for p in _ICON_DIR.glob("*.svg") if p.stem != "app")
