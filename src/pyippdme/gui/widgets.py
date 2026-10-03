# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Small widgets and helpers the windows share."""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QAbstractButton,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QToolButton,
    QWidget,
)

from pyippdme.gui.icons import load_icon

__all__ = ["Led", "icon_button", "mono", "section_label", "spin", "tool_button"]


def spin(
    lo: float,
    hi: float,
    value: float = 0.0,
    step: float = 1.0,
    decimals: int = 3,
    suffix: str = "",
) -> QDoubleSpinBox:
    """Make a number box that applies on Enter or on losing focus, not on every keystroke."""
    box = QDoubleSpinBox()
    box.setRange(lo, hi)
    box.setDecimals(decimals)
    box.setSingleStep(step)
    box.setValue(value)
    box.setKeyboardTracking(False)
    if suffix:
        box.setSuffix(suffix)
    return box


def icon_button(icon: str, text: str = "", tip: str = "") -> QPushButton:
    """Make a push button with a themed icon."""
    button = QPushButton(load_icon(icon), text)
    if tip:
        button.setToolTip(tip)
    return button


def tool_button(icon: str, tip: str, *, checkable: bool = False, size: int = 22) -> QToolButton:
    """Make a compact icon-only button."""
    button = QToolButton()
    button.setIcon(load_icon(icon))
    button.setIconSize(QSize(size, size))
    button.setToolTip(tip)
    button.setCheckable(checkable)
    button.setAutoRaise(True)
    return button


def mono(widget: QWidget, point_size: int = 11) -> None:
    """Set a fixed-width font, for coordinates."""
    font = QFont("monospace")
    font.setStyleHint(QFont.StyleHint.Monospace)
    font.setPointSize(point_size)
    widget.setFont(font)


def section_label(text: str) -> QLabel:
    """Make a bold heading for a group of controls."""
    label = QLabel(f"<b>{text}</b>")
    label.setTextFormat(Qt.TextFormat.RichText)
    return label


class Led(QWidget):
    """A small coloured lamp with a text: green on, grey off, red for a fault."""

    def __init__(self, text: str, tip: str = "") -> None:
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self._lamp = QLabel()
        self._label = QLabel(text)
        layout.addWidget(self._lamp)
        layout.addWidget(self._label)
        self._state = "off"
        if tip:
            self.setToolTip(tip)
        self.set_state("off")

    def set_state(self, state: str, text: str | None = None) -> None:
        """``on``, ``off`` or ``error``; ``text`` replaces the label."""
        self._state = state
        if text is not None:
            self._label.setText(text)
        icon = {"on": "led-on", "error": "led-error"}.get(state, "led-off")
        self._lamp.setPixmap(load_icon(icon).pixmap(14, 14))

    @property
    def state(self) -> str:
        return self._state

    @property
    def text(self) -> str:
        return self._label.text()


def separator() -> QFrame:
    """Make a thin vertical line."""
    line = QFrame()
    line.setFrameShape(QFrame.Shape.VLine)
    line.setFrameShadow(QFrame.Shadow.Sunken)
    return line


def uncheck_all(buttons: list[QAbstractButton]) -> None:
    for button in buttons:
        button.setChecked(False)
