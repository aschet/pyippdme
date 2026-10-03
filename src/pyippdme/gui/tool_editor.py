# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""An editor for a :class:`~pyippdme.twin.spec.ToolSpec`, generated from its dataclass fields."""

from __future__ import annotations

from dataclasses import fields, replace
from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLineEdit,
    QSpinBox,
    QWidget,
)

from pyippdme.twin.spec import HEADS, MODES, TIPS, ToolSpec

_CHOICES = {"mode": MODES, "head": HEADS, "tip": TIPS}
#: Fields that are not edited here.
_SKIP = ("name", "color", "rack")


class ToolEditor(QWidget):
    """One input per field of ``ToolSpec``; new fields of the dataclass appear without changes."""

    edited = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._form = QFormLayout(self)
        self._widgets: dict[str, QWidget] = {}
        self._spec = ToolSpec("tool")
        self._loading = False
        for f in fields(ToolSpec):
            if f.name in _SKIP:
                continue
            widget = self._make(f.name, getattr(self._spec, f.name))
            self._widgets[f.name] = widget
            self._form.addRow(f.name.replace("_", " "), widget)

    def _make(self, name: str, default: Any) -> QWidget:
        widget: QWidget
        if name in _CHOICES:
            box = QComboBox()
            box.addItems(list(_CHOICES[name]))
            box.currentTextChanged.connect(lambda _: self._changed())
            widget = box
        elif isinstance(default, bool):
            check = QCheckBox()
            check.toggled.connect(lambda _: self._changed())
            widget = check
        elif isinstance(default, int):
            spin = QSpinBox()
            spin.setRange(1, 100000)
            spin.setKeyboardTracking(False)
            spin.valueChanged.connect(lambda _: self._changed())
            widget = spin
        elif isinstance(default, str):
            line = QLineEdit()
            line.editingFinished.connect(self._changed)
            widget = line
        else:  # floats, and the optional touch speed (0 = use the protocol's PtMeasPar speed)
            dspin = QDoubleSpinBox()
            dspin.setRange(0.0, 100000.0)
            dspin.setDecimals(3)
            dspin.setKeyboardTracking(False)
            dspin.valueChanged.connect(lambda _: self._changed())
            widget = dspin
        return widget

    def load(self, spec: ToolSpec) -> None:
        self._loading = True
        self._spec = spec
        for name, widget in self._widgets.items():
            value = getattr(spec, name)
            if isinstance(widget, QComboBox):
                widget.setCurrentText(str(value))
            elif isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            elif isinstance(widget, QSpinBox):
                widget.setValue(int(value))
            elif isinstance(widget, QLineEdit):
                widget.setText(str(value))
            elif isinstance(widget, QDoubleSpinBox):
                widget.setValue(0.0 if value is None else float(value))
        self._loading = False

    def spec(self) -> ToolSpec:
        """Return the spec with the inputs' values; an invalid combination raises ``ValueError``."""
        values: dict[str, Any] = {}
        for name, widget in self._widgets.items():
            if isinstance(widget, QComboBox):
                values[name] = widget.currentText()
            elif isinstance(widget, QCheckBox):
                values[name] = widget.isChecked()
            elif isinstance(widget, QSpinBox):
                values[name] = widget.value()
            elif isinstance(widget, QLineEdit):
                values[name] = widget.text()
            elif isinstance(widget, QDoubleSpinBox):
                value = widget.value()
                values[name] = None if name == "touch_speed" and value == 0.0 else value
        return replace(self._spec, **values)

    def _changed(self) -> None:
        if not self._loading:
            self.edited.emit()
