# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""The tool creator: build a probe system from the component library."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from pyippdme.gui.widgets import icon_button
from pyippdme.twin import DigitalTwin
from pyippdme.twin.library import LIBRARY, build_tool
from pyippdme.twin.spec import TIPS, ToolSpec
from pyippdme.twin.toolmath import OPTICAL_MODES, drop

__all__ = ["ToolCreator"]


class ToolCreator(QWidget):
    """Pick head, extension, sensor, stylus and tip; see the result; add the tool to the machine."""

    tool_created = Signal(str)

    def __init__(self, twin: DigitalTwin, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.twin = twin
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.name_edit = QLineEdit(self._free_name())
        self.name_edit.setToolTip("The name a client uses in ChangeTool (letters and digits)")
        form.addRow("Name", self.name_edit)
        self.star_tip = QComboBox()
        self.star_tip.addItems(list(TIPS))
        self.combos: dict[str, QComboBox] = {}
        for category in ("head", "extension", "sensor", "stylus", "tip"):
            combo = QComboBox()
            for component in LIBRARY[category]:
                combo.addItem(component.name)
                combo.setItemData(combo.count() - 1, component.description, 3)  # tool tip role
            self.combos[category] = combo
            form.addRow(category.capitalize(), combo)
        self.combos["sensor"].setCurrentText("touch trigger (kinematic)")
        self.combos["stylus"].setCurrentText("stem 30 mm")
        self.combos["tip"].setCurrentText("ball 3 mm")
        form.addRow("Star tip", self.star_tip)
        layout.addLayout(form)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet("color: gray;")
        layout.addWidget(self.summary)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        add = icon_button("add", "Add to the machine", "Make the tool available to clients")
        add.clicked.connect(self.add_tool)
        layout.addWidget(add)
        layout.addStretch(1)
        for combo in (*self.combos.values(), self.star_tip):
            combo.currentTextChanged.connect(self._changed)
        self._changed()

    def _free_name(self) -> str:
        names = set(self.twin.machine.tools)
        index = 1
        while f"Custom{index}" in names:
            index += 1
        return f"Custom{index}"

    def _optical(self) -> bool:
        sensor = next(c for c in LIBRARY["sensor"] if c.name == self.combos["sensor"].currentText())
        return sensor.fields.get("mode") in OPTICAL_MODES

    def spec(self) -> ToolSpec:
        """Return the tool the current choices describe (raises ``ValueError`` if invalid)."""
        optical = self._optical()
        stylus = self.combos["stylus"].currentText()
        star = stylus == "5-way star"
        return build_tool(
            self.name_edit.text().strip() or "Tool",
            sensor=self.combos["sensor"].currentText(),
            head=self.combos["head"].currentText(),
            extension=self.combos["extension"].currentText(),
            stylus=None if optical else stylus,
            tip=None if optical else self.combos["tip"].currentText(),
            star_tip=self.star_tip.currentText() if star and not optical else "down",
        )

    def _changed(self, *_: object) -> None:
        optical = self._optical()
        star = self.combos["stylus"].currentText() == "5-way star"
        self.combos["stylus"].setEnabled(not optical)
        self.combos["tip"].setEnabled(not optical)
        self.star_tip.setEnabled(star and not optical)
        try:
            spec = self.spec()
        except (ValueError, KeyError) as error:
            self.summary.setText(str(error))
            return
        self.summary.setText(
            f"{spec.mode} sensor on a {spec.head} head; the tool centre point hangs "
            f"{drop(spec):.0f} mm below the head pivot."
        )

    def add_tool(self) -> None:
        name = self.name_edit.text().strip()
        if not name or not name.isalnum():
            self.status.setText("The name must be letters and digits only")
            return
        try:
            spec = self.spec()
        except (ValueError, KeyError) as error:
            self.status.setText(f"Not created: {error}")
            return
        self.twin.set_tool_spec(spec)
        self.status.setText(f"{name} added; a client sees it in EnumTools")
        self.tool_created.emit(name)
        self.name_edit.setText(self._free_name())
