# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""The coordinate systems of the machine: where they are, which one the client works in.

The chain of 6.5.1 (``MachineCsy`` to ``PartCsy``) is listed with what the client placed
(``SetCsyTransformation``, ``LoadCoordSystem``). A row picked here is drawn bold in the 3D view,
and a combo box chooses the system the position read-out above the view is shown in. The client
owns the active system; this panel only shows it.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from pyippdme.gui.widgets import mono
from pyippdme.twin import DigitalTwin
from pyippdme.twin.twin import CsyFrame
from pyippdme.types.csy import CSY_CHAIN

__all__ = ["CsyPanel"]

_HEADERS = ["System", "Client", "X", "Y", "Z", "Theta", "Psi", "Phi"]


class CsyPanel(QWidget):
    """List of the coordinate systems, with the selection that the 3D view and read-out use."""

    #: The system picked in the table (``None`` for no selection).
    selected = Signal(object)
    #: The system the position read-out is shown in.
    readout_changed = Signal(str)
    #: Show or hide the triads in the 3D view.
    show_changed = Signal(bool)

    def __init__(self, twin: DigitalTwin, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.twin = twin
        self._signature: object = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        self.table = QTableWidget(len(CSY_CHAIN), len(_HEADERS))
        self.table.setHorizontalHeaderLabels(_HEADERS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self._row_picked)
        mono(self.table, 9)
        layout.addWidget(self.table, 1)
        form = QFormLayout()
        self.show_check = QCheckBox("Draw the systems in the 3D view")
        self.show_check.setChecked(True)
        self.show_check.toggled.connect(self.show_changed.emit)
        self.readout_combo = QComboBox()
        self.readout_combo.addItems(CSY_CHAIN)
        self.readout_combo.setToolTip(
            "The coordinate system of the position above the 3D view; a client works in its own"
        )
        self.readout_combo.currentTextChanged.connect(self.readout_changed.emit)
        self.position = QLabel("")
        mono(self.position, 10)
        form.addRow(self.show_check)
        form.addRow("Show the position in", self.readout_combo)
        form.addRow("Position", self.position)
        layout.addLayout(form)
        self.note = QLabel(
            "A client chooses its system with SetCoordSystem and places systems with "
            "SetCsyTransformation. The active one is marked; a system that coincides with "
            "the machine system has no triad of its own."
        )
        self.note.setWordWrap(True)
        self.note.setStyleSheet("color: gray;")
        layout.addWidget(self.note)
        self.refresh()

    def _row_picked(self) -> None:
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        self.selected.emit(CSY_CHAIN[rows[0].row()] if rows else None)

    def refresh(self) -> None:
        """Update the table and the read-out; cheap when nothing changed."""
        frames = self.twin.csy_frames()
        position = self.twin.snapshot().position
        name = self.readout_combo.currentText()
        x, y, z = self.twin.to_csy(position, name)
        self.position.setText(f"X {x:9.3f}  Y {y:9.3f}  Z {z:9.3f}")
        signature = tuple((f.name, f.active, f.matrix.tobytes()) for f in frames)
        if signature == self._signature:
            return
        self._signature = signature
        for row, frame in enumerate(frames):
            self._fill(row, frame)

    def _fill(self, row: int, frame: CsyFrame) -> None:
        placement = frame.placement
        values = (
            ["", "", "", "", "", ""]
            if placement is None
            else [
                f"{v:.3f}"
                for v in (
                    placement.x0,
                    placement.y0,
                    placement.z0,
                    placement.theta,
                    placement.psi,
                    placement.phi,
                )
            ]
        )
        state = "active" if frame.active else ("placed" if placement is not None else "")
        if frame.name == "MachineCsy":
            state = "active" if frame.active else "base"
        elif not np.allclose(frame.matrix, np.eye(4)) and placement is None:
            state = "follows the rotary table" if frame.name == "RotaryTableVarCsy" else state
        for column, text in enumerate([frame.name, state, *values]):
            item = QTableWidgetItem(text)
            if frame.active:
                font = item.font()
                font.setBold(True)
                item.setFont(font)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, column, item)
