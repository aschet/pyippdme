# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""A table that shows the evaluation of the measured points against the check artefact."""

from __future__ import annotations

from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import QTableWidget, QTableWidgetItem, QWidget

from pyippdme.twin.features import Result

_COLUMNS = ("mode", "feature", "quantity", "n", "nominal", "measured", "error µm", "limit µm", "")


class EvaluationTable(QTableWidget):
    """Shows ``DigitalTwin.evaluate()``: what each mode measured and whether it is within the MPE."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(0, len(_COLUMNS), parent)
        self.setHorizontalHeaderLabels(list(_COLUMNS))
        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.setAlternatingRowColors(True)

    def show_results(self, results: dict[str, list[Result]]) -> None:
        rows = [(mode, r) for mode, items in results.items() for r in items]
        self.setRowCount(len(rows))
        for row, (mode, r) in enumerate(rows):
            limit = "" if r.limit is None else f"{r.limit * 1000:.1f}"
            verdict = "" if r.ok is None else ("ok" if r.ok else "over")
            cells = (
                mode,
                r.feature,
                r.quantity,
                str(r.points),
                f"{r.nominal:.4f}",
                f"{r.measured:.4f}",
                f"{r.error * 1000:+.1f}",
                limit,
                verdict,
            )
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if col == len(cells) - 1 and verdict:
                    item.setForeground(QBrush(QColor(70, 170, 90) if r.ok else QColor(220, 90, 70)))
                self.setItem(row, col, item)
        self.resizeColumnsToContents()
