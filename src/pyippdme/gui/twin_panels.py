# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Panels of the simulator window: tool definitions, check artefact and safety.

Each panel only calls the public API of :class:`~pyippdme.twin.DigitalTwin`; none holds simulation
state of its own, so another front end can offer the same functions.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import replace
from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from pyippdme.gui.evaluation import EvaluationTable
from pyippdme.gui.icons import load_icon
from pyippdme.gui.tool_editor import ToolEditor
from pyippdme.loop import LoopThread
from pyippdme.twin import DigitalTwin
from pyippdme.twin.artifact import build_check_artifact, build_reference_sphere
from pyippdme.twin.check import MODES, CheckOptions, run_check_against_server
from pyippdme.twin.spec import ToolSpec


class ToolPanel(QWidget):
    """Edit the tool definitions of the machine; changes reach the protocol at once."""

    def __init__(self, twin: DigitalTwin, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.twin = twin
        layout = QVBoxLayout(self)
        self.tools = QListWidget()
        self.tools.setMaximumHeight(130)
        self.tools.currentTextChanged.connect(self._select)
        layout.addWidget(self.tools)
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Tool name (protocol name: letters, digits)")
        layout.addWidget(self.name_edit)
        self.editor = ToolEditor()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.editor)
        layout.addWidget(scroll, 1)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        row = QHBoxLayout()
        for text, icon, slot in (
            ("Apply", "ok", self.apply),
            ("New copy", "tool", self.copy),
            ("Remove", "clear", self.remove),
        ):
            button = QPushButton(load_icon(icon), text)
            button.clicked.connect(slot)
            row.addWidget(button)
        layout.addLayout(row)
        self.refresh()

    def refresh(self) -> None:
        current = self.tools.currentItem().text() if self.tools.currentItem() else None
        self.tools.blockSignals(True)
        self.tools.clear()
        self.tools.addItems(sorted(self.twin.machine.tools))
        self.tools.blockSignals(False)
        names = sorted(self.twin.machine.tools)
        if names:
            self.tools.setCurrentRow(names.index(current) if current in names else 0)
            self._select(self.tools.currentItem().text())

    def _select(self, name: str) -> None:
        spec = self.twin.machine.tools.get(name)
        if spec is not None:
            self.name_edit.setText(spec.name)
            self.editor.load(spec)
            self.status.clear()

    def current_spec(self) -> ToolSpec:
        return replace(self.editor.spec(), name=self.name_edit.text().strip())

    def apply(self) -> None:
        try:
            spec = self.current_spec()
            if not spec.name or not spec.name.isalnum():
                raise ValueError("the tool name must be letters and digits only")
            self.twin.set_tool_spec(spec)
        except ValueError as exc:
            self.status.setText(f"Not applied: {exc}")
            return
        self.status.setText(f"{spec.name} applied")
        self.refresh()

    def copy(self) -> None:
        item = self.tools.currentItem()
        base = item.text() if item else "Tool"
        names = set(self.twin.machine.tools)
        index = 2
        while f"{base}{index}" in names:
            index += 1
        self.name_edit.setText(f"{base}{index}")
        self.apply()
        self.tools.setCurrentRow(sorted(self.twin.machine.tools).index(f"{base}{index}"))

    def remove(self) -> None:
        item = self.tools.currentItem()
        if item is None:
            return
        if item.text() == self.twin.tool_name():
            self.status.setText("The active tool cannot be removed")
            return
        self.twin.remove_tool(item.text())
        self.refresh()


class CheckPanel(QWidget):
    """The check artefact: place it, measure it, see the evaluation."""

    progress = Signal(str)
    finished = Signal(str)

    def __init__(
        self,
        twin: DigitalTwin,
        port: Callable[[], int | None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.twin = twin
        self._port = port
        #: The check program is a client; it runs on a loop of its own, not the server's.
        self._runner = LoopThread("ippdme-check")
        self._future: Future[None] | None = None
        layout = QVBoxLayout(self)
        place = QHBoxLayout()
        for text, slot in (
            ("Place check artefact", self.add_artifact),
            ("Place reference sphere", self.add_sphere),
        ):
            button = QPushButton(load_icon("tool" if "sphere" in text else "csy"), text)
            button.clicked.connect(slot)
            place.addWidget(button)
        layout.addLayout(place)
        modes = QGroupBox("Run the check program with a client (needs the server running)")
        grid = QGridLayout(modes)
        self.mode_checks: dict[str, QCheckBox] = {}
        for i, mode in enumerate(MODES):
            check = QCheckBox(mode)
            check.setChecked(mode in CheckOptions().modes)
            self.mode_checks[mode] = check
            grid.addWidget(check, i // 4, i % 4)
        self.run_button = QPushButton(load_icon("run"), "Run check program")
        self.run_button.clicked.connect(self.run_program)
        grid.addWidget(self.run_button, 2, 0, 1, 4)
        layout.addWidget(modes)
        row = QHBoxLayout()
        evaluate = QPushButton(load_icon("ok"), "Evaluate")
        evaluate.clicked.connect(self.evaluate)
        clear = QPushButton(load_icon("clear"), "Clear points")
        clear.clicked.connect(twin.clear_measurements)
        row.addWidget(evaluate)
        row.addWidget(clear)
        layout.addLayout(row)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.table = EvaluationTable()
        layout.addWidget(self.table, 1)
        self.progress.connect(lambda text: self.status.setText(text))
        self.finished.connect(self._finished)

    def _place(self, make: Callable[[], Any]) -> None:
        obj = make()
        self.twin.add_object(obj)
        self.twin.rest_on_table(obj)
        self.twin.set_pose(obj, obj.pose)

    def add_artifact(self) -> None:
        self._place(build_check_artifact)

    def add_sphere(self) -> None:
        self._place(build_reference_sphere)

    def evaluate(self) -> dict[str, list[Any]]:
        results = self.twin.evaluate()
        self.table.show_results(results)
        if not results:
            self.status.setText(
                "Nothing to evaluate: place the check artefact and measure it first"
                if self.twin.artifact() is None
                else "No points measured yet"
            )
        else:
            failed = sum(1 for items in results.values() for r in items if r.ok is False)
            self.status.setText(
                f"{failed} result(s) over the limit" if failed else "All within the limits"
            )
        return results

    def options(self) -> CheckOptions:
        return CheckOptions(modes=tuple(m for m, c in self.mode_checks.items() if c.isChecked()))

    def run_program(self) -> None:
        obj = self.twin.artifact()
        port = self._port()
        if obj is None or obj.artifact is None:
            self.status.setText("Place the check artefact first")
            return
        if port is None:
            self.status.setText("Start the server first")
            return
        rotary = self.twin.machine.rotary_pose(self.twin.snapshot().rotary)
        data = obj.artifact.placed(obj.world_pose(rotary))
        options = self.options()
        self.run_button.setEnabled(False)
        self.status.setText("Running …")

        async def go() -> None:
            await run_check_against_server(
                "127.0.0.1", port, data, options, lambda text: self.progress.emit(text)
            )

        self._future = self._runner.submit(go())
        self._future.add_done_callback(
            lambda f: self.finished.emit("" if f.exception() is None else str(f.exception()))
        )

    def _finished(self, error: str) -> None:
        self.run_button.setEnabled(True)
        if error:
            QMessageBox.warning(self, "Check program", error)
            self.status.setText(f"Stopped: {error}")
        else:
            self.status.setText("Check program finished")
            self.evaluate()

    def shutdown(self) -> None:
        """Stop the helper thread (the window calls this when it closes)."""
        self._runner.close()


class SafetyPanel(QWidget):
    """Emergency stop, air supply, and which tools are qualified."""

    def __init__(self, twin: DigitalTwin, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.twin = twin
        layout = QVBoxLayout(self)
        self.estop = QPushButton(load_icon("abort"), "Emergency stop")
        self.estop.setCheckable(True)
        self.estop.toggled.connect(self._estop)
        self.air = QCheckBox("Air pressure ok")
        self.air.setChecked(True)
        self.air.toggled.connect(twin.set_air_ok)
        self.require = QCheckBox("Require qualified tools for measuring")
        self.require.setChecked(twin.require_qualification)
        self.require.toggled.connect(lambda on: setattr(twin, "require_qualification", on))
        layout.addWidget(self.estop)
        layout.addWidget(self.air)
        layout.addWidget(self.require)
        layout.addWidget(QLabel("Qualified (tool, head position):"))
        self.qualified = QListWidget()
        layout.addWidget(self.qualified, 1)
        self.refresh()

    def _estop(self, pressed: bool) -> None:
        self.twin.set_estop(pressed)
        self.estop.setText("Release emergency stop" if pressed else "Emergency stop")

    def sync(self) -> None:
        """Show the state the twin has (it can change by other means, e.g. a script)."""
        for box, value in ((self.estop, self.twin.estop), (self.air, self.twin.air_ok)):
            if box.isChecked() != value:
                box.blockSignals(True)
                box.setChecked(value)
                box.blockSignals(False)

    def refresh(self) -> None:
        self.qualified.clear()
        for name, orientation in sorted(self.twin.qualified, key=lambda q: (q[0], str(q[1]))):
            where = "" if orientation is None else f"  A={orientation[0]:g}° B={orientation[1]:g}°"
            self.qualified.addItem(name + where)
