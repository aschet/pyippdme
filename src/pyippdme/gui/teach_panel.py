# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Teach-in and programs: the jog box, manual points, and offline program generation.

Everything the user does at the machine's jog box reaches the client the way the standard defines
it (5.5.3): ``KeyPress``, a clearance point as ``GoTo(...)`` and a picked point as ``PtMeas(...)``,
all with the tag ``E0000``. Programs are lists of I++ DME command lines.
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from pyippdme.gui.widgets import icon_button, mono, spin
from pyippdme.loop import LoopThread
from pyippdme.twin import DigitalTwin
from pyippdme.twin.programs import run_program, touch_program
from pyippdme.types.vec3 import Vec3

__all__ = ["TeachPanel"]

_DIRECTIONS = {
    "down (-Z)": (0.0, 0.0, -1.0),
    "up (+Z)": (0.0, 0.0, 1.0),
    "+X": (1.0, 0.0, 0.0),
    "-X": (-1.0, 0.0, 0.0),
    "+Y": (0.0, 1.0, 0.0),
    "-Y": (0.0, -1.0, 0.0),
}
_KEYS = ("Done", "Del", "F1", "F2", "F3", "F4")


def _outcome(future: Future[Any], failure: str = "") -> str:
    """Return an error text for a finished future, or ``""`` if it succeeded."""
    if future.cancelled():
        return "cancelled"
    error = future.exception()
    if error is not None:
        return f"{type(error).__name__}: {error}"
    return "" if future.result() is not False else failure


class TeachPanel(QWidget):
    """The jog box, the events a user triggers at the machine, and programs."""

    #: Result of a background job: ``(what, error text or empty)``; shown in :attr:`status`.
    job_finished = Signal(str, str)
    #: A line of a program run finished: ``(text, ok)``.
    line_done = Signal(str, bool)
    #: A generated program is ready.
    generated = Signal(list)

    def __init__(
        self,
        twin: DigitalTwin,
        submit: Callable[[Coroutine[Any, Any, Any]], Future[Any]],
        port: Callable[[], int | None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.twin = twin
        self._submit = submit
        self._port = port
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ippdme-teach")
        #: The program runs as a client on a loop of its own, like a real client program would.
        self._runner = LoopThread("ippdme-program")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        jog = QGroupBox("Jog box")
        jog_layout = QVBoxLayout(jog)
        step_row = QHBoxLayout()
        step_row.addWidget(QLabel("Step"))
        self.jog_step = spin(0.01, 200, 5.0, 1.0, 2, " mm")
        step_row.addWidget(self.jog_step)
        jog_layout.addLayout(step_row)
        grid = QGridLayout()
        self.jog_buttons: dict[str, Any] = {}
        for col, axis in enumerate("XYZ"):
            for row, sign in enumerate((1, -1)):
                label = f"{'+' if sign > 0 else '-'}{axis}"
                button = icon_button("jog", label, f"Move {label} by one step (needs EnableUser)")
                button.clicked.connect(lambda _=False, a=col, s=sign: self.jog(a, s))
                self.jog_buttons[label] = button
                grid.addWidget(button, row, col)
        jog_layout.addLayout(grid)
        layout.addWidget(jog)

        goto = QGroupBox("Go to (collision-free)")
        goto_form = QFormLayout(goto)
        self.target = [spin(-5000, 5000, 0.0, 10.0, 2, " mm") for _ in range(3)]
        target_row = QWidget()
        row_layout = QHBoxLayout(target_row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        for box in self.target:
            row_layout.addWidget(box)
        goto_form.addRow("Target", target_row)
        buttons = QWidget()
        buttons_layout = QHBoxLayout(buttons)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        here = icon_button("goto", "Here", "Take the current position")
        here.clicked.connect(self.take_position)
        self.go_button = icon_button("run", "Go", "Plan a path around obstacles and drive it")
        self.go_button.clicked.connect(self.go_safely)
        buttons_layout.addWidget(here)
        buttons_layout.addWidget(self.go_button)
        goto_form.addRow("", buttons)
        layout.addWidget(goto)

        teach = QGroupBox("Teach-in: events to the client")
        teach_layout = QVBoxLayout(teach)
        keys = QGridLayout()
        self.key_buttons: dict[str, Any] = {}
        for i, key in enumerate(_KEYS):
            button = icon_button("key", key, f"Send KeyPress({key}) to the client")
            button.clicked.connect(lambda _=False, k=key: self.press(k))
            self.key_buttons[key] = button
            keys.addWidget(button, i // 3, i % 3)
        teach_layout.addLayout(keys)
        pick_row = QHBoxLayout()
        self.direction = QComboBox()
        self.direction.addItems(list(_DIRECTIONS))
        self.pick_button = icon_button("pick", "Pick point", "Touch the part and tell the client")
        self.pick_button.clicked.connect(self.pick)
        self.clearance_button = icon_button(
            "clearance", "Clearance", "Tell the client the current position as a clearance point"
        )
        self.clearance_button.clicked.connect(self.clearance)
        pick_row.addWidget(self.direction)
        pick_row.addWidget(self.pick_button)
        pick_row.addWidget(self.clearance_button)
        teach_layout.addLayout(pick_row)
        tool_row = QHBoxLayout()
        self.tool_combo = QComboBox()
        self.tool_button = icon_button("tool", "Change tool", "Change the tool at the machine")
        self.tool_button.clicked.connect(self.change_tool)
        tool_row.addWidget(self.tool_combo, 1)
        tool_row.addWidget(self.tool_button)
        teach_layout.addLayout(tool_row)
        layout.addWidget(teach)

        program = QGroupBox("Program")
        program_layout = QVBoxLayout(program)
        gen_row = QHBoxLayout()
        self.per_feature = QSpinBox()
        self.per_feature.setRange(1, 200)
        self.per_feature.setValue(4)
        self.per_feature.setSuffix(" points per feature")
        self.generate_button = icon_button(
            "generate", "Generate", "Create a collision-free program for the check artefact"
        )
        self.generate_button.clicked.connect(self.generate)
        gen_row.addWidget(self.per_feature)
        gen_row.addWidget(self.generate_button)
        program_layout.addLayout(gen_row)
        self.program = QPlainTextEdit()
        mono(self.program, 9)
        self.program.setPlaceholderText(
            "One I++ command per line. Generate a program, load one, or type your own."
        )
        program_layout.addWidget(self.program, 1)
        file_row = QHBoxLayout()
        load = icon_button("open", "Open…", "Load a file of commands")
        load.clicked.connect(self.load_program)
        save = icon_button("save", "Save…", "Save the program")
        save.clicked.connect(self.save_program)
        self.run_button = icon_button(
            "run", "Run", "Simulate the program with a client of this window (needs the server)"
        )
        self.run_button.clicked.connect(self.run)
        for widget in (load, save, self.run_button):
            file_row.addWidget(widget)
        program_layout.addLayout(file_row)
        layout.addWidget(program, 1)

        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.job_finished.connect(self._job_finished)
        self.line_done.connect(self._line_done)
        self.generated.connect(lambda lines: self.program.setPlainText("\n".join(lines)))
        self.refresh_tools()

    # -- helpers ----------------------------------------------------------------------------

    def refresh_tools(self) -> None:
        current = self.tool_combo.currentText()
        self.tool_combo.clear()
        self.tool_combo.addItems(sorted(self.twin.machine.tools))
        if current:
            self.tool_combo.setCurrentText(current)

    def _server(self) -> Any | None:
        return self.twin.server

    def _send(self, make: Callable[[Any], Coroutine[Any, Any, bool]], what: str) -> None:
        server = self._server()
        if server is None or self._port() is None:
            self.status.setText("Start the server first.")
            return
        future = self._submit(make(server))
        future.add_done_callback(
            lambda f: self.job_finished.emit(
                what, _outcome(f, "no client, or the user is not enabled")
            )
        )

    def _job_finished(self, what: str, error: str) -> None:
        self.status.setText(f"{what}: {error}" if error else f"{what}: done")
        self.go_button.setEnabled(True)
        self.generate_button.setEnabled(True)
        self.run_button.setEnabled(True)

    # -- jog --------------------------------------------------------------------------------

    def jog(self, axis: int, sign: int) -> bool:
        delta = [0.0, 0.0, 0.0]
        delta[axis] = sign * self.jog_step.value()
        ok = self.twin.jog(*delta)
        if not ok:
            self.status.setText(
                "The jog box needs a client with EnableUser, a homed machine, and a target "
                "inside the machine volume."
            )
        return ok

    def take_position(self) -> None:
        x, y, z = self.twin.snapshot().position
        for box, value in zip(self.target, (x, y, z), strict=True):
            box.setValue(value)

    def go_safely(self) -> None:
        target: Vec3 = (self.target[0].value(), self.target[1].value(), self.target[2].value())
        self.go_button.setEnabled(False)
        self.status.setText("Planning the path …")

        async def go() -> bool:
            await self.twin.go_safely(target)
            return True

        future = self._submit(go())
        future.add_done_callback(lambda f: self.job_finished.emit("Go to", _outcome(f)))

    # -- events to the client ---------------------------------------------------------------

    def press(self, key: str) -> None:
        self._send(lambda server: server.key_press(key), f"KeyPress({key})")

    def pick(self) -> None:
        picked = self.twin.pick_point(_DIRECTIONS[self.direction.currentText()])
        if picked is None:
            self.status.setText("Nothing to touch within reach in that direction.")
            return
        point, normal = picked
        x, y, z = self.twin.to_client(point)
        i, j, k = self.twin.to_client_direction(normal)
        self._send(lambda server: server.manual_point(x, y, z, (i, j, k)), "PtMeas event")

    def clearance(self) -> None:
        x, y, z = self.twin.to_client(self.twin.snapshot().position)
        self._send(lambda server: server.clearance_point(x, y, z), "GoTo event")

    def change_tool(self) -> None:
        name = self.tool_combo.currentText()
        if name:
            self._send(lambda server: server.change_tool(name), f"ChangeTool({name})")

    # -- programs ---------------------------------------------------------------------------

    def generate(self) -> None:
        obj = self.twin.artifact()
        if obj is None or obj.artifact is None:
            self.status.setText("Place the check artefact first (Check artefact tab).")
            return
        rotary = self.twin.machine.rotary_pose(self.twin.snapshot().rotary)
        data = obj.artifact.placed(obj.world_pose(rotary))
        limit = self.per_feature.value()
        counts: dict[str, int] = {}
        points: list[tuple[Vec3, Vec3]] = []
        for p in data.touch:
            if counts.get(p.feature, 0) < limit:
                counts[p.feature] = counts.get(p.feature, 0) + 1
                points.append((p.point, p.normal))
        self.generate_button.setEnabled(False)
        self.status.setText(f"Planning {len(points)} points …")

        def work() -> None:
            try:
                lines = touch_program(self.twin, points)
            except ValueError as error:
                self.job_finished.emit("Generate", str(error))
                return
            self.generated.emit(lines)
            self.job_finished.emit("Generate", "")

        self._pool.submit(work)

    def load_program(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open a program", "", "Text (*.txt *.prg);;All (*)"
        )
        if path:
            self.program.setPlainText(Path(path).read_text())

    def save_program(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save the program", "program.txt", "Text (*.txt)"
        )
        if path:
            Path(path).write_text(self.program.toPlainText() + "\n")

    def commands(self) -> list[str]:
        """Return the lines of the program without blanks and ``#`` comments."""
        return [
            line.strip()
            for line in self.program.toPlainText().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]

    def run(self) -> None:
        port = self._port()
        lines = self.commands()
        if port is None:
            self.status.setText("Start the server first.")
            return
        if not lines:
            self.status.setText("The program is empty.")
            return
        self.run_button.setEnabled(False)
        self.status.setText(f"Running {len(lines)} commands …")

        async def go() -> bool:
            return await run_program(
                "127.0.0.1", port, lines, on_line=lambda t, ok: self.line_done.emit(t, ok)
            )

        future = self._runner.submit(go())
        future.add_done_callback(
            lambda f: self.job_finished.emit("Program", _outcome(f, "stopped at an error"))
        )

    def _line_done(self, text: str, ok: bool) -> None:
        self.status.setText(("" if ok else "Failed: ") + text)
        if not ok:
            self.run_button.setEnabled(True)

    def shutdown(self) -> None:
        """Stop the helper threads (the window calls this when it closes)."""
        self._pool.shutdown(wait=False, cancel_futures=True)
        self._runner.close()
