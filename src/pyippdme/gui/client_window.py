# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""A Qt command client: connect to any I++ DME server, run commands from dialogs or by hand.

All protocol work happens in :class:`~pyippdme.client.host.ClientHost` on its own thread; this
window only turns clicks into command lines (:mod:`pyippdme.client.recipes`,
:mod:`pyippdme.client.commandform`) and shows what comes back: the log, the measured points, the
last response, and a status bar with the machine position, the homing state and errors.
"""

from __future__ import annotations

import csv
import html
import io
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QGuiApplication, QKeyEvent
from PySide6.QtWidgets import (
    QApplication,
    QCompleter,
    QDockWidget,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from pyippdme.cli._interaction import (
    Acked,
    Completed,
    ConnectionLost,
    Failed,
    ParseFailed,
    Received,
    format_error,
)
from pyippdme.client import commandform, recipes
from pyippdme.client.host import ClientHost
from pyippdme.client.report import report_from_payload
from pyippdme.gui import client_dialogs as dialogs
from pyippdme.gui.icons import app_icon, load_icon
from pyippdme.gui.pointcloud import PointCloudView
from pyippdme.protocol.ast import (
    BasicName,
    DataPayload,
    Items,
    NameValue,
    NumericData,
    String,
    StringValue,
)
from pyippdme.protocol.transport import DEFAULT_PORT
from pyippdme.types.vec3 import Vec3

_POLL_MS = 400
_LOG_LIMIT = 5000
_STATUS_TEXTS = frozenset(recipes.status_lines())


class _Bridge(QObject):
    """Hands what the client thread reports over to the GUI thread."""

    data = Signal(str, object)
    finished = Signal(object, str)  # a Future's exception (or None), what it was for
    result = Signal(object, str)  # a Future's result, what it was for


class _HistoryLine(QLineEdit):
    """A line edit that walks through earlier entries with Up and Down."""

    def __init__(self) -> None:
        super().__init__()
        self.history: list[str] = []
        self._index = 0

    def remember(self, text: str) -> None:
        if text and (not self.history or self.history[-1] != text):
            self.history.append(text)
        self._index = len(self.history)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        key = event.key()
        if key == Qt.Key.Key_Up and self.history:
            self._index = max(0, self._index - 1)
            self.setText(self.history[self._index])
        elif key == Qt.Key.Key_Down and self.history:
            self._index = min(len(self.history), self._index + 1)
            self.setText(self.history[self._index] if self._index < len(self.history) else "")
        else:
            super().keyPressEvent(event)


def _table(headers: list[str]) -> QTableWidget:
    table = QTableWidget(0, len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.horizontalHeader().setStretchLastSection(True)
    table.verticalHeader().setVisible(False)
    table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
    return table


def _names(payloads: list[DataPayload]) -> list[str]:
    names: list[str] = []
    for payload in payloads:
        if isinstance(payload, StringValue | NameValue):
            names.append(str(payload.value))
        elif isinstance(payload, Items):
            for item in payload.values:
                names.extend(str(a.value) for a in item.args if isinstance(a, String | BasicName))
    return names


class ClientWindow(QMainWindow):
    """The command client."""

    def __init__(self, host: ClientHost | None = None) -> None:
        super().__init__()
        self.host = host or ClientHost()
        self.position: Vec3 | None = None
        self._bridge = _Bridge()
        self._bridge.data.connect(self._on_event)
        self._bridge.finished.connect(self._on_finished)
        self._bridge.result.connect(self._on_result)
        self._polling = False
        self._echoed = ""
        self._queries: dict[str, Callable[[list[DataPayload]], None]] = {}
        self._collected: dict[str, list[DataPayload]] = {}
        self._dialogs: dict[str, dialogs.TaskDialog] = {}
        self.setWindowTitle("pyippdme command client")
        self.setWindowIcon(app_icon())
        self.resize(1280, 800)
        self._build_toolbars()
        self._build_central()
        self._build_docks()
        self._build_status_bar()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll)
        self._timer.start(_POLL_MS)
        self._update_connection_state()

    # -- construction ---------------------------------------------------------------------

    def _action(self, text: str, icon: str, slot: Callable[[], Any], tip: str = "") -> QAction:
        action = QAction(load_icon(icon), text, self)
        action.setToolTip(tip or text)
        action.triggered.connect(lambda _=False: slot())
        return action

    def _build_toolbars(self) -> None:
        bar = QToolBar("Connection")
        bar.setMovable(False)
        self.addToolBar(bar)
        self.host_edit = QLineEdit("127.0.0.1")
        self.host_edit.setFixedWidth(150)
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(DEFAULT_PORT)
        bar.addWidget(QLabel(" Server "))
        bar.addWidget(self.host_edit)
        bar.addWidget(QLabel(" : "))
        bar.addWidget(self.port_spin)
        self.connect_action = self._action("Connect", "connect", self.toggle_connection)
        self.virtual_action = self._action(
            "Virtual CMM", "virtual", self.connect_virtual, "Start a virtual CMM in this program"
        )
        bar.addAction(self.connect_action)
        bar.addAction(self.virtual_action)

        machine = QToolBar("Machine")
        machine.setMovable(False)
        self.addToolBar(machine)
        self.enable_action = self._action(
            "Enable",
            "enable",
            lambda: self.send("EnableUser()"),
            "EnableUser: allow the program to move the machine",
        )
        self.home_action = self._action(
            "Home", "home", lambda: self.send("Home()"), "Home the machine"
        )
        self.abort_action = self._action(
            "Abort", "abort", lambda: self.send("AbortE()"), "AbortE: stop the current motion"
        )
        for action in (self.enable_action, self.home_action, self.abort_action):
            machine.addAction(action)

        tasks = QToolBar("Measure")
        tasks.setMovable(False)
        tasks.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        self.addToolBar(tasks)
        self.task_actions = [
            self._action("Move", "goto", lambda: self.open_dialog("move"), "Move to a position"),
            self._action("Point", "point", lambda: self.open_dialog("point"), "Measure a point"),
            self._action("Line scan", "line", lambda: self.open_dialog("line"), "Scan a line"),
            self._action(
                "Circle / helix",
                "circle",
                lambda: self.open_dialog("arc"),
                "Scan a circle or helix",
            ),
            self._action(
                "Tools", "tool", lambda: self.open_dialog("tool"), "Change or select tool"
            ),
            self._action(
                "Speeds", "speed", lambda: self.open_dialog("speed"), "Speeds and accelerations"
            ),
            self._action(
                "Optical", "optical", lambda: self.open_dialog("optical"), "Acquire with the sensor"
            ),
        ]
        for action in self.task_actions:
            tasks.addAction(action)

        files = QToolBar("Files")
        files.setMovable(False)
        self.addToolBar(files)
        files.addAction(
            self._action("Run script", "script", self.run_script, "Run a file of commands")
        )
        files.addAction(
            self._action(
                "Save commands", "save", self.save_history, "Save the commands sent so far"
            )
        )
        files.addAction(self._action("Clear log", "clear", self.clear_log))

    def _build_central(self) -> None:
        central = QWidget()
        layout = QVBoxLayout(central)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFontFamily("monospace")
        layout.addWidget(self.log, 1)
        row = QHBoxLayout()
        self.command_line = _HistoryLine()
        self.command_line.setPlaceholderText("Command, e.g. GoTo(X(10),Y(20))   (Up/Down: history)")
        completer = QCompleter(commandform.command_names())
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.command_line.setCompleter(completer)
        self.command_line.returnPressed.connect(self._send_typed)
        send = QPushButton(load_icon("run"), "Send")
        send.clicked.connect(self._send_typed)
        row.addWidget(self.command_line, 1)
        row.addWidget(send)
        layout.addLayout(row)
        self.setCentralWidget(central)

    def _dock(self, title: str, widget: QWidget, area: Qt.DockWidgetArea) -> QDockWidget:
        dock = QDockWidget(title, self)
        dock.setWidget(widget)
        self.addDockWidget(area, dock)
        return dock

    def _build_docks(self) -> None:
        # Every command of the protocol, with a form for its arguments.
        panel = QWidget()
        box = QVBoxLayout(panel)
        self.command_filter = QLineEdit()
        self.command_filter.setPlaceholderText("Search commands")
        self.command_filter.setClearButtonEnabled(True)
        self.command_list = QListWidget()
        self.command_list.addItems(commandform.command_names())
        self.command_filter.textChanged.connect(self._filter_commands)
        self.command_list.currentTextChanged.connect(self._show_command_form)
        self.form_host = QWidget()
        self.form_layout = QFormLayout(self.form_host)
        self.form_edits: dict[str, QLineEdit] = {}
        self.form_status = QLabel()
        self.form_status.setWordWrap(True)
        build = QPushButton(load_icon("script"), "Copy to command line")
        build.clicked.connect(lambda: self._build_from_form(send=False))
        run = QPushButton(load_icon("run"), "Send")
        run.clicked.connect(lambda: self._build_from_form(send=True))
        buttons = QHBoxLayout()
        buttons.addWidget(build)
        buttons.addWidget(run)
        box.addWidget(self.command_filter)
        box.addWidget(self.command_list, 1)
        box.addWidget(self.form_host)
        box.addWidget(self.form_status)
        box.addLayout(buttons)
        self._dock("All commands", panel, Qt.DockWidgetArea.LeftDockWidgetArea)

        points = QWidget()
        pbox = QVBoxLayout(points)
        self.points = _table(["#", "X", "Y", "Z", "From"])
        pbox.addWidget(self.points)
        prow = QHBoxLayout()
        copy = QPushButton(load_icon("save"), "Copy as CSV")
        copy.clicked.connect(self.copy_points)
        wipe = QPushButton(load_icon("clear"), "Clear")
        wipe.clicked.connect(lambda: self.points.setRowCount(0))
        prow.addWidget(copy)
        prow.addWidget(wipe)
        pbox.addLayout(prow)
        self._dock("Measured points", points, Qt.DockWidgetArea.RightDockWidgetArea)

        cloud = QWidget()
        cbox = QVBoxLayout(cloud)
        self.cloud = PointCloudView()
        cbox.addWidget(self.cloud, 1)
        crow = QHBoxLayout()
        for text, icon, slot in (
            ("Fit", "goto", self.cloud.fit),
            ("Copy as CSV", "save", self.copy_cloud),
            ("Save .xyz", "save", self.save_cloud),
            ("Clear", "clear", self.cloud.clear),
        ):
            button = QPushButton(load_icon(icon), text)
            button.clicked.connect(lambda _=False, f=slot: f())
            crow.addWidget(button)
        cbox.addLayout(crow)
        self.cloud_dock = self._dock("Point cloud", cloud, Qt.DockWidgetArea.RightDockWidgetArea)

        self.response = _table(["Name", "Value"])
        self._dock("Last response", self.response, Qt.DockWidgetArea.RightDockWidgetArea)

    def _build_status_bar(self) -> None:
        bar = self.statusBar()
        self.connection_label = QLabel()
        self.position_label = QLabel("Position: -")
        self.position_label.setFont(self.log.font())
        self.homed_label = QLabel()
        self.error_label = QLabel()
        self.error_label.setStyleSheet("color: #e74c3c;")
        self.error_label.setTextFormat(Qt.TextFormat.PlainText)
        self.clear_errors = QPushButton(load_icon("clear"), "Clear errors")
        self.clear_errors.setFlat(True)
        self.clear_errors.clicked.connect(lambda: self.send("ClearAllErrors()"))
        for w in (self.connection_label, self.position_label, self.homed_label):
            bar.addPermanentWidget(w)
        bar.addWidget(self.error_label, 1)
        bar.addPermanentWidget(self.clear_errors)

    # -- connection -----------------------------------------------------------------------

    def _watch(self, future: Any, what: str) -> None:
        future.add_done_callback(lambda f: self._bridge.finished.emit(f.exception(), what))

    def toggle_connection(self) -> None:
        if self.host.connected:
            self._watch(self.host.disconnect(), "disconnect")
            return
        self._append(f"Connecting to {self.host_edit.text()}:{self.port_spin.value()} …", "info")
        self._watch(self.host.connect(self.host_edit.text(), self.port_spin.value()), "connect")

    def connect_virtual(self) -> None:
        if self.host.connected:
            self._watch(self.host.disconnect(), "disconnect")
        self._append("Starting a virtual CMM …", "info")
        self._watch(self.host.connect_embedded(), "connect")

    def _on_finished(self, error: object, what: str) -> None:
        if error is not None:
            self._show_error(f"{what}: {error}")
            self._append(f"{what} failed: {error}", "error")
        elif what == "connect":
            self._append("Connected", "ok")
            self.error_label.clear()
            self.send("StartSession()")
        elif what == "disconnect":
            self._append("Disconnected", "info")
            self.position = None
        self._update_connection_state()

    def _update_connection_state(self) -> None:
        connected = self.host.connected
        self.connect_action.setText("Disconnect" if connected else "Connect")
        self.connect_action.setIcon(load_icon("disconnect" if connected else "connect"))
        for action in (self.enable_action, self.home_action, self.abort_action, *self.task_actions):
            action.setEnabled(connected)
        self.host_edit.setEnabled(not connected)
        self.port_spin.setEnabled(not connected)
        self.connection_label.setText(
            ("  Connected" + (" (virtual CMM)" if self.host.embedded else ""))
            if connected
            else "  Not connected"
        )
        if not connected:
            self.position_label.setText("Position: -")
            self.homed_label.clear()

    # -- sending ----------------------------------------------------------------------------

    def send(self, text: str) -> None:
        self.send_lines([text])

    def send_lines(self, lines: list[str]) -> None:
        if not self.host.connected:
            self._append("Not connected", "error")
            return
        for line in lines:
            self.command_line.remember(line)
        self._watch(self.host.run_sequence(lines, self._relay), "command")

    def _relay(self, text: str, event: object) -> None:
        self._bridge.data.emit(text, event)

    def _send_typed(self) -> None:
        text = self.command_line.text().strip()
        if text:
            self.send(text)
            self.command_line.clear()

    def query(self, text: str, callback: Callable[[list[DataPayload]], None]) -> None:
        """Run ``text`` and hand its data responses to ``callback`` (on the GUI thread)."""
        self._queries[text] = callback
        self.send(text)

    # -- showing results --------------------------------------------------------------------

    def _append(self, text: str, kind: str = "") -> None:
        colors = {"error": "#e74c3c", "ok": "#27ae60", "info": "#7f8c8d", "cmd": "#2f6fb0"}
        color = colors.get(kind)
        escaped = html.escape(text)
        self.log.append(f'<span style="color:{color}">{escaped}</span>' if color else escaped)
        document = self.log.document()
        if document is not None and document.blockCount() > _LOG_LIMIT:
            self.log.clear()

    def clear_log(self) -> None:
        self.log.clear()

    def _show_error(self, text: str) -> None:
        self.error_label.setText("⚠ " + text)
        self.error_label.setToolTip(text)

    def _on_event(self, text: str, event: object) -> None:
        status = text in _STATUS_TEXTS
        if isinstance(event, Received):
            self._on_data(text, event.payload, status)
        if status and not isinstance(event, Failed | ConnectionLost):
            return
        if text != self._echoed:
            self._echoed = text
            self._append(f"> {text}", "cmd")
        match event:
            case ParseFailed(error):
                self._append(f"Parse error: {error}", "error")
                self._show_error(f"Parse error: {error}")
            case Acked():
                pass
            case Received(payload):
                self._append(f"# {payload.to_wire()}")
            case Completed():
                self._append("%", "ok")
                self.error_label.clear()
                self._finish_query(text)
                self._refresh_status()
            case Failed(error):
                line = format_error(error)
                self._append(line, "error")
                self._show_error(f"Error {error.number}: {error.text or error.cause}")
                self._queries.pop(text, None)
                self._collected.pop(text, None)
            case ConnectionLost(error):
                self._append(f"Connection lost: {error}", "error")
                self._show_error(f"Connection lost: {error}")
                self._update_connection_state()

    def _finish_query(self, text: str) -> None:
        callback = self._queries.pop(text, None)
        data = self._collected.pop(text, [])
        if callback is not None:
            callback(data)

    def _on_data(self, text: str, payload: DataPayload, status: bool) -> None:
        if text in self._queries:
            self._collected.setdefault(text, []).append(payload)
        if isinstance(payload, Items):
            try:
                report = report_from_payload(payload)
            except TypeError:
                return
            self._apply_report(text, dict(report), status)
        elif isinstance(payload, NumericData) and text.startswith(("ScanOn", "ScanIn")):
            values = [v.value for v in payload.values]
            for i in range(0, len(values) - 2, 3):
                self._add_point((values[i], values[i + 1], values[i + 2]), text)
                self.cloud.add_points([(values[i], values[i + 1], values[i + 2])])
                self.position = (values[i], values[i + 1], values[i + 2])
            self._show_position()

    def _apply_report(self, text: str, values: dict[str, Any], status: bool) -> None:
        if all(isinstance(values.get(a), float) for a in "XYZ"):
            self.position = (values["X"], values["Y"], values["Z"])
            self._show_position()
            if text.startswith("PtMeas"):
                self._add_point(self.position, text.split("(")[0])
        if "IsHomed" in values:
            homed = bool(values["IsHomed"])
            self.homed_label.setText("  Homed" if homed else "  Not homed")
        if "ActiveError" in values:
            self._show_error(
                f"Active error {values['ActiveError']:g} (severity {values.get('Severity', '?')})"
            )
        elif status and "IsHomed" in values:
            self.error_label.clear()
        if not status:
            self.response.setRowCount(0)
            for name, value in commandform.result_rows(values):
                row = self.response.rowCount()
                self.response.insertRow(row)
                self.response.setItem(row, 0, QTableWidgetItem(name))
                self.response.setItem(row, 1, QTableWidgetItem(value))

    def _show_position(self) -> None:
        if self.position is not None:
            x, y, z = self.position
            self.position_label.setText(f"X {x:9.3f}   Y {y:9.3f}   Z {z:9.3f}")

    def _add_point(self, point: Vec3, source: str) -> None:
        row = self.points.rowCount()
        self.points.insertRow(row)
        for column, text in enumerate([str(row + 1), *(f"{v:.4f}" for v in point), source]):
            self.points.setItem(row, column, QTableWidgetItem(text))
        self.points.scrollToBottom()

    def copy_points(self) -> None:
        out = io.StringIO()
        writer = csv.writer(out)
        writer.writerow(["n", "x", "y", "z", "from"])
        for r in range(self.points.rowCount()):
            writer.writerow([self.points.item(r, c).text() for c in range(5)])  # type: ignore[union-attr]
        QGuiApplication.clipboard().setText(out.getvalue())

    # -- status polling ---------------------------------------------------------------------

    def _poll(self) -> None:
        if self.host.connected and not self._polling:
            self._refresh_status()

    def _refresh_status(self) -> None:
        if self._polling or not self.host.connected:
            return
        self._polling = True
        future = self.host.run_sequence(recipes.status_lines(), self._relay)
        future.add_done_callback(lambda _: setattr(self, "_polling", False))

    # -- dialogs ----------------------------------------------------------------------------

    def _current_position(self) -> Vec3 | None:
        return self.position

    def open_dialog(self, name: str) -> dialogs.TaskDialog:
        dialog = self._dialogs.get(name)
        if dialog is None:
            if name == "move":
                dialog = dialogs.MoveDialog(self._current_position, self)
            elif name == "point":
                dialog = dialogs.PointDialog(self._current_position, self)
            elif name == "line":
                dialog = dialogs.ScanLineDialog(self._current_position, self)
            elif name == "arc":
                dialog = dialogs.ScanArcDialog(self._current_position, self)
            elif name == "speed":
                dialog = dialogs.SpeedDialog(self)
            elif name == "optical":
                dialog = self._optical_dialog()
            else:
                tools = dialogs.ToolDialog(self)
                tools.refresh_requested.connect(
                    lambda: self.query("EnumTools()", lambda d: tools.set_tools(_names(d)))
                )
                dialog = tools
            dialog.run_requested.connect(self.send_lines)
            self._dialogs[name] = dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        if isinstance(dialog, dialogs.ToolDialog) and not dialog.tools.count():
            dialog.refresh_requested.emit()
        return dialog

    def _optical_dialog(self) -> dialogs.OpticalDialog:
        dialog = dialogs.OpticalDialog(self._current_position, self)
        dialog.info_requested.connect(self.read_sensor_info)
        dialog.acquire_requested.connect(self.acquire)
        return dialog

    def read_sensor_info(self) -> None:
        if self.host.connected:
            self._watch_result(self.host.sensor_info(), "sensor")

    def acquire(self, options: dict[str, Any]) -> None:
        if not self.host.connected:
            self._append("Not connected", "error")
            return
        self._append(
            f"> DataAcquire {options['name']} ({options['acquisition_type']}, "
            f"{len(options['points'])} positions)",
            "cmd",
        )
        self._watch_result(self.host.acquire(**options), "acquire")

    def _watch_result(self, future: Any, what: str) -> None:
        def done(f: Any) -> None:
            if f.exception() is not None:
                self._bridge.finished.emit(f.exception(), what)
            else:
                self._bridge.result.emit(f.result(), what)

        future.add_done_callback(done)

    def _on_result(self, result: object, what: str) -> None:
        if what == "sensor":
            info = result
            text = (
                f"{info.tool_name} ({info.tool_kind}); raw data: "  # type: ignore[attr-defined]
                f"{', '.join(info.technologies) or 'none'}, format {info.data_format or '-'}"  # type: ignore[attr-defined]
            )
            self._append(f"Sensor: {text}", "ok")
            dialog = self._dialogs.get("optical")
            if isinstance(dialog, dialogs.OpticalDialog):
                dialog.set_info("Sensor: " + text)
        elif what == "acquire":
            points = result.points()  # type: ignore[attr-defined]
            self.cloud.set_points(points)
            self.cloud.fit()
            self.cloud_dock.raise_()
            self._append(f"Acquired {len(points)} points", "ok")
            self.error_label.clear()
            self._refresh_status()

    def _cloud_csv(self) -> str:
        out = io.StringIO()
        writer = csv.writer(out)
        writer.writerow(["x", "y", "z"])
        writer.writerows(self.cloud.points.tolist())
        return out.getvalue()

    def copy_cloud(self) -> None:
        QGuiApplication.clipboard().setText(self._cloud_csv())

    def save_cloud(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save the points", "points.xyz", "XYZ (*.xyz)")
        if path:
            Path(path).write_text(
                "".join(f"{x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in self.cloud.points.tolist())
            )

    # -- all commands panel -----------------------------------------------------------------

    def _filter_commands(self, text: str) -> None:
        needle = text.casefold()
        for i in range(self.command_list.count()):
            item = self.command_list.item(i)
            if item is not None:
                item.setHidden(needle not in item.text().casefold())

    def _show_command_form(self, name: str) -> None:
        while self.form_layout.rowCount():
            self.form_layout.removeRow(0)
        self.form_edits.clear()
        fields = commandform.command_fields(name)
        for field in fields:
            edit = QLineEdit()
            edit.setPlaceholderText(field.hint)
            label = field.name + ("" if field.mandatory else " (optional)")
            self.form_layout.addRow(label, edit)
            self.form_edits[field.name] = edit
        self.form_status.setText("" if fields else "No arguments.")

    def _build_from_form(self, *, send: bool) -> None:
        item = self.command_list.currentItem()
        if item is None:
            return
        name = item.text()
        try:
            line = commandform.build_command_line(
                name, {k: e.text() for k, e in self.form_edits.items()}
            )
        except ValueError as exc:
            self.form_status.setText(str(exc))
            return
        self.form_status.setText("")
        if send:
            self.send(line)
        else:
            self.command_line.setText(line)
            self.command_line.setFocus()

    # -- scripts ----------------------------------------------------------------------------

    def run_script(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Run a file of commands", "", "Text (*.txt *.prg);;All (*)"
        )
        if path:
            self.run_script_file(Path(path))

    def run_script_file(self, path: Path) -> None:
        lines = [
            line.strip()
            for line in path.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        self._append(f"Running {path.name}: {len(lines)} commands", "info")
        self.send_lines(lines)

    def save_history(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save the commands", "commands.txt", "Text (*.txt)"
        )
        if path:
            Path(path).write_text("\n".join(self.command_line.history) + "\n")

    def closeEvent(self, event: object) -> None:  # noqa: N802
        self._timer.stop()
        self.host.shutdown()
        super().closeEvent(event)  # type: ignore[arg-type]


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="ippdme client-gui", description="Qt command client for an I++ DME server"
    )
    parser.add_argument("host", nargs="?", help="server address (default: type it in the window)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--virtual", action="store_true", help="start a virtual CMM in this process"
    )
    from pyippdme.cli import virtual

    virtual.add_arguments(parser)
    args = parser.parse_args(argv)
    virtual.configure(virtual.options_from_args(args))
    app = QApplication(sys.argv[:1])
    window = ClientWindow()
    if args.host:
        window.host_edit.setText(args.host)
    window.port_spin.setValue(args.port)
    window.show()
    if args.virtual:
        window.connect_virtual()
    elif args.host:
        window.toggle_connection()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
