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
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QGuiApplication, QKeyEvent, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QCompleter,
    QDockWidget,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QToolBar,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from pyippdme.client import commandform, recipes
from pyippdme.client.host import ClientHost
from pyippdme.client.interaction import (
    Acked,
    Completed,
    ConnectionLost,
    Failed,
    ParseFailed,
    Received,
    format_error,
)
from pyippdme.client.report import report_from_payload
from pyippdme.gui import client_dialogs as dialogs
from pyippdme.gui.icons import app_icon, load_icon
from pyippdme.gui.pointcloud import PointCloudView
from pyippdme.gui.widgets import Led, icon_button, mono, separator
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
_STATUS_BATCH = 0  # the id of the background status poll; user batches count up from 1


class _Bridge(QObject):
    """Hands what the client thread reports over to the GUI thread."""

    data = Signal(int, int, str, object)  # batch, line index, command text, event
    finished = Signal(object, str)  # a Future's exception (or None), what it was for
    result = Signal(object, str)  # a Future's result, what it was for
    batch_done = Signal(int, object, object)  # batch, exception (or None), all lines succeeded


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


class _Batch:
    """The command lines of one click, with the bookkeeping to log them in order."""

    def __init__(self, lines: list[str], first_number: int, origin: dialogs.TaskDialog | None):
        self.lines = lines
        self.first_number = first_number
        self.origin = origin
        self.started: dict[int, float] = {}
        self.begin = time.monotonic()
        self.error = ""


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
        self._bridge.batch_done.connect(self._on_batch_done)
        self._polling = False
        self._next_batch = 1
        self._next_number = 1
        self._batches: dict[int, _Batch] = {}
        self._queries: dict[tuple[int, int], Callable[[list[DataPayload]], None]] = {}
        self._collected: dict[tuple[int, int], list[DataPayload]] = {}
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
        self._append(
            "Not connected. Type the address of an I++ DME server and press Connect (F5), "
            "or press Virtual CMM (Ctrl+Shift+V) to try the client without a machine.",
            "info",
        )

    # -- construction ---------------------------------------------------------------------

    def _action(
        self, text: str, icon: str, slot: Callable[[], Any], tip: str = "", shortcut: str = ""
    ) -> QAction:
        action = QAction(load_icon(icon), text, self)
        action.setToolTip(f"{tip or text} ({shortcut})" if shortcut else tip or text)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        action.triggered.connect(lambda _=False: slot())
        self.addAction(action)  # shortcuts work wherever the focus is
        return action

    def _build_toolbars(self) -> None:
        bar = QToolBar("Connection")
        bar.setMovable(False)
        self.addToolBar(bar)
        self.host_edit = QLineEdit("127.0.0.1")
        self.host_edit.setFixedWidth(150)
        self.host_edit.returnPressed.connect(self.toggle_connection)
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(DEFAULT_PORT)
        bar.addWidget(QLabel(" Server "))
        bar.addWidget(self.host_edit)
        bar.addWidget(QLabel(" : "))
        bar.addWidget(self.port_spin)
        self.connect_action = self._action(
            "Connect", "connect", self.toggle_connection, "Connect to the server", "F5"
        )
        self.virtual_action = self._action(
            "Virtual CMM",
            "virtual",
            self.connect_virtual,
            "Start a virtual CMM in this program",
            "Ctrl+Shift+V",
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
            "Abort", "abort", lambda: self.send("AbortE()"), "AbortE: stop the motion", "Esc"
        )
        for action in (self.enable_action, self.home_action):
            machine.addAction(action)

        tasks = QToolBar("Measure")
        tasks.setMovable(False)
        tasks.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        self.addToolBar(tasks)
        self.task_actions = [
            self._action(
                "Move", "goto", lambda: self.open_dialog("move"), "Move to a position", "Ctrl+1"
            ),
            self._action(
                "Point", "point", lambda: self.open_dialog("point"), "Measure a point", "Ctrl+2"
            ),
            self._action(
                "Line scan", "line", lambda: self.open_dialog("line"), "Scan a line", "Ctrl+3"
            ),
            self._action(
                "Circle / helix",
                "circle",
                lambda: self.open_dialog("arc"),
                "Scan a circle or helix",
                "Ctrl+4",
            ),
            self._action(
                "Tools", "tool", lambda: self.open_dialog("tool"), "Change or select tool", "Ctrl+5"
            ),
            self._action(
                "Speeds",
                "speed",
                lambda: self.open_dialog("speed"),
                "Speeds and accelerations",
                "Ctrl+6",
            ),
            self._action(
                "Optical",
                "optical",
                lambda: self.open_dialog("optical"),
                "Acquire with the sensor",
                "Ctrl+7",
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
        files.addAction(self._action("Clear log", "clear", self.clear_log, "Clear the log"))
        self.focus_action = self._action(
            "Command line", "run", lambda: self.command_line.setFocus(), "Type a command", "Ctrl+L"
        )

    def _build_central(self) -> None:
        central = QWidget()
        layout = QVBoxLayout(central)
        self.strip = QWidget()
        self._build_strip(self.strip)
        layout.addWidget(self.strip)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFontFamily("monospace")
        layout.addWidget(self.log, 1)
        row = QHBoxLayout()
        self.command_line = _HistoryLine()
        self.command_line.setPlaceholderText(
            "Command, e.g. GoTo(X(10),Y(20))   (Up/Down: history, Ctrl+L: focus)"
        )
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

    def _build_strip(self, strip: QWidget) -> None:
        """Lamps for the connection and the machine, the tool and a large position read-out."""
        layout = QHBoxLayout(strip)
        layout.setContentsMargins(4, 2, 4, 2)
        self.connection_label = Led("Not connected", "The link to the server")
        self.session_led = Led("No session", "StartSession has to be accepted by the server")
        self.homed_label = Led("", "Whether the machine is homed")
        self.user_led = Led("User disabled", "EnableUser lets the program move the machine")
        self.busy_led = Led("Ready", "Commands that are still running")
        self.tool_label = QLabel("")
        self.position_label = QLabel("X -   Y -   Z -")
        mono(self.position_label, 15)
        for widget in (
            self.connection_label,
            self.session_led,
            separator(),
            self.homed_label,
            self.user_led,
            separator(),
            self.busy_led,
        ):
            layout.addWidget(widget)
        layout.addWidget(separator())
        layout.addWidget(self.tool_label)
        layout.addStretch(1)
        layout.addWidget(self.position_label)
        self.abort_button = icon_button("abort", "Abort", "AbortE: stop the motion (Esc)")
        self.abort_button.setStyleSheet(
            "QPushButton { background: #c0392b; color: white; font-weight: bold;"
            " padding: 4px 10px; }"
            "QPushButton:disabled { background: #7f8c8d; }"
        )
        self.abort_button.clicked.connect(self.abort_action.trigger)
        layout.addWidget(self.abort_button)

    def _dock(self, title: str, widget: QWidget, area: Qt.DockWidgetArea) -> QDockWidget:
        dock = QDockWidget(title, self)
        dock.setWidget(widget)
        self.addDockWidget(area, dock)
        return dock

    def _build_docks(self) -> None:
        # Every command of the protocol, grouped by task, with a form for its arguments.
        panel = QWidget()
        box = QVBoxLayout(panel)
        self.command_filter = QLineEdit()
        self.command_filter.setPlaceholderText("Search commands")
        self.command_filter.setClearButtonEnabled(True)
        self.command_list = QTreeWidget()
        self.command_list.setHeaderHidden(True)
        for group, names in commandform.grouped_commands().items():
            parent = QTreeWidgetItem([group])
            font = parent.font(0)
            font.setBold(True)
            parent.setFont(0, font)
            parent.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.command_list.addTopLevelItem(parent)
            for name in names:
                QTreeWidgetItem(parent, [name])
        self.command_filter.textChanged.connect(self._filter_commands)
        self.command_list.currentItemChanged.connect(
            lambda item, _: self._show_command_form(self._command_of(item))
        )
        self.command_list.itemDoubleClicked.connect(lambda *_: self._build_from_form(send=False))
        self.form_host = QWidget()
        self.form_layout = QFormLayout(self.form_host)
        self.form_edits: dict[str, QLineEdit] = {}
        self.form_status = QLabel("Pick a command to fill in its arguments.")
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

        # Results on the right, as tabs: the table of points, the cloud, the last response.
        self.tabs = QTabWidget()
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
        self.tabs.addTab(points, load_icon("point"), "Points")

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
        self.cloud_tab = cloud
        self.tabs.addTab(cloud, load_icon("optical"), "Point cloud")

        self.response = _table(["Name", "Value"])
        self.tabs.addTab(self.response, load_icon("script"), "Last response")
        self._dock("Results", self.tabs, Qt.DockWidgetArea.RightDockWidgetArea)

    def _build_status_bar(self) -> None:
        bar = self.statusBar()
        self.error_label = QLabel()
        self.error_label.setStyleSheet("color: #e74c3c; font-weight: bold;")
        self.error_label.setTextFormat(Qt.TextFormat.PlainText)
        self.clear_errors = QPushButton(load_icon("clear"), "Clear errors")
        self.clear_errors.setFlat(True)
        self.clear_errors.clicked.connect(lambda: self.send("ClearAllErrors()"))
        bar.addWidget(self.error_label, 1)
        bar.addPermanentWidget(self.clear_errors)

    # -- connection -----------------------------------------------------------------------

    def _watch(self, future: Any, what: str) -> None:
        future.add_done_callback(lambda f: self._bridge.finished.emit(f.exception(), what))

    def toggle_connection(self) -> None:
        if self.host.connected:
            self._watch(self.host.disconnect(), "disconnect")
            return
        self._append(f"Connecting to {self.host_edit.text()}:{self.port_spin.value()} ...", "info")
        self._watch(self.host.connect(self.host_edit.text(), self.port_spin.value()), "connect")

    def connect_virtual(self) -> None:
        if self.host.connected:
            self._watch(self.host.disconnect(), "disconnect")
        self._append("Starting a virtual CMM ...", "info")
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
        self.abort_button.setEnabled(connected)
        self.host_edit.setEnabled(not connected)
        self.port_spin.setEnabled(not connected)
        if connected:
            self.connection_label.set_state(
                "on", "Connected" + (" (virtual CMM)" if self.host.embedded else "")
            )
            return
        self.connection_label.set_state("off", "Not connected")
        self.session_led.set_state("off", "No session")
        self.homed_label.set_state("off", "")
        self.user_led.set_state("off", "User disabled")
        self.busy_led.set_state("off", "Ready")
        self.tool_label.clear()
        self.position_label.setText("X -   Y -   Z -")

    # -- sending ----------------------------------------------------------------------------

    def send(self, text: str) -> None:
        self.send_lines([text])

    def send_lines(self, lines: list[str], origin: dialogs.TaskDialog | None = None) -> int:
        """Run the lines one after the other; return the id of the batch (0 if not sent)."""
        if not self.host.connected:
            self._append("Not connected: connect to a server first (F5).", "error")
            if origin is not None:
                origin.set_result(False, "not connected")
            return 0
        for line in lines:
            self.command_line.remember(line)
        batch_id = self._next_batch
        self._next_batch += 1
        self._batches[batch_id] = _Batch(lines, self._next_number, origin)
        self._next_number += len(lines)
        if origin is not None:
            origin.set_busy()
        self._submit(batch_id, lines)
        self._update_busy()
        return batch_id

    def _submit(self, batch_id: int, lines: list[str]) -> None:
        def relay(index: int, text: str, event: object) -> None:
            self._bridge.data.emit(batch_id, index, text, event)

        def done(f: Any) -> None:
            self._bridge.batch_done.emit(
                batch_id, f.exception(), None if f.exception() else f.result()
            )

        self.host.run_lines(lines, relay).add_done_callback(done)

    def _send_typed(self) -> None:
        text = self.command_line.text().strip()
        if text:
            self.send(text)
            self.command_line.clear()

    def query(self, text: str, callback: Callable[[list[DataPayload]], None]) -> None:
        """Run ``text`` and hand its data responses to ``callback`` (on the GUI thread)."""
        batch_id = self.send_lines([text])
        if batch_id:
            self._queries[(batch_id, 0)] = callback

    def _update_busy(self) -> None:
        if not self._batches:
            self.busy_led.set_state("off", "Ready")
            return
        first = next(iter(self._batches.values()))
        running = first.lines[max(0, min(len(first.started), len(first.lines)) - 1)]
        more = len(self._batches) - 1
        self.busy_led.set_state("on", f"Busy: {running}" + (f" (+{more})" if more else ""))

    def _on_batch_done(self, batch_id: int, error: object, ok: object) -> None:
        if batch_id == _STATUS_BATCH:
            self._polling = False
            return
        batch = self._batches.pop(batch_id, None)
        self._update_busy()
        if batch is None:
            return
        if error is not None:
            batch.error = str(error)
            self._append(f"Failed: {error}", "error")
        if batch.origin is not None:
            took = time.monotonic() - batch.begin
            batch.origin.set_result(error is None and bool(ok), batch.error or f"{took:.2f} s")
        for key in [k for k in self._queries if k[0] == batch_id]:
            self._queries.pop(key, None)
            self._collected.pop(key, None)

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
        self.error_label.setText("Error: " + text)
        self.error_label.setToolTip(text)

    def _on_event(self, batch_id: int, index: int, text: str, event: object) -> None:
        status = batch_id == _STATUS_BATCH
        key = (batch_id, index)
        if isinstance(event, Received):
            self._on_data(key, text, event.payload, status)
        if status and not isinstance(event, Failed | ConnectionLost):
            return
        batch = self._batches.get(batch_id)
        number = ""
        if batch is not None:
            if index not in batch.started:
                batch.started[index] = time.monotonic()
                self._append(f"#{batch.first_number + index} > {text}", "cmd")
                self._update_busy()
            number = f"#{batch.first_number + index} "
        elif not status:
            self._append(f"> {text}", "cmd")
        match event:
            case ParseFailed(error):
                self._append(f"{number}Parse error: {error}", "error")
                self._show_error(f"Parse error: {error}")
                self._fail(batch, str(error))
            case Acked():
                pass
            case Received(payload):
                self._append(f"{number}# {payload.to_wire()}")
            case Completed():
                took = time.monotonic() - batch.started[index] if batch else 0.0
                self._append(f"{number}done in {took:.2f} s", "ok")
                self.error_label.clear()
                self._finish_query(key)
                if text.startswith("StartSession"):
                    self.session_led.set_state("on", "Session open")
                elif text.startswith("EndSession"):
                    self.session_led.set_state("off", "No session")
                self._refresh_status()
            case Failed(error):
                self._append(f"{number}{format_error(error)}", "error")
                message = f"{error.number}: {error.text or error.cause}"
                self._show_error(message)
                self._fail(batch, message)
                self._queries.pop(key, None)
                self._collected.pop(key, None)
            case ConnectionLost(error):
                self._append(f"Connection lost: {error}", "error")
                self._show_error(f"Connection lost: {error}")
                self._fail(batch, f"connection lost: {error}")
                self._update_connection_state()

    @staticmethod
    def _fail(batch: _Batch | None, message: str) -> None:
        if batch is not None:
            batch.error = message

    def _finish_query(self, key: tuple[int, int]) -> None:
        callback = self._queries.pop(key, None)
        data = self._collected.pop(key, [])
        if callback is not None:
            callback(data)

    def _on_data(self, key: tuple[int, int], text: str, payload: DataPayload, status: bool) -> None:
        if key in self._queries:
            self._collected.setdefault(key, []).append(payload)
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
            self.homed_label.set_state("on" if homed else "off", "Homed" if homed else "Not homed")
        if "IsUserEnabled" in values:
            enabled = bool(values["IsUserEnabled"])
            self.user_led.set_state(
                "on" if enabled else "off", "User enabled" if enabled else "User disabled"
            )
        if "Tool.Name" in values:
            self.tool_label.setText(f"Tool: {values['Tool.Name']}")
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
        self._submit(_STATUS_BATCH, recipes.status_lines())

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
            dialog.run_requested.connect(lambda lines, d=dialog: self.send_lines(lines, origin=d))
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
            self._append("Not connected: connect to a server first (F5).", "error")
            return
        self._append(
            f"> DataAcquire {options['name']} ({options['acquisition_type']}, "
            f"{len(options['points'])} positions)",
            "cmd",
        )
        dialog = self._dialogs.get("optical")
        if dialog is not None:
            dialog.set_busy()
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
            self.tabs.setCurrentWidget(self.cloud_tab)
            self._append(f"Acquired {len(points)} points", "ok")
            self.error_label.clear()
            dialog = self._dialogs.get("optical")
            if dialog is not None:
                dialog.set_result(True, f"{len(points)} points")
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

    @staticmethod
    def _command_of(item: QTreeWidgetItem | None) -> str:
        """Return the command name of a tree item; empty for a group heading."""
        return item.text(0) if item is not None and item.parent() is not None else ""

    def _filter_commands(self, text: str) -> None:
        needle = text.casefold()
        for g in range(self.command_list.topLevelItemCount()):
            group = self.command_list.topLevelItem(g)
            if group is None:
                continue
            shown = 0
            for c in range(group.childCount()):
                child = group.child(c)
                if child is None:
                    continue
                hidden = needle not in child.text(0).casefold()
                child.setHidden(hidden)
                shown += not hidden
            group.setHidden(shown == 0)
            group.setExpanded(bool(needle) and shown > 0)

    def _show_command_form(self, name: str) -> None:
        while self.form_layout.rowCount():
            self.form_layout.removeRow(0)
        self.form_edits.clear()
        if not name:
            self.form_status.setText("Pick a command to fill in its arguments.")
            return
        fields = commandform.command_fields(name)
        for field in fields:
            edit = QLineEdit()
            edit.setPlaceholderText(field.hint)
            label = field.name + ("" if field.mandatory else " (optional)")
            self.form_layout.addRow(label, edit)
            self.form_edits[field.name] = edit
        self.form_status.setText("" if fields else "No arguments.")

    def _build_from_form(self, *, send: bool) -> None:
        name = self._command_of(self.command_list.currentItem())
        if not name:
            return
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
