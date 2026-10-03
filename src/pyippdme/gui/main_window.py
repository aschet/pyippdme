# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""The simulator window: 3D view of the twin plus the controls to set it up and run it."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QCloseEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from pyippdme.gui.twin_panels import CheckPanel, SafetyPanel, ToolPanel
from pyippdme.gui.viewport import Viewport
from pyippdme.protocol.transport import DEFAULT_PORT
from pyippdme.server.host import ServerHost
from pyippdme.twin import DigitalTwin, MachineModel, TwinEvent, demo_sample, geometry
from pyippdme.twin.cad import SUPPORTED_SUFFIXES, CadError
from pyippdme.twin.objects import SceneObject
from pyippdme.twin.spec import PRESETS

_TIME_SCALES = (("Instant", 0.0), ("Real time", 1.0), ("2x", 2.0), ("5x", 5.0), ("20x", 20.0))
_CAD_FILTER = "CAD files (" + " ".join(f"*{s}" for s in SUPPORTED_SUFFIXES) + ");;All files (*)"
_LOG_LIMIT = 3000


class _Bridge(QObject):
    """Hands twin events from the server thread to the GUI thread."""

    event = Signal(object)  # type: ignore[assignment]


def _spin(
    lo: float, hi: float, value: float = 0.0, step: float = 1.0, decimals: int = 3
) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(lo, hi)
    box.setDecimals(decimals)
    box.setSingleStep(step)
    box.setValue(value)
    box.setKeyboardTracking(False)
    return box


class MainWindow(QMainWindow):
    """Virtual CMM with a visible machine; clients connect over TCP with I++ DME."""

    def __init__(self, twin: DigitalTwin | None = None, host: ServerHost | None = None) -> None:
        super().__init__()
        self.twin = twin or DigitalTwin(time_scale=1.0)
        self.host = host or ServerHost()
        self._bridge = _Bridge()
        self._bridge.event.connect(self._on_event)
        self.twin.add_listener(self._bridge.event.emit)
        self.setWindowTitle("pyippdme virtual CMM")
        self.resize(1500, 900)

        self.viewport = Viewport(self.twin)
        self.setCentralWidget(self.viewport)
        self._selected: SceneObject | None = None
        self._updating_pose = False

        self._build_server_dock()
        self._build_machine_dock()
        self._build_scene_dock()
        self._build_jog_dock()
        self._build_log_dock()
        self._build_panels()
        self._build_menus()
        self.statusBar().showMessage("Server stopped")

        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._tick)
        self._timer.start()
        self._refresh_scene_list()
        self._refresh_machine_info()

    # -- docks -------------------------------------------------------------------------

    def _dock(self, title: str, widget: QWidget, area: Qt.DockWidgetArea) -> QDockWidget:
        dock = QDockWidget(title, self)
        dock.setWidget(widget)
        dock.setObjectName(title)
        self.addDockWidget(area, dock)
        return dock

    def _build_server_dock(self) -> None:
        w = QWidget()
        form = QFormLayout(w)
        self.host_edit = QLineEdit("127.0.0.1")
        self.port_spin = QSpinBox()
        self.port_spin.setRange(0, 65535)
        self.port_spin.setValue(DEFAULT_PORT)
        self.start_button = QPushButton("Start server")
        self.start_button.clicked.connect(self.toggle_server)
        self.client_label = QLabel("no client")
        self.class_label = QLabel("")
        self.class_label.setWordWrap(True)
        form.addRow("Host", self.host_edit)
        form.addRow("Port", self.port_spin)
        form.addRow(self.start_button)
        form.addRow("Client", self.client_label)
        form.addRow("Machine class", self.class_label)
        self._dock("Server", w, Qt.DockWidgetArea.LeftDockWidgetArea)

    def _build_machine_dock(self) -> None:
        w = QWidget()
        layout = QVBoxLayout(w)
        form = QFormLayout()
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(list(PRESETS))
        self.preset_combo.currentTextChanged.connect(self._preset_changed)
        self.rotary_check = QCheckBox("Rotary table")
        self.rotary_check.toggled.connect(
            lambda _: self._preset_changed(self.preset_combo.currentText())
        )
        self.time_combo = QComboBox()
        for label, _ in _TIME_SCALES:
            self.time_combo.addItem(label)
        self.time_combo.setCurrentIndex(1)
        self.time_combo.currentIndexChanged.connect(
            lambda i: setattr(self.twin, "time_scale", _TIME_SCALES[i][1])
        )
        self.override = QSlider(Qt.Orientation.Horizontal)
        self.override.setRange(1, 100)
        self.override.setValue(100)
        self.override.valueChanged.connect(lambda v: setattr(self.twin, "speed_override", v / 100))
        self.noise_check = QCheckBox("Measuring noise (MPE)")
        self.noise_check.setChecked(True)
        self.noise_check.toggled.connect(lambda on: setattr(self.twin, "noise_enabled", on))
        self.temperature = _spin(0, 60, 20.0, 0.5, 1)
        self.temperature.valueChanged.connect(lambda v: setattr(self.twin, "temperature", v))
        form.addRow("Preset", self.preset_combo)
        form.addRow(self.rotary_check)
        form.addRow("Motion", self.time_combo)
        form.addRow("Speed override", self.override)
        form.addRow(self.noise_check)
        form.addRow("Part temperature °C", self.temperature)
        layout.addLayout(form)
        self.machine_info = QLabel()
        self.machine_info.setWordWrap(True)
        layout.addWidget(self.machine_info)
        row = QHBoxLayout()
        load = QPushButton("Load machine…")
        load.clicked.connect(self.load_machine)
        export = QPushButton("Export…")
        export.clicked.connect(self.export_machine)
        row.addWidget(load)
        row.addWidget(export)
        layout.addLayout(row)
        layout.addStretch(1)
        self._dock("Machine", w, Qt.DockWidgetArea.LeftDockWidgetArea)

    def _build_scene_dock(self) -> None:
        w = QWidget()
        layout = QVBoxLayout(w)
        self.scene_list = QListWidget()
        self.scene_list.currentRowChanged.connect(self._select_object)
        layout.addWidget(self.scene_list)
        buttons = QGridLayout()
        for i, (text, slot) in enumerate(
            (
                ("Load sample CAD…", self.load_sample),
                ("Demo sample", self.add_demo_sample),
                ("Add box fixture", lambda: self.add_fixture("box")),
                ("Add cylinder fixture", lambda: self.add_fixture("cylinder")),
                ("Load fixture CAD…", self.load_fixture),
                ("Remove", self.remove_selected),
            )
        ):
            b = QPushButton(text)
            b.clicked.connect(slot)
            buttons.addWidget(b, i // 2, i % 2)
        layout.addLayout(buttons)
        group = QGroupBox("Placement (machine coordinates)")
        grid = QGridLayout(group)
        self.pose_boxes: list[QDoubleSpinBox] = []
        for i, name in enumerate(("X", "Y", "Z", "RX", "RY", "RZ")):
            box = _spin(-5000, 5000, 0.0, 1.0 if i < 3 else 5.0, 3)
            box.valueChanged.connect(self._pose_edited)
            self.pose_boxes.append(box)
            grid.addWidget(QLabel(name), i % 3, (i // 3) * 2)
            grid.addWidget(box, i % 3, (i // 3) * 2 + 1)
        self.on_rotary_check = QCheckBox("Mounted on rotary table")
        self.on_rotary_check.toggled.connect(self._on_rotary_toggled)
        self.visible_check = QCheckBox("Visible")
        self.visible_check.setChecked(True)
        self.visible_check.toggled.connect(self._visible_toggled)
        drop = QPushButton("Rest on table, centre")
        drop.clicked.connect(self.rest_selected)
        grid.addWidget(self.on_rotary_check, 3, 0, 1, 4)
        grid.addWidget(self.visible_check, 4, 0, 1, 2)
        grid.addWidget(drop, 4, 2, 1, 2)
        layout.addWidget(group)
        self._dock("Scene", w, Qt.DockWidgetArea.RightDockWidgetArea)

    def _build_jog_dock(self) -> None:
        w = QWidget()
        layout = QVBoxLayout(w)
        self.dro = QLabel()
        self.dro.setStyleSheet("font-family: monospace; font-size: 13px;")
        self.state_label = QLabel()
        self.state_label.setWordWrap(True)
        layout.addWidget(self.dro)
        layout.addWidget(self.state_label)
        step_row = QHBoxLayout()
        step_row.addWidget(QLabel("Jog step mm"))
        self.jog_step = _spin(0.01, 200, 5.0, 1.0, 2)
        step_row.addWidget(self.jog_step)
        layout.addLayout(step_row)
        grid = QGridLayout()
        for col, axis in enumerate("XYZ"):
            for row, sign in enumerate((1, -1)):
                b = QPushButton(f"{'+' if sign > 0 else '-'}{axis}")
                b.clicked.connect(lambda _=False, a=col, s=sign: self._jog(a, s))
                grid.addWidget(b, row, col)
        layout.addLayout(grid)
        stats = QHBoxLayout()
        self.stats_label = QLabel()
        clear = QPushButton("Clear points")
        clear.clicked.connect(self.twin.clear_measurements)
        stats.addWidget(self.stats_label)
        stats.addWidget(clear)
        layout.addLayout(stats)
        layout.addStretch(1)
        self._dock("Machine position", w, Qt.DockWidgetArea.RightDockWidgetArea)

    def _build_panels(self) -> None:
        self.tool_panel = ToolPanel(self.twin)
        self.check_panel = CheckPanel(self.twin, self.host.submit, lambda: self.host.port)
        self.safety_panel = SafetyPanel(self.twin)
        right = Qt.DockWidgetArea.RightDockWidgetArea
        tools = self._dock("Tools", self.tool_panel, right)
        check = self._dock("Check artefact", self.check_panel, right)
        safety = self._dock("Safety", self.safety_panel, right)
        scene = next(d for d in self.findChildren(QDockWidget) if d.windowTitle() == "Scene")
        for dock in (tools, check, safety):
            self.tabifyDockWidget(scene, dock)
        scene.raise_()

    def _build_log_dock(self) -> None:
        w = QWidget()
        layout = QVBoxLayout(w)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(_LOG_LIMIT)
        self.log.setStyleSheet("font-family: monospace;")
        row = QHBoxLayout()
        self.wire_check = QCheckBox("Show protocol lines")
        self.wire_check.setChecked(True)
        clear = QPushButton("Clear")
        clear.clicked.connect(self.log.clear)
        row.addWidget(self.wire_check)
        row.addStretch(1)
        row.addWidget(clear)
        layout.addLayout(row)
        layout.addWidget(self.log)
        self._dock("Log", w, Qt.DockWidgetArea.BottomDockWidgetArea)

    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        for text, slot in (
            ("Load sample CAD…", self.load_sample),
            ("Load machine…", self.load_machine),
            ("Export machine…", self.export_machine),
            ("Save screenshot…", self.save_screenshot),
        ):
            action = QAction(text, self)
            action.triggered.connect(slot)
            file_menu.addAction(action)
        file_menu.addSeparator()
        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)
        view_menu = self.menuBar().addMenu("&View")
        reset = QAction("Reset camera", self)
        reset.triggered.connect(self.viewport.fit)
        view_menu.addAction(reset)
        for text, attr in (
            ("Show machine", "show_machine"),
            ("Show probed points", "show_contacts"),
            ("Show scan clouds", "show_clouds"),
        ):
            action = QAction(text, self, checkable=True)
            action.setChecked(True)
            action.toggled.connect(lambda on, a=attr: self._set_view_flag(a, on))
            view_menu.addAction(action)
        for dock in self.findChildren(QDockWidget):
            view_menu.addAction(dock.toggleViewAction())

    # -- server ------------------------------------------------------------------------

    def _set_view_flag(self, name: str, on: bool) -> None:
        setattr(self.viewport, name, on)
        self.viewport.update()

    def toggle_server(self) -> None:
        if self.host.running:
            self.host.stop()
            self.start_button.setText("Start server")
            self.statusBar().showMessage("Server stopped")
            return
        try:
            server = self.twin.create_server()
            port = self.host.start(server, self.host_edit.text(), self.port_spin.value())
        except OSError as error:
            QMessageBox.critical(self, "Cannot start the server", str(error))
            return
        self.port_spin.setValue(port)
        self.class_label.setText(str(server.machine_class))
        self.start_button.setText("Stop server")
        self.statusBar().showMessage(f"Listening on {self.host_edit.text()}:{port}")

    # -- machine -----------------------------------------------------------------------

    def _preset_changed(self, name: str) -> None:
        if name in PRESETS:
            self.twin.set_machine(MachineModel.default(name, rotary=self.rotary_check.isChecked()))
            self._refresh_machine_info()
            self.viewport.fit()

    def load_machine(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load machine", "", "machine.toml or STEP (*.toml *.step *.stp);;All files (*)"
        )
        if not path:
            return
        try:
            p = Path(path)
            model = (
                MachineModel.from_directory(p.parent)
                if p.suffix.lower() == ".toml"
                else MachineModel.from_step(p)
            )
        except (CadError, OSError, ValueError, KeyError) as error:
            QMessageBox.critical(self, "Cannot load the machine", str(error))
            return
        self.twin.set_machine(model)
        self._refresh_machine_info()
        self.viewport.fit()

    def export_machine(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Export machine to directory")
        if directory:
            path = self.twin.machine.export(directory)
            self.statusBar().showMessage(f"Wrote {path}")

    def _refresh_machine_info(self) -> None:
        s = self.twin.machine.spec
        derived = f"\nEstimated from CAD: {', '.join(s.derived)}" if s.derived else ""
        self.machine_info.setText(
            f"{s.name}\nTravel {s.travel[0]:g} x {s.travel[1]:g} x {s.travel[2]:g} mm\n"
            f"{s.max_speed:g} mm/s, {s.acceleration:g} mm/s²\n"
            f"MPE_E = {s.accuracy.a_um:g} + L/{s.accuracy.k:g} µm, "
            f"MPE_P = {s.accuracy.probing_um:g} µm"
            f"{derived}"
        )

    # -- scene -------------------------------------------------------------------------

    def _cad_dialog(self, title: str) -> str:
        path, _ = QFileDialog.getOpenFileName(self, title, "", _CAD_FILTER)
        return path

    def load_sample(self) -> None:
        path = self._cad_dialog("Load sample CAD")
        if path:
            self._add(lambda: self.twin.load_sample(path))

    def load_fixture(self) -> None:
        path = self._cad_dialog("Load fixture CAD")
        if path:
            self._add(
                lambda: self.twin.place_sample(
                    SceneObject.from_file(path, "fixture"), replace=False
                )
            )

    def add_demo_sample(self) -> None:
        self._add(lambda: self.twin.place_sample(demo_sample()))

    def add_fixture(self, kind: str) -> None:
        size = (60.0, 40.0, 25.0) if kind == "box" else (20.0, 40.0, 0.0)
        s = self.twin.machine.spec
        self._add(
            lambda: self.twin.add_fixture(
                kind, size, (s.travel[0] / 2 + 100, s.travel[1] / 2, s.table_top_z)
            )
        )

    def _add(self, make: object) -> None:
        try:
            obj = make()  # type: ignore[operator]
        except (CadError, OSError) as error:
            QMessageBox.critical(self, "Cannot load the file", str(error))
            return
        self._refresh_scene_list(select=obj)

    def remove_selected(self) -> None:
        if self._selected is not None:
            self.twin.remove_object(self._selected)
            self._selected = None
            self._refresh_scene_list()

    def rest_selected(self) -> None:
        if self._selected is None:
            return
        obj = self._selected
        self.twin.rest_on_table(obj)
        self.twin.set_pose(obj, obj.pose)
        self._load_pose(obj)

    def _refresh_scene_list(self, select: SceneObject | None = None) -> None:
        self.scene_list.blockSignals(True)
        self.scene_list.clear()
        for obj in self.twin.objects:
            item = QListWidgetItem(f"[{obj.kind}] {obj.name}")
            item.setData(Qt.ItemDataRole.UserRole, obj.id)
            self.scene_list.addItem(item)
        self.scene_list.blockSignals(False)
        row = next((i for i, o in enumerate(self.twin.objects) if o is select), -1)
        if row < 0 and self.twin.objects:
            row = 0
        self.scene_list.setCurrentRow(row)
        self._select_object(row)

    def _select_object(self, row: int) -> None:
        self._selected = self.twin.objects[row] if 0 <= row < len(self.twin.objects) else None
        if self._selected is not None:
            self._load_pose(self._selected)

    def _load_pose(self, obj: SceneObject) -> None:
        self._updating_pose = True
        for box, value in zip(self.pose_boxes, geometry.decompose(obj.pose), strict=True):
            box.setValue(value)
        self.on_rotary_check.setChecked(obj.on_rotary)
        self.visible_check.setChecked(obj.visible)
        self._updating_pose = False

    def _pose_edited(self) -> None:
        if self._updating_pose or self._selected is None:
            return
        self.twin.set_pose(self._selected, geometry.pose(*(b.value() for b in self.pose_boxes)))

    def _on_rotary_toggled(self, on: bool) -> None:
        if not self._updating_pose and self._selected is not None:
            self._selected.on_rotary = on
            self.twin.set_pose(self._selected, self._selected.pose)

    def _visible_toggled(self, on: bool) -> None:
        if not self._updating_pose and self._selected is not None:
            self._selected.visible = on
            self.twin.set_pose(self._selected, self._selected.pose)

    # -- jog and status ----------------------------------------------------------------

    def _jog(self, axis: int, sign: int) -> None:
        delta = [0.0, 0.0, 0.0]
        delta[axis] = sign * self.jog_step.value()
        if not self.twin.jog(*delta):
            self.statusBar().showMessage(
                "Jog needs a connected client with the jog box enabled, within the machine volume",
                4000,
            )

    def _tick(self) -> None:
        snap = self.twin.snapshot()
        self.dro.setText(
            f"X {snap.position[0]:10.3f}\nY {snap.position[1]:10.3f}\n"
            f"Z {snap.position[2]:10.3f}\nR {snap.rotary:10.3f}"
        )
        flags = [
            "moving" if snap.moving else "idle",
            "homed" if snap.homed else "not homed",
            f"tool {snap.tool_name} ({snap.tool_mode})",
            "qualified" if snap.qualified else "not qualified",
        ]
        if snap.estop:
            flags.append("EMERGENCY STOP")
        if not snap.air_ok:
            flags.append("no air")
        self.state_label.setText(", ".join(flags) + (f"\n{snap.error}" if snap.error else ""))
        self.client_label.setText(snap.peer or "no client")
        self.stats_label.setText(
            f"{len(self.twin.contacts)} points, {sum(len(c) for c in self.twin.clouds)} scan points"
        )
        if snap.machine_class:
            self.class_label.setText(snap.machine_class)
        self.viewport.update()

    def _on_event(self, event: object) -> None:
        if not isinstance(event, TwinEvent):
            return
        if event.kind in ("scene", "machine", "tools"):
            if event.kind == "machine":
                self._refresh_machine_info()
            if event.kind in ("machine", "tools"):
                self.tool_panel.refresh()
            return
        if event.kind in ("qualified", "safety"):
            self.safety_panel.refresh()
            self.safety_panel.sync()
        if event.kind in ("line_in", "line_out"):
            if self.wire_check.isChecked():
                self.log.appendPlainText(
                    ("> " if event.kind == "line_in" else "< ") + str(event.data["line"])
                )
            return
        detail = " ".join(f"{k}={v}" for k, v in event.data.items())
        self.log.appendPlainText(f"* {event.kind} {detail}")

    def save_screenshot(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save screenshot", "cmm.png", "PNG (*.png)")
        if path:
            self.viewport.render_to_image(1600, 1000).save(path)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        self._timer.stop()
        self.host.shutdown()
        super().closeEvent(event)
