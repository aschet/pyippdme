# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""The simulator window: the machine in 3D, and the panels to set it up, run it and teach it.

The window only assembles panels (:mod:`pyippdme.gui.scene_panel`, ``machine_panel``,
``twin_panels``, ``teach_panel``, ``library_panel``) around the 3D view, wires the toolbar and
menus, and shows the machine state in the :class:`~pyippdme.gui.status_strip.StatusStrip`. The
simulation itself is :class:`~pyippdme.twin.DigitalTwin`; a client connects to its server over TCP.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QActionGroup, QCloseEvent, QKeySequence
from PySide6.QtWidgets import (
    QCheckBox,
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from pyippdme.gui.csy_panel import CsyPanel
from pyippdme.gui.icons import app_icon, load_icon
from pyippdme.gui.library_panel import ToolCreator
from pyippdme.gui.machine_panel import MachinePanel
from pyippdme.gui.scene_panel import ScenePanel
from pyippdme.gui.status_strip import StatusStrip
from pyippdme.gui.teach_panel import TeachPanel
from pyippdme.gui.twin_panels import CheckPanel, SafetyPanel, ToolPanel
from pyippdme.gui.viewport import FOLLOW_VIEWS, VIEW_PRESETS, Viewport
from pyippdme.protocol.transport import DEFAULT_PORT
from pyippdme.server.host import ServerHost
from pyippdme.twin import DigitalTwin, TwinEvent

_LOG_LIMIT = 3000
_VIEWS = (
    ("isometric", "view-iso", "Isometric", "1"),
    ("top", "view-top", "Top", "2"),
    ("front", "view-front", "Front", "3"),
    ("right", "view-side", "Side", "4"),
    ("follow", "view-follow", "Follow tool", "5"),
    ("probe", "view-probe", "Probe camera", "6"),
    ("table", "view-table", "Table camera", "7"),
)


class _Bridge(QObject):
    """Hands twin events from the server thread to the GUI thread."""

    event = Signal(object)  # type: ignore[assignment]


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
        self.setWindowIcon(app_icon())
        self.resize(1560, 940)

        self.viewport = Viewport(self.twin)
        self.strip = StatusStrip()
        center = QWidget()
        column = QVBoxLayout(center)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        column.addWidget(self.strip)
        column.addWidget(self.viewport, 1)
        self.setCentralWidget(center)

        self.scene_panel = ScenePanel(self.twin)
        self.machine_panel = MachinePanel(self.twin)
        self.csy_panel = CsyPanel(self.twin)
        self.csy_panel.selected.connect(self._csy_selected)
        self.csy_panel.show_changed.connect(self._csy_shown)
        self.tool_panel = ToolPanel(self.twin)
        self.tool_creator = ToolCreator(self.twin)
        self.check_panel = CheckPanel(self.twin, lambda: self.host.port)
        self.safety_panel = SafetyPanel(self.twin)
        self.teach_panel = TeachPanel(self.twin, self.host.submit, lambda: self.host.port)
        self._build_docks()
        self._build_actions()
        self._build_toolbars()
        self._build_menus()
        self.machine_panel.on_machine_changed += [
            self.viewport.fit,
            self.tool_panel.refresh,
            self.teach_panel.refresh_tools,
        ]
        self.tool_creator.tool_created.connect(lambda _: self.tool_panel.refresh())
        self.scene_panel.changed.connect(self.viewport.update)

        self._painted_key: object = None
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._tick)
        self._timer.start()
        self.strip.set_server(False)
        self.statusBar().showMessage("Start the server (F5) so that a client can connect.")

    # -- compatibility names ------------------------------------------------------------------

    @property
    def scene_list(self):  # type: ignore[no-untyped-def]
        return self.scene_panel.scene_list

    @property
    def pose_boxes(self):  # type: ignore[no-untyped-def]
        return self.scene_panel.pose_boxes

    @property
    def machine_info(self):  # type: ignore[no-untyped-def]
        return self.machine_panel.info

    def add_demo_sample(self) -> None:
        self.scene_panel.add_demo_sample()

    def load_sample(self) -> None:
        self.scene_panel.load_sample()

    def remove_selected(self) -> None:
        self.scene_panel.remove_selected()

    def _refresh_scene_list(self) -> None:
        self.scene_panel.refresh()

    def _refresh_machine_info(self) -> None:
        self.machine_panel.refresh()

    # -- layout -------------------------------------------------------------------------------

    def _dock(self, title: str, widget: QWidget, area: Qt.DockWidgetArea) -> QDockWidget:
        dock = QDockWidget(title, self)
        dock.setWidget(widget)
        dock.setObjectName(title)
        dock.setWindowIcon(load_icon("settings"))
        self.addDockWidget(area, dock)
        return dock

    def _build_docks(self) -> None:
        self._dock("Scene", self.scene_panel, Qt.DockWidgetArea.LeftDockWidgetArea)

        tools = QTabWidget()
        tools.addTab(self.tool_panel, load_icon("tool"), "Edit")
        tools.addTab(self.tool_creator, load_icon("add"), "Create")
        self.tools_tabs = tools
        right = Qt.DockWidgetArea.RightDockWidgetArea
        pages = QTabWidget()
        pages.setTabPosition(QTabWidget.TabPosition.North)
        pages.addTab(self.machine_panel, load_icon("machine"), "Machine")
        pages.addTab(tools, load_icon("tool"), "Tools")
        pages.addTab(self.csy_panel, load_icon("csy"), "Coordinates")
        pages.addTab(self.teach_panel, load_icon("teach"), "Teach-in")
        pages.addTab(self.check_panel, load_icon("check"), "Check")
        pages.addTab(self.safety_panel, load_icon("safety"), "Safety")
        self.pages = pages
        self.right_dock = self._dock("Machine and tools", pages, right)
        self.right_dock.setMinimumWidth(380)

        log = QWidget()
        layout = QVBoxLayout(log)
        layout.setContentsMargins(6, 4, 6, 4)
        row = QHBoxLayout()
        self.wire_check = QCheckBox("Protocol lines")
        self.wire_check.setChecked(True)
        self.wire_check.setToolTip("Show every line the client and the server exchange")
        clear = QPushButton(load_icon("clear"), "Clear")
        clear.clicked.connect(lambda: self.log.clear())
        self.stats_label = QLabel()
        row.addWidget(self.wire_check)
        row.addWidget(self.stats_label, 1)
        row.addWidget(clear)
        layout.addLayout(row)
        self.log: QPlainTextEdit = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(_LOG_LIMIT)
        self.log.setStyleSheet("font-family: monospace;")
        layout.addWidget(self.log)
        self.log_dock = self._dock("Log", log, Qt.DockWidgetArea.BottomDockWidgetArea)
        self.resizeDocks([self.log_dock], [140], Qt.Orientation.Vertical)

    # -- actions ------------------------------------------------------------------------------

    def _action(
        self,
        icon: str,
        text: str,
        slot: object,
        shortcut: str | QKeySequence.StandardKey | None = None,
        tip: str = "",
        *,
        checkable: bool = False,
    ) -> QAction:
        action = QAction(load_icon(icon), text, self)
        action.setCheckable(checkable)
        if shortcut is not None:
            action.setShortcut(shortcut)
        action.setToolTip(
            f"{tip or text}" + (f" ({action.shortcut().toString()})" if shortcut else "")
        )
        action.setStatusTip(tip or text)
        if checkable:
            action.toggled.connect(slot)
        else:
            action.triggered.connect(lambda _=False: slot())  # type: ignore[operator]
        return action

    def _build_actions(self) -> None:
        self.server_action = self._action(
            "server-start", "Start server", self.toggle_server, "F5", "Let a client connect"
        )
        self.estop_action = self._action(
            "abort",
            "Emergency stop",
            self._estop_toggled,
            "Ctrl+E",
            "Press the emergency stop: the machine brakes and loses its reference",
            checkable=True,
        )
        self.open_action = self._action(
            "open",
            "Open sample",
            self.load_sample,
            QKeySequence.StandardKey.Open,
            "Load a CAD part",
        )
        self.demo_action = self._action(
            "sample", "Demo block", self.add_demo_sample, None, "Place the built-in demo block"
        )
        self.artifact_action = self._action(
            "check",
            "Check artefact",
            self.check_panel.add_artifact,
            None,
            "Place the check artefact",
        )
        self.view_group = QActionGroup(self)
        self.view_actions: dict[str, QAction] = {}
        for name, icon, text, key in _VIEWS:
            action = self._action(
                icon,
                text,
                lambda on, n=name: self._view_toggled(n, on),
                key,
                f"{text} camera",
                checkable=True,
            )
            self.view_group.addAction(action)
            self.view_actions[name] = action
        self.view_actions["isometric"].setChecked(True)
        self.fit_action = self._action(
            "view-fit",
            "Fit",
            lambda: self.set_view("fit"),
            "F",
            "Fit the whole machine in the view",
        )
        self.screenshot_action = self._action(
            "view-front", "Screenshot", self.save_screenshot, None, "Save the 3D view as an image"
        )

    def _build_toolbars(self) -> None:
        server = QToolBar("Server")
        server.setMovable(False)
        server.setObjectName("Server")
        server.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.addToolBar(server)
        self.host_edit = QLineEdit("127.0.0.1")
        self.host_edit.setFixedWidth(130)
        self.host_edit.setToolTip("Address the server listens on")
        self.port_spin = QSpinBox()
        self.port_spin.setRange(0, 65535)
        self.port_spin.setValue(DEFAULT_PORT)
        self.port_spin.setToolTip("Port of the server (0 picks a free one)")
        server.addWidget(QLabel(" Listen on "))
        server.addWidget(self.host_edit)
        server.addWidget(QLabel(" : "))
        server.addWidget(self.port_spin)
        server.addAction(self.server_action)
        server.addAction(self.estop_action)

        scene = QToolBar("Scene")
        scene.setMovable(False)
        scene.setObjectName("Scene")
        scene.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        self.addToolBar(scene)
        scene.addAction(self.open_action)
        scene.addAction(self.demo_action)
        fixture = QToolButton()
        fixture.setText("Fixture")
        fixture.setIcon(load_icon("fixture"))
        fixture.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        fixture.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(fixture)
        menu.addAction("Box", lambda: self.scene_panel.add_fixture("box"))
        menu.addAction("Cylinder", lambda: self.scene_panel.add_fixture("cylinder"))
        menu.addAction("From CAD…", self.scene_panel.load_fixture)
        fixture.setMenu(menu)
        fixture.setToolTip("Add a fixture to hold the sample")
        scene.addWidget(fixture)
        scene.addAction(self.artifact_action)

        views = QToolBar("Views")
        views.setMovable(False)
        views.setObjectName("Views")
        views.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.addToolBar(views)
        for action in self.view_actions.values():
            views.addAction(action)
        views.addAction(self.fit_action)
        views.addAction(self.screenshot_action)

        work = QToolBar("Work")
        work.setMovable(False)
        work.setObjectName("Work")
        work.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        self.addToolBar(work)
        for icon, text, page, tip in (
            ("teach", "Teach-in", self.teach_panel, "Jog box, manual points and programs"),
            (
                "library",
                "Tool library",
                self.tool_creator,
                "Build a tool from the component library",
            ),
            ("safety", "Safety", self.safety_panel, "Emergency stop, air supply, qualification"),
        ):
            action = self._action(icon, text, lambda p=page: self.show_page(p), None, tip)
            work.addAction(action)

    def _build_menus(self) -> None:
        bar = self.menuBar()
        file_menu = bar.addMenu("&File")
        file_menu.addAction(self.open_action)
        file_menu.addAction("Load fixture CAD…", self.scene_panel.load_fixture)
        file_menu.addSeparator()
        file_menu.addAction("Load machine…", self.machine_panel.load_machine)
        file_menu.addAction("Export machine…", self._export_machine)
        file_menu.addSeparator()
        file_menu.addAction(self.screenshot_action)
        quit_action = QAction("Quit", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        machine_menu = bar.addMenu("&Machine")
        machine_menu.addAction(self.server_action)
        machine_menu.addAction(self.estop_action)

        view_menu = bar.addMenu("&View")
        for action in self.view_actions.values():
            view_menu.addAction(action)
        view_menu.addAction(self.fit_action)
        view_menu.addSeparator()
        for text, attr in (
            ("Show machine", "show_machine"),
            ("Show probed points", "show_contacts"),
            ("Show scan clouds", "show_clouds"),
        ):
            action = QAction(text, self, checkable=True)
            action.setChecked(True)
            action.toggled.connect(lambda on, a=attr: self._set_view_flag(a, on))
            view_menu.addAction(action)
        view_menu.addSeparator()
        for dock in self.findChildren(QDockWidget):
            view_menu.addAction(dock.toggleViewAction())

    # -- views --------------------------------------------------------------------------------

    def set_view(self, name: str) -> None:
        """Switch the camera (``isometric``, ``top``, ``front``, ``right``, ``follow``, ...)."""
        if name in VIEW_PRESETS or name in FOLLOW_VIEWS or name == "fit":
            self.viewport.set_view(name)
            if name in self.view_actions:
                self.view_actions[name].setChecked(True)

    def _view_toggled(self, name: str, on: bool) -> None:
        if on:
            self.set_view(name)

    def _set_view_flag(self, name: str, on: bool) -> None:
        setattr(self.viewport, name, on)
        self.viewport.update()

    def show_page(self, page: QWidget) -> None:
        """Bring a panel of the right dock to the front."""
        self.right_dock.show()
        if page is self.tool_creator:
            self.pages.setCurrentWidget(self.tools_tabs)
            self.tools_tabs.setCurrentWidget(self.tool_creator)
        else:
            self.pages.setCurrentWidget(page)

    # -- server -------------------------------------------------------------------------------

    def toggle_server(self) -> None:
        if self.host.running:
            self.host.stop()
            self.server_action.setIcon(load_icon("server-start"))
            self.server_action.setText("Start server")
            self.host_edit.setEnabled(True)
            self.port_spin.setEnabled(True)
            self.strip.set_server(False)
            self.statusBar().showMessage("Server stopped")
            return
        try:
            server = self.twin.create_server()
            port = self.host.start(server, self.host_edit.text(), self.port_spin.value())
        except OSError as error:
            QMessageBox.critical(self, "Cannot start the server", str(error))
            return
        self.port_spin.setValue(port)
        self.server_action.setIcon(load_icon("server-stop"))
        self.server_action.setText("Stop server")
        self.host_edit.setEnabled(False)
        self.port_spin.setEnabled(False)
        address = f"{self.host_edit.text()}:{port}"
        self.strip.set_server(True, address)
        self.statusBar().showMessage(f"A client can connect to {address}", 6000)

    def _estop_toggled(self, pressed: bool) -> None:
        if self.safety_panel.estop.isChecked() != pressed:
            self.safety_panel.estop.setChecked(pressed)
        self.estop_action.setText("Release emergency stop" if pressed else "Emergency stop")

    def _export_machine(self) -> None:
        path = self.machine_panel.export_machine()
        if path:
            self.statusBar().showMessage(f"Wrote {path}", 6000)

    # -- status and events ----------------------------------------------------------------------

    def _csy_selected(self, name: object) -> None:
        self.viewport.csy_selected = name if isinstance(name, str) else None
        self.viewport.update()

    def _csy_shown(self, shown: bool) -> None:
        self.viewport.show_csys = shown
        self.viewport.update()

    def _tick(self) -> None:
        snap = self.twin.snapshot()
        self.csy_panel.refresh()
        name = self.csy_panel.readout_combo.currentText()
        self.strip.update_from(snap, (name, self.twin.to_csy(snap.position, name)))
        self.stats_label.setText(
            f"{len(self.twin.contacts)} probed points, "
            f"{sum(len(c) for c in self.twin.clouds)} scan points"
        )
        if self.estop_action.isChecked() != snap.estop:
            self.estop_action.blockSignals(True)
            self.estop_action.setChecked(snap.estop)
            self.estop_action.blockSignals(False)
        # Painting the scene is the expensive part: only repaint when something changed.
        key = (
            snap.position,
            snap.rotary,
            snap.tool_name,
            snap.head_position,
            snap.changing_tool,
            self.twin.scene_version,
            self.csy_panel._signature,
            len(self.twin.contacts),
            len(self.twin.clouds),
        )
        if key != self._painted_key:
            self._painted_key = key
            self.viewport.update()

    def _on_event(self, event: object) -> None:
        if not isinstance(event, TwinEvent):
            return
        if event.kind in ("scene", "machine", "tools"):
            if event.kind == "machine":
                self.machine_panel.refresh()
            if event.kind in ("machine", "tools"):
                self.tool_panel.refresh()
                self.teach_panel.refresh_tools()
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
        self.teach_panel.shutdown()
        self.check_panel.shutdown()
        self.host.shutdown()
        super().closeEvent(event)
