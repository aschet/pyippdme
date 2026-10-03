# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Task dialogs of the command client: fill in a form, run the command. No command text to type.

Each dialog stays open (it is not modal) so a measurement can be repeated with changed values.
It never talks to the machine itself: it builds command lines with :mod:`pyippdme.client.recipes`
and asks its owner to run them through :attr:`TaskDialog.run_requested`.
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from pyippdme.client import optical, recipes
from pyippdme.gui.icons import load_icon
from pyippdme.types.csy import CSY_CHAIN, LIVE_TRANSFORM_NAMES
from pyippdme.types.vec3 import Vec3

PositionProvider = Callable[[], Vec3 | None]


def _spin(
    lo: float = -100000.0, hi: float = 100000.0, value: float = 0.0, decimals: int = 3
) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(lo, hi)
    box.setDecimals(decimals)
    box.setValue(value)
    box.setKeyboardTracking(False)
    box.setSuffix(" mm")
    return box


class VectorEdit(QWidget):
    """Three number boxes for a point or a direction."""

    changed = Signal()

    def __init__(self, suffix: str = " mm", labels: str = "XYZ") -> None:
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.boxes = []
        for label in labels:
            box = _spin()
            box.setSuffix(suffix)
            box.setPrefix(f"{label} ")
            box.valueChanged.connect(lambda _=0.0: self.changed.emit())
            layout.addWidget(box)
            self.boxes.append(box)

    def value(self) -> Vec3:
        return (self.boxes[0].value(), self.boxes[1].value(), self.boxes[2].value())

    def set_value(self, value: Vec3) -> None:
        for box, v in zip(self.boxes, value, strict=True):
            box.setValue(v)


class TaskDialog(QDialog):
    """Base: a form, a live preview of the command, and Run / Close."""

    run_requested = Signal(list)

    def __init__(self, title: str, icon: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowIcon(load_icon(icon))
        self.setModal(False)
        outer = QVBoxLayout(self)
        heading = QLabel(f"<b>{title}</b>")
        outer.addWidget(heading)
        self.form = QFormLayout()
        outer.addLayout(self.form)
        self.preview = QLineEdit()
        self.preview.setReadOnly(True)
        self.preview.setFont(self.font())
        outer.addWidget(self.preview)
        row = QHBoxLayout()
        self.outcome = QLabel("")
        self.outcome.setWordWrap(True)
        row.addWidget(self.outcome, 1)
        self.run_button = QPushButton(load_icon("run"), "Run")
        self.run_button.setDefault(True)
        self.run_button.clicked.connect(self._run)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        row.addWidget(self.run_button)
        row.addWidget(close)
        outer.addLayout(row)

    def lines(self) -> list[str]:
        raise NotImplementedError

    def refresh(self) -> None:
        try:
            self.preview.setText("  ;  ".join(self.lines()))
            self.run_button.setEnabled(True)
        except ValueError as exc:
            self.preview.setText(str(exc))
            self.run_button.setEnabled(False)

    def _run(self) -> None:
        self.run_requested.emit(self.lines())

    def set_busy(self) -> None:
        """Show that the commands of this dialog are running."""
        self.run_button.setEnabled(False)
        self.outcome.setStyleSheet("color: #7f8c8d;")
        self.outcome.setText("Running ...")

    def set_result(self, ok: bool, text: str) -> None:
        """Show how the commands of this dialog ended."""
        self.outcome.setStyleSheet("color: #27ae60;" if ok else "color: #e74c3c;")
        self.outcome.setText(("Done. " if ok else "Failed: ") + text)
        self.refresh()

    def showEvent(self, event: object) -> None:  # noqa: N802
        self.refresh()
        super().showEvent(event)  # type: ignore[arg-type]


def _connect_all(dialog: TaskDialog, *widgets: QWidget) -> None:
    for w in widgets:
        for name in ("changed", "valueChanged", "toggled", "currentIndexChanged", "textChanged"):
            signal = getattr(w, name, None)
            if signal is not None:
                signal.connect(lambda *_: dialog.refresh())
                break


class MoveDialog(TaskDialog):
    """Move to a position (``GoTo``) or by a distance (``Step``)."""

    def __init__(self, position: PositionProvider, parent: QWidget | None = None) -> None:
        super().__init__("Move", "goto", parent)
        self._position = position
        self._axes: dict[str, tuple[QCheckBox, QDoubleSpinBox]] = {}
        for axis in "XYZR":
            check = QCheckBox(axis)
            check.setChecked(axis != "R")
            box = _spin()
            if axis == "R":
                box.setSuffix(" °")
            row = QHBoxLayout()
            row.addWidget(check)
            row.addWidget(box, 1)
            self.form.addRow(row)
            self._axes[axis] = (check, box)
            _connect_all(self, check, box)
        self.relative = QCheckBox("Relative to the current position (Step)")
        self.sync = QCheckBox("All axes start and end together (Sync)")
        current = QPushButton(load_icon("point"), "Take current position")
        current.clicked.connect(self._take_position)
        self.form.addRow(self.relative)
        self.form.addRow(self.sync)
        self.form.addRow(current)
        _connect_all(self, self.relative, self.sync)

    def _take_position(self) -> None:
        position = self._position()
        if position is not None:
            for axis, v in zip("XYZ", position, strict=True):
                self._axes[axis][1].setValue(v)

    def lines(self) -> list[str]:
        values = {a: (b.value() if c.isChecked() else None) for a, (c, b) in self._axes.items()}
        if all(v is None for v in values.values()):
            raise ValueError("choose at least one axis")
        return [
            recipes.goto_line(
                values["X"],
                values["Y"],
                values["Z"],
                r=values["R"],
                sync=True if self.sync.isChecked() else None,
                relative=self.relative.isChecked(),
            )
        ]


class _DirectionPicker(QWidget):
    """Six axis buttons that set a direction vector."""

    def __init__(self, target: VectorEdit) -> None:
        super().__init__()
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        for index, (name, vector) in enumerate(recipes.AXIS_DIRECTIONS.items()):
            button = QPushButton(name)
            button.setFixedWidth(44)
            button.clicked.connect(lambda _=False, v=vector: target.set_value(v))
            layout.addWidget(button, index % 2, index // 2)


class PointDialog(TaskDialog):
    """Measure a point: nominal position and probing direction."""

    def __init__(self, position: PositionProvider, parent: QWidget | None = None) -> None:
        super().__init__("Measure a point", "point", parent)
        self._position = position
        self.nominal = VectorEdit()
        self.direction = VectorEdit("", "IJK")
        self.direction.set_value((0.0, 0.0, -1.0))
        self.count = QSpinBox()
        self.count.setRange(1, 100)
        self.count.setSuffix(" times")
        current = QPushButton(load_icon("goto"), "Take current position")
        current.clicked.connect(self._take_position)
        self.form.addRow("Nominal point", self.nominal)
        self.form.addRow("", current)
        self.form.addRow("Probing direction", self.direction)
        self.form.addRow("", _DirectionPicker(self.direction))
        self.form.addRow("Repeat", self.count)
        _connect_all(self, self.nominal, self.direction, self.count)

    def _take_position(self) -> None:
        position = self._position()
        if position is not None:
            self.nominal.set_value(position)

    def lines(self) -> list[str]:
        direction = self.direction.value()
        if not any(direction):
            raise ValueError("the probing direction must not be zero")
        line = recipes.pt_meas_line(self.nominal.value(), direction)
        return [line] * self.count.value()


class ScanLineDialog(TaskDialog):
    """Scan a straight line."""

    def __init__(self, position: PositionProvider, parent: QWidget | None = None) -> None:
        super().__init__("Scan a line", "line", parent)
        self._position = position
        self.start = VectorEdit()
        self.end = VectorEdit()
        self.end.set_value((50.0, 0.0, 0.0))
        self.direction = VectorEdit("", "IJK")
        self.direction.set_value((0.0, 0.0, 1.0))
        self.step = _spin(0.001, 1000.0, 2.0)
        current = QPushButton(load_icon("goto"), "Start at the current position")
        current.clicked.connect(self._take_position)
        self.form.addRow("Start", self.start)
        self.form.addRow("", current)
        self.form.addRow("End", self.end)
        self.form.addRow("Probing direction", self.direction)
        self.form.addRow("", _DirectionPicker(self.direction))
        self.form.addRow("Point spacing", self.step)
        _connect_all(self, self.start, self.end, self.direction, self.step)

    def _take_position(self) -> None:
        position = self._position()
        if position is not None:
            self.start.set_value(position)

    def lines(self) -> list[str]:
        if self.start.value() == self.end.value():
            raise ValueError("start and end are the same point")
        return recipes.scan_line_lines(
            self.start.value(), self.end.value(), self.direction.value(), self.step.value()
        )


class ScanArcDialog(TaskDialog):
    """Scan a circle, an arc or a helix."""

    def __init__(self, position: PositionProvider, parent: QWidget | None = None) -> None:
        super().__init__("Scan a circle or helix", "circle", parent)
        self._position = position
        self.center = VectorEdit()
        self.start = VectorEdit()
        self.start.set_value((20.0, 0.0, 0.0))
        self.normal = VectorEdit("", "IJK")
        self.normal.set_value((0.0, 0.0, 1.0))
        self.delta = _spin(-3600.0, 3600.0, 360.0)
        self.delta.setSuffix(" °")
        self.surface = _spin(-90.0, 90.0, 0.0)
        self.surface.setSuffix(" °")
        self.step = _spin(0.001, 360.0, 5.0)
        self.step.setSuffix(" °")
        self.pitch = _spin(-1000.0, 1000.0, 0.0)
        self.pitch.setSpecialValueText("none (flat circle)")
        self.pitch.setMinimum(-1000.0)
        self.form.addRow("Centre", self.center)
        self.form.addRow("Start point", self.start)
        self.form.addRow("Axis (plane normal)", self.normal)
        self.form.addRow("", _DirectionPicker(self.normal))
        self.form.addRow("Angle to scan", self.delta)
        self.form.addRow("Surface angle", self.surface)
        self.form.addRow("Angular spacing", self.step)
        self.form.addRow("Helix pitch", self.pitch)
        _connect_all(
            self, self.center, self.start, self.normal, self.delta, self.surface, self.step
        )
        self.pitch.valueChanged.connect(lambda _=0.0: self.refresh())

    def lines(self) -> list[str]:
        if self.center.value() == self.start.value():
            raise ValueError("the start point is the centre")
        return recipes.scan_circle_lines(
            self.center.value(),
            self.start.value(),
            self.normal.value(),
            self.delta.value(),
            self.surface.value(),
            self.step.value(),
            pitch=self.pitch.value(),
        )


class ToolDialog(TaskDialog):
    """Pick a tool from the machine's list and change to it."""

    refresh_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Tools", "tool", parent)
        self.tools = QListWidget()
        self.tools.setMinimumHeight(140)
        self.tools.currentTextChanged.connect(lambda _: self.refresh())
        reload = QPushButton(load_icon("refresh"), "Ask the machine for its tools")
        reload.clicked.connect(self.refresh_requested.emit)
        self.change = QRadioButton("Change tool (fetches it from the rack)")
        self.change.setChecked(True)
        self.select = QRadioButton("Select tool (already mounted)")
        group = QButtonGroup(self)
        group.addButton(self.change)
        group.addButton(self.select)
        self.change.toggled.connect(lambda _: self.refresh())
        self.form.addRow(reload)
        self.form.addRow(self.tools)
        self.form.addRow(self.change)
        self.form.addRow(self.select)

    def set_tools(self, names: list[str]) -> None:
        self.tools.clear()
        self.tools.addItems(names)
        if names:
            self.tools.setCurrentRow(0)

    def lines(self) -> list[str]:
        item = self.tools.currentItem()
        if item is None:
            raise ValueError("choose a tool (ask the machine for its tools first)")
        name = item.text()
        return [
            recipes.change_tool_line(name)
            if self.change.isChecked()
            else recipes.set_tool_line(name)
        ]


class CoordSystemDialog(TaskDialog):
    """Coordinate systems: choose the one to work in, place one, save and load named ones."""

    #: The client asks the machine for the named systems / the placement of ``csy``.
    names_requested = Signal()
    placement_requested = Signal(str)

    ACTIONS = (
        ("select", "Work in a coordinate system"),
        ("place", "Place a coordinate system"),
        ("save", "Save the active part system as ..."),
        ("load", "Load a saved system"),
        ("delete", "Delete a saved system"),
    )

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Coordinate systems", "csy", parent)
        self.action = QComboBox()
        for key, label in self.ACTIONS:
            self.action.addItem(label, key)
        self.csy = QComboBox()
        self.csy.addItems(CSY_CHAIN)
        self.target = QComboBox()
        self.target.addItems(LIVE_TRANSFORM_NAMES)
        self.target.currentTextChanged.connect(lambda _: self._placement_changed())
        self.offset = VectorEdit()
        self.theta = _spin(0.0, 180.0, 0.0)
        self.psi = _spin(-360.0, 360.0, 0.0)
        self.phi = _spin(-360.0, 360.0, 0.0)
        for box in (self.theta, self.psi, self.phi):
            box.setSuffix(" °")
        self.read = QPushButton(load_icon("refresh"), "Read it from the machine")
        self.read.clicked.connect(lambda: self.placement_requested.emit(self.target.currentText()))
        self.saved = QComboBox()
        self.saved.setEditable(True)
        self.reload = QPushButton(load_icon("refresh"), "Ask for the saved systems")
        self.reload.clicked.connect(self.names_requested.emit)
        self.form.addRow("Action", self.action)
        self.form.addRow("Work in", self.csy)
        self.form.addRow("Place", self.target)
        self.form.addRow("Offset X, Y, Z", self.offset)
        self.form.addRow("Theta", self.theta)
        self.form.addRow("Psi", self.psi)
        self.form.addRow("Phi", self.phi)
        self.form.addRow(self.read)
        self.form.addRow("Name", self.saved)
        self.form.addRow(self.reload)
        self.action.currentIndexChanged.connect(lambda _: self._action_changed())
        _connect_all(
            self, self.action, self.csy, self.target, self.offset, self.theta, self.psi, self.phi
        )
        self.saved.editTextChanged.connect(lambda _: self.refresh())
        self._action_changed()

    def _action_changed(self) -> None:
        key = self.action.currentData()
        rows = {
            self.csy: key == "select",
            self.target: key == "place",
            self.offset: key == "place",
            self.theta: key == "place",
            self.psi: key == "place",
            self.phi: key == "place",
            self.read: key == "place",
            self.saved: key in ("save", "load", "delete"),
            self.reload: key in ("load", "delete"),
        }
        for widget, shown in rows.items():
            widget.setVisible(shown)
            label = self.form.labelForField(widget)
            if label is not None:
                label.setVisible(shown)
        self.refresh()

    def _placement_changed(self) -> None:
        self.refresh()

    def set_names(self, names: list[str]) -> None:
        """Show the saved systems the machine listed."""
        self.saved.clear()
        self.saved.addItems(names)

    def set_placement(self, offset: Vec3, theta: float, psi: float, phi: float) -> None:
        """Show the placement the machine reported for the chosen system."""
        self.offset.set_value(offset)
        self.theta.setValue(theta)
        self.psi.setValue(psi)
        self.phi.setValue(phi)

    def lines(self) -> list[str]:
        key = self.action.currentData()
        if key == "select":
            return [recipes.set_coord_system_line(self.csy.currentText())]
        if key == "place":
            return [
                recipes.set_csy_transformation_line(
                    self.target.currentText(),
                    self.offset.value(),
                    self.theta.value(),
                    self.psi.value(),
                    self.phi.value(),
                )
            ]
        name = self.saved.currentText().strip()
        if not name:
            raise ValueError("give the name of the saved system")
        command = {
            "save": "SaveActiveCoordSystem",
            "load": "LoadCoordSystem",
            "delete": "DeleteCoordSystem",
        }[key]
        return [recipes.named_csy_line(command, name)]


class SpeedDialog(TaskDialog):
    """Speed and acceleration of moves, point measurements and scans."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Speeds", "speed", parent)
        self.block = QComboBox()
        for key, label in recipes.SPEED_PARAMETERS.items():
            self.block.addItem(label, key)
        self.speed = _spin(0.001, 10000.0, 100.0)
        self.speed.setSuffix(" mm/s")
        self.accel = _spin(0.001, 100000.0, 500.0)
        self.accel.setSuffix(" mm/s²")
        self.set_speed = QCheckBox("Speed")
        self.set_speed.setChecked(True)
        self.set_accel = QCheckBox("Acceleration")
        self.form.addRow("For", self.block)
        self.form.addRow(self.set_speed, self.speed)
        self.form.addRow(self.set_accel, self.accel)
        _connect_all(self, self.block, self.speed, self.accel, self.set_speed, self.set_accel)

    def lines(self) -> list[str]:
        lines = recipes.speed_lines(
            self.block.currentData(),
            self.speed.value() if self.set_speed.isChecked() else None,
            self.accel.value() if self.set_accel.isChecked() else None,
        )
        if not lines:
            raise ValueError("choose speed, acceleration or both")
        return lines


class OpticalDialog(TaskDialog):
    """Acquire points with the active tool: ``DataAcquire`` and the transfer."""

    acquire_requested = Signal(dict)
    info_requested = Signal()

    def __init__(self, position: PositionProvider, parent: QWidget | None = None) -> None:
        super().__init__("Optical sensor", "optical", parent)
        self._position = position
        self.info = QLabel("Sensor not read yet")
        self.info.setWordWrap(True)
        read = QPushButton(load_icon("refresh"), "Read sensor info")
        read.clicked.connect(self.info_requested.emit)
        self.kind = QComboBox()
        self.kind.addItems(optical.ACQUISITION_TYPES)
        self.kind.setCurrentText("Sweep")
        self.start = VectorEdit()
        self.end = VectorEdit()
        self.end.set_value((50.0, 0.0, 0.0))
        self.count = QSpinBox()
        self.count.setRange(1, 10000)
        self.count.setValue(2)
        self.count.setSuffix(" positions")
        self.step = _spin(0.0, 1000.0, 1.0)
        self.step.setSpecialValueText("not set")
        self.axis = VectorEdit("", "IJK")
        self.axis.set_value((0.0, 0.0, 1.0))
        self.format = QComboBox()
        self.format.addItems(("double", "float"))
        self.name = QLineEdit("Acq1")
        current = QPushButton(load_icon("goto"), "Start at the current position")
        current.clicked.connect(self._take_position)
        self.form.addRow(self.info)
        self.form.addRow(read)
        self.form.addRow("Name", self.name)
        self.form.addRow("Type", self.kind)
        self.form.addRow("From", self.start)
        self.form.addRow("", current)
        self.form.addRow("To", self.end)
        self.form.addRow("Positions", self.count)
        self.form.addRow("Largest step", self.step)
        self.form.addRow("Sensor axis (away from the surface)", self.axis)
        self.form.addRow("", _DirectionPicker(self.axis))
        self.form.addRow("Sample format", self.format)
        _connect_all(
            self,
            self.kind,
            self.start,
            self.end,
            self.count,
            self.step,
            self.axis,
            self.format,
            self.name,
        )
        self.run_button.setText("Acquire")

    def _take_position(self) -> None:
        position = self._position()
        if position is not None:
            self.start.set_value(position)

    def set_info(self, text: str) -> None:
        self.info.setText(text)

    def options(self) -> dict[str, object]:
        """Return the arguments of :func:`pyippdme.client.optical.acquire`."""
        name = self.name.text().strip()
        if not name:
            raise ValueError("give the acquisition a name")
        kind = self.kind.currentText()
        axis = self.axis.value()
        if not any(axis):
            raise ValueError("the sensor axis must not be zero")
        count = 1 if kind == "SingleShot" and not any(self.end.value()) else self.count.value()
        if kind == "SingleShot":
            points = optical.path_points(self.start.value(), self.start.value(), 1, primary=axis)
        else:
            if count < 2:
                raise ValueError("a multi-shot or sweep needs at least two positions")
            points = optical.path_points(self.start.value(), self.end.value(), count, primary=axis)
        return {
            "name": name,
            "acquisition_type": kind,
            "points": points,
            "step_width": self.step.value() or None,
            "data_format": self.format.currentText(),
        }

    def lines(self) -> list[str]:
        """Summarise for the preview; the acquisition itself is not a command line."""
        options = self.options()
        count = len(options["points"])  # type: ignore[arg-type]
        return [f"DataAcquire {options['name']}: {options['acquisition_type']}, {count} positions"]

    def _run(self) -> None:
        self.acquire_requested.emit(self.options())
