# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""The scene: samples and fixtures on the table, and where each one stands."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from pyippdme.gui.icons import load_icon
from pyippdme.gui.widgets import icon_button, spin, tool_button
from pyippdme.twin import DigitalTwin, demo_sample, geometry
from pyippdme.twin.cad import SUPPORTED_SUFFIXES, CadError
from pyippdme.twin.objects import SceneObject

__all__ = ["ScenePanel"]

CAD_FILTER = "CAD files (" + " ".join(f"*{s}" for s in SUPPORTED_SUFFIXES) + ");;All files (*)"
_KIND_ICONS = {"sample": "sample", "fixture": "fixture"}


class ScenePanel(QWidget):
    """A list of the objects on the table with visibility, and the placement of the selected one."""

    #: The selection or the list changed (the window may refresh other views).
    changed = Signal()

    def __init__(self, twin: DigitalTwin, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.twin = twin
        self.selected: SceneObject | None = None
        self._updating = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        add_row = QHBoxLayout()
        self.add_buttons = {
            "sample": icon_button("open", "Sample…", "Load a sample from a CAD file"),
            "demo": icon_button("sample", "Demo", "Place the built-in demo block"),
            "box": icon_button("fixture", "Box", "Add a box fixture"),
            "cylinder": icon_button("fixture", "Cylinder", "Add a cylinder fixture"),
            "fixture": icon_button("open", "Fixture…", "Load a fixture from a CAD file"),
        }
        self.add_buttons["sample"].clicked.connect(self.load_sample)
        self.add_buttons["demo"].clicked.connect(self.add_demo_sample)
        self.add_buttons["box"].clicked.connect(lambda: self.add_fixture("box"))
        self.add_buttons["cylinder"].clicked.connect(lambda: self.add_fixture("cylinder"))
        self.add_buttons["fixture"].clicked.connect(self.load_fixture)
        grid = QGridLayout()
        for i, button in enumerate(self.add_buttons.values()):
            grid.addWidget(button, i // 2, i % 2)
        add_row.addLayout(grid)
        layout.addLayout(add_row)

        self.empty = QLabel(
            "The table is empty.\nPlace a sample from a CAD file (STEP, IGES, STL, BREP), or use "
            "the demo block."
        )
        self.empty.setWordWrap(True)
        self.empty.setStyleSheet("color: gray;")
        layout.addWidget(self.empty)
        self.scene_list = QListWidget()
        self.scene_list.currentRowChanged.connect(self._select)
        self.scene_list.itemChanged.connect(self._item_changed)
        layout.addWidget(self.scene_list, 1)

        remove_row = QHBoxLayout()
        self.remove_button = tool_button("clear", "Remove the selected object")
        self.remove_button.clicked.connect(self.remove_selected)
        self.drop_button = icon_button(
            "view-table", "Rest on table", "Put the object on the table, centred"
        )
        self.drop_button.clicked.connect(self.rest_selected)
        remove_row.addWidget(self.drop_button)
        remove_row.addStretch(1)
        remove_row.addWidget(self.remove_button)
        layout.addLayout(remove_row)

        self.inspector = QGroupBox("Placement (machine coordinates)")
        pose = QGridLayout(self.inspector)
        self.pose_boxes = []
        for i, name in enumerate(("X", "Y", "Z", "RX", "RY", "RZ")):
            box = spin(-5000, 5000, 0.0, 1.0 if i < 3 else 5.0, 3, " mm" if i < 3 else " °")
            box.valueChanged.connect(self._pose_edited)
            self.pose_boxes.append(box)
            pose.addWidget(QLabel(name), i % 3, (i // 3) * 2)
            pose.addWidget(box, i % 3, (i // 3) * 2 + 1)
        self.on_rotary_check = QCheckBox("Mounted on the rotary table")
        self.on_rotary_check.toggled.connect(self._on_rotary_toggled)
        pose.addWidget(self.on_rotary_check, 3, 0, 1, 4)
        layout.addWidget(self.inspector)
        self.refresh()

    # -- adding and removing ----------------------------------------------------------------

    def _cad_dialog(self, title: str) -> str:
        path, _ = QFileDialog.getOpenFileName(self, title, "", CAD_FILTER)
        return path

    def load_sample(self) -> None:
        path = self._cad_dialog("Load sample CAD")
        if path:
            self.add(lambda: self.twin.load_sample(Path(path)))

    def load_fixture(self) -> None:
        path = self._cad_dialog("Load fixture CAD")
        if path:
            self.add(
                lambda: self.twin.place_sample(
                    SceneObject.from_file(path, "fixture"), replace=False
                )
            )

    def add_demo_sample(self) -> None:
        self.add(lambda: self.twin.place_sample(demo_sample()))

    def add_fixture(self, kind: str) -> None:
        size = (60.0, 40.0, 25.0) if kind == "box" else (20.0, 40.0, 0.0)
        s = self.twin.machine.spec
        self.add(
            lambda: self.twin.add_fixture(
                kind, size, (s.travel[0] / 2 + 100, s.travel[1] / 2, s.table_top_z)
            )
        )

    def add(self, make: Callable[[], SceneObject]) -> None:
        """Run ``make`` (which places an object) and select the result; show a CAD error."""
        try:
            obj = make()
        except (CadError, OSError) as error:
            QMessageBox.critical(self, "Cannot load the file", str(error))
            return
        self.refresh(select=obj)

    def remove_selected(self) -> None:
        if self.selected is not None:
            self.twin.remove_object(self.selected)
            self.selected = None
            self.refresh()

    def rest_selected(self) -> None:
        if self.selected is None:
            return
        self.twin.rest_on_table(self.selected)
        self.twin.set_pose(self.selected, self.selected.pose)
        self._load_pose(self.selected)

    # -- list and placement -----------------------------------------------------------------

    def refresh(self, select: SceneObject | None = None) -> None:
        """Rebuild the list from the twin; select ``select`` or the first object."""
        self.scene_list.blockSignals(True)
        self.scene_list.clear()
        for obj in self.twin.objects:
            item = QListWidgetItem(
                load_icon(_KIND_ICONS.get(obj.kind, "cube")), f"{obj.name}   [{obj.kind}]"
            )
            item.setData(Qt.ItemDataRole.UserRole, obj.id)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if obj.visible else Qt.CheckState.Unchecked)
            item.setToolTip("Tick to show the object in the 3D view and to its sensors")
            self.scene_list.addItem(item)
        self.scene_list.blockSignals(False)
        row = next((i for i, o in enumerate(self.twin.objects) if o is select), -1)
        if row < 0 and self.twin.objects:
            row = 0
        self.scene_list.setCurrentRow(row)
        self._select(row)
        empty = not self.twin.objects
        self.empty.setVisible(empty)
        self.scene_list.setVisible(not empty)
        self.changed.emit()

    def _select(self, row: int) -> None:
        self.selected = self.twin.objects[row] if 0 <= row < len(self.twin.objects) else None
        self.inspector.setEnabled(self.selected is not None)
        self.remove_button.setEnabled(self.selected is not None)
        self.drop_button.setEnabled(self.selected is not None)
        if self.selected is not None:
            self._load_pose(self.selected)

    def _load_pose(self, obj: SceneObject) -> None:
        self._updating = True
        for box, value in zip(self.pose_boxes, geometry.decompose(obj.pose), strict=True):
            box.setValue(value)
        self.on_rotary_check.setChecked(obj.on_rotary)
        self._updating = False

    def _pose_edited(self) -> None:
        if self._updating or self.selected is None:
            return
        self.twin.set_pose(self.selected, geometry.pose(*(b.value() for b in self.pose_boxes)))

    def _on_rotary_toggled(self, on: bool) -> None:
        if not self._updating and self.selected is not None:
            self.selected.on_rotary = on
            self.twin.set_pose(self.selected, self.selected.pose)

    def _item_changed(self, item: QListWidgetItem) -> None:
        row = self.scene_list.row(item)
        if self._updating or not 0 <= row < len(self.twin.objects):
            return
        obj = self.twin.objects[row]
        obj.visible = item.checkState() == Qt.CheckState.Checked
        self.twin.set_pose(obj, obj.pose)
