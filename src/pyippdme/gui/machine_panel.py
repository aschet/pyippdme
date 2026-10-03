# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""The machine: which one, how fast it runs, and what errors it makes."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from pyippdme.gui.widgets import icon_button, spin
from pyippdme.twin import DigitalTwin, MachineModel
from pyippdme.twin.cad import CadError
from pyippdme.twin.library import ROTARY_TABLES, machine_names
from pyippdme.twin.spec import PRESETS, RACK_SIDES, MachineSpec

__all__ = ["MachinePanel"]

TIME_SCALES = (("Instant", 0.0), ("Real time", 1.0), ("2x", 2.0), ("5x", 5.0), ("20x", 20.0))


class MachinePanel(QWidget):
    """Choose the machine from the library or from files; set playback, noise and temperature."""

    def __init__(self, twin: DigitalTwin, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._syncing = False
        self.twin = twin
        #: Called after the machine was replaced.
        self.on_machine_changed: list[Callable[[], None]] = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        form = QFormLayout()
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(machine_names())
        self.preset_combo.setToolTip("Machines of the library: travel, speed and accuracy")
        self.preset_combo.currentTextChanged.connect(self._preset_changed)
        self.rotary_combo = QComboBox()
        self.rotary_combo.addItem("no rotary table")
        self.rotary_combo.addItems(list(ROTARY_TABLES))
        self.rotary_combo.currentTextChanged.connect(self._rotary_changed)
        self.table_combo = QComboBox()
        self.table_combo.addItem("fixed (bridge carries Y)", "fixed")
        self.table_combo.addItem("moves in Y (fixed bridge)", "moving-y")
        self.table_combo.setToolTip(
            "A moving table carries the rack, rotary table, fixtures and parts with it"
        )
        self.rack_combo = QComboBox()
        for side in RACK_SIDES:
            self.rack_combo.addItem(side, side)
        self.rack_combo.setToolTip("The table edge the tool rack stands at")
        self.rack_inset = spin(-500, 500, 45.0, 5.0, 0, " mm")
        self.rack_inset.setToolTip("Distance of the rack ports from that edge of the travel range")
        self.time_combo = QComboBox()
        for label, _ in TIME_SCALES:
            self.time_combo.addItem(label)
        self.time_combo.setCurrentIndex(1)
        for combo in (self.table_combo, self.rack_combo):
            combo.currentIndexChanged.connect(lambda _=0: self._layout_changed())
        self.rack_inset.valueChanged.connect(lambda _=0: self._layout_changed())
        self.time_combo.setToolTip("How fast the simulated machine runs compared to a real one")
        self.time_combo.currentIndexChanged.connect(
            lambda i: setattr(self.twin, "time_scale", TIME_SCALES[i][1])
        )
        self.override = QSlider(Qt.Orientation.Horizontal)
        self.override.setRange(1, 100)
        self.override.setValue(100)
        self.override.setToolTip("Speed override, like the dial of a real controller")
        self.override.valueChanged.connect(lambda v: setattr(self.twin, "speed_override", v / 100))
        self.noise_check = QCheckBox("Measuring errors (MPE)")
        self.noise_check.setChecked(True)
        self.noise_check.setToolTip("Add the machine's and the probe's measuring errors to points")
        self.noise_check.toggled.connect(lambda on: setattr(self.twin, "noise_enabled", on))
        self.temperature = spin(0, 60, 20.0, 0.5, 1, " °C")
        self.temperature.setToolTip("Temperature of the part; steel expands from 20 °C")
        self.temperature.valueChanged.connect(lambda v: setattr(self.twin, "temperature", v))
        form.addRow("Machine", self.preset_combo)
        form.addRow("Rotary table", self.rotary_combo)
        form.addRow("Table", self.table_combo)
        form.addRow("Tool rack at", self.rack_combo)
        form.addRow("Rack inset", self.rack_inset)
        form.addRow("Motion", self.time_combo)
        form.addRow("Speed override", self.override)
        form.addRow(self.noise_check)
        form.addRow("Part temperature", self.temperature)
        layout.addLayout(form)
        self.info = QLabel()
        self.info.setWordWrap(True)
        self.info.setStyleSheet("color: gray;")
        layout.addWidget(self.info)
        row = QHBoxLayout()
        load = icon_button("open", "Load…", "Load a machine.toml or a STEP assembly")
        load.clicked.connect(self.load_machine)
        export = icon_button("save", "Export…", "Write the machine as machine.toml and STEP files")
        export.clicked.connect(self.export_machine)
        row.addWidget(load)
        row.addWidget(export)
        layout.addLayout(row)
        layout.addStretch(1)
        self.refresh()

    # -- machine ----------------------------------------------------------------------------

    def _spec_for_choice(self, *, preset: bool = False) -> MachineSpec:
        spec = PRESETS[self.preset_combo.currentText()]
        if not preset:
            spec = spec.with_(
                table_kind=self.table_combo.currentData(),
                rack_side=self.rack_combo.currentData(),
                rack_inset=self.rack_inset.value(),
            )
        table = self.rotary_combo.currentText()
        if table in ROTARY_TABLES:
            spec = spec.with_(
                rotary_origin=(spec.travel[0] / 2, spec.travel[1] / 2, spec.table_top_z),
                **ROTARY_TABLES[table],
            )
        return spec

    def _apply_choice(self, *, preset: bool = False) -> None:
        self.twin.set_machine(MachineModel.default(self._spec_for_choice(preset=preset)))
        self.refresh()
        for callback in self.on_machine_changed:
            callback()

    def _preset_changed(self, name: str) -> None:
        if name in PRESETS:
            self._apply_choice(preset=True)

    def _layout_changed(self) -> None:
        if not self._syncing:
            self._apply_choice()

    def _rotary_changed(self, _name: str) -> None:
        self._apply_choice()

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
        self.refresh()
        for callback in self.on_machine_changed:
            callback()

    def export_machine(self) -> str | None:
        directory = QFileDialog.getExistingDirectory(self, "Export machine to directory")
        if not directory:
            return None
        return str(self.twin.machine.export(directory))

    def refresh(self) -> None:
        """Show the numbers of the machine in the twin."""
        s = self.twin.machine.spec
        for key, preset in PRESETS.items():
            if preset.name == s.name:
                self.preset_combo.blockSignals(True)
                self.preset_combo.setCurrentText(key)
                self.preset_combo.blockSignals(False)
        self._syncing = True
        self.table_combo.setCurrentIndex(self.table_combo.findData(s.table_kind))
        self.rack_combo.setCurrentIndex(self.rack_combo.findData(s.rack_side))
        self.rack_inset.setValue(s.rack_inset)
        self._syncing = False
        has_table = s.rotary_origin is not None
        self.rotary_combo.blockSignals(True)
        if not has_table:
            self.rotary_combo.setCurrentIndex(0)
        self.rotary_combo.blockSignals(False)
        derived = f"\nEstimated from CAD: {', '.join(s.derived)}" if s.derived else ""
        self.info.setText(
            f"{s.name}\nTravel {s.travel[0]:g} x {s.travel[1]:g} x {s.travel[2]:g} mm\n"
            f"{s.max_speed:g} mm/s, {s.acceleration:g} mm/s²\n"
            f"MPE_E = {s.accuracy.a_um:g} + L/{s.accuracy.k:g} µm, "
            f"MPE_P = {s.accuracy.probing_um:g} µm"
            f"{derived}"
        )
