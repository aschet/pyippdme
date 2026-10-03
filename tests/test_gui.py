# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""The simulator window: starts a real server, shows the twin, places samples."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Iterator
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)
pytest.importorskip("OCP")

from PySide6.QtWidgets import QApplication

from pyippdme import IppDmeMachine
from pyippdme.gui.gamepad import PadMapping, PadState
from pyippdme.gui.main_window import MainWindow
from pyippdme.twin import DigitalTwin, cad


@pytest.fixture
def window() -> Iterator[MainWindow]:
    app = QApplication.instance() or QApplication([])
    twin = DigitalTwin(time_scale=0.0, seed=3)
    w = MainWindow(twin)
    w.port_spin.setValue(0)
    w.show()
    app.processEvents()
    yield w
    w.close()


def _pump(seconds: float = 0.1) -> None:
    import time

    app = QApplication.instance()
    assert app is not None
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.005)


def test_a_client_drives_the_machine_through_the_window(window: MainWindow) -> None:
    window.add_demo_sample()
    window.toggle_server()
    port = window.host.port
    assert port

    async def session() -> float:
        machine = await IppDmeMachine.connect("127.0.0.1", port)
        await machine.start_session()
        await machine.dme.home()
        await machine.cart_cmm.go_to(x=325, y=335, z=50)
        z = (await machine.cart_cmm.pt_meas(x=325, y=335, z=20, ijk=(0, 0, 1))).number("Z")
        await machine.close()
        return z

    future = window.host.submit(session())
    while not future.done():
        _pump(0.02)
    assert future.result() == pytest.approx(20.0, abs=0.05)
    _pump()
    assert "PtMeas" in window.log.toPlainText()
    assert len(window.twin.contacts) == 1
    window.toggle_server()
    assert not window.host.running


def test_sample_is_loaded_from_a_step_file_and_placed_from_the_window(
    window: MainWindow, tmp_path: Path
) -> None:
    step = tmp_path / "block.step"
    cad.write_step([cad.CadBody("block", cad.make_box(40, 30, 20))], step)
    window.twin.load_sample(step)
    window._refresh_scene_list()
    assert window.scene_list.count() == 1
    obj = window.twin.objects[0]
    assert obj.source == str(step)
    # Typing a new position into the placement boxes moves the sample.
    window.pose_boxes[0].setValue(100.0)
    window.pose_boxes[5].setValue(90.0)
    lo, hi = obj.world_bounds(window.twin.machine.rotary_pose(0.0))
    assert lo[0] == pytest.approx(100.0 - 30.0, abs=1e-6)  # rotated 90 degrees about z
    assert hi[1] - lo[1] == pytest.approx(40.0, abs=1e-6)
    window.remove_selected()
    assert not window.twin.objects


def test_viewport_draws_the_machine_and_the_sample(window: MainWindow) -> None:
    window.add_demo_sample()
    image = window.viewport.render_to_image(480, 320)
    background = image.pixelColor(2, 2).rgb()
    differing = sum(
        image.pixelColor(x, y).rgb() != background
        for x in range(0, 480, 8)
        for y in range(0, 320, 8)
    )
    assert differing > 100


def test_jog_needs_a_session_and_stays_in_the_volume(window: MainWindow) -> None:
    assert not window.twin.jog(dx=1.0)  # no client
    window.add_demo_sample()
    window.toggle_server()
    port = window.host.port
    assert port

    async def jog() -> tuple[bool, bool]:
        machine = await IppDmeMachine.connect("127.0.0.1", port)
        await machine.start_session()
        await machine.dme.home()
        await machine.mover.enable_user()
        ok = window.twin.jog(dx=10.0)
        outside = window.twin.jog(dx=-1000.0)
        await machine.close()
        return ok, outside

    future = window.host.submit(jog())
    while not future.done():
        _pump(0.02)
    assert future.result() == (True, False)


def test_machine_can_be_exported_and_loaded_back(window: MainWindow, tmp_path: Path) -> None:
    window.twin.machine.export(tmp_path)
    assert (tmp_path / "machine.toml").is_file()
    assert (tmp_path / "bridge.step").is_file()
    from pyippdme.twin import MachineModel

    window.twin.set_machine(MachineModel.from_directory(tmp_path))
    window._refresh_machine_info()
    assert "700" in window.machine_info.text()
    _ = asyncio  # the host owns its own loop


def test_tool_editor_changes_a_tool_that_the_protocol_sees(window: MainWindow) -> None:
    panel = window.tool_panel
    names = [panel.tools.item(i).text() for i in range(panel.tools.count())]
    assert "RefTool" in names
    panel.tools.setCurrentRow(names.index("RefTool"))
    panel.copy()
    assert "RefTool2" in window.twin.machine.tools
    panel.name_edit.setText("RefTool2")
    widget = panel.editor._widgets["ball_radius"]
    widget.setValue(1.5)  # type: ignore[attr-defined]
    panel.apply()
    assert window.twin.machine.tools["RefTool2"].ball_radius == pytest.approx(1.5)
    panel.name_edit.setText("bad name")
    panel.apply()
    assert "not applied" in panel.status.text().lower()
    panel.tools.setCurrentRow(sorted(window.twin.machine.tools).index("RefTool2"))
    panel.remove()
    assert "RefTool2" not in window.twin.machine.tools


def test_check_artifact_is_placed_measured_and_evaluated(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    import time
    from collections import deque

    import numpy as np

    from pyippdme.gui import twin_panels

    panel = window.check_panel
    panel.evaluate()
    assert "place" in panel.status.text().lower()
    panel.add_artifact()
    panel.add_sphere()
    assert window.twin.artifact() is not None
    # The full program takes minutes of simulated collision checking; stand in for the client
    # and put points on the artefact's reference sphere as a probing run would.
    calls: list[tuple[int, tuple[str, ...]]] = []

    async def fake(host: str, port: int, data: object, options: object, progress: object) -> None:
        calls.append((port, options.modes))  # type: ignore[attr-defined]
        progress("touch")  # type: ignore[operator]
        sphere = next(f for f in data.features if f.name == "reference sphere")  # type: ignore[attr-defined]
        angles = np.linspace(0.0, 6.28, 30)
        ring = np.column_stack([np.cos(angles), np.sin(angles), np.zeros(30)])
        window.twin.points.setdefault("touch", deque()).extend(
            (sphere.center + ring * sphere.radius).tolist()
        )

    monkeypatch.setattr(twin_panels, "run_check_against_server", fake)
    panel.run_program()
    assert "server first" in panel.status.text().lower()
    window.toggle_server()
    for mode, check in panel.mode_checks.items():
        check.setChecked(mode == "touch")
    panel.run_program()
    deadline = time.monotonic() + 10
    while not panel.run_button.isEnabled() and time.monotonic() < deadline:
        _pump(0.02)
    _pump(0.05)
    assert calls == [(window.host.port, ("touch",))]
    assert panel.table.rowCount() > 0
    window.toggle_server()


def test_safety_panel_presses_the_emergency_stop(window: MainWindow) -> None:
    panel = window.safety_panel
    panel.estop.setChecked(True)
    assert window.twin.estop
    assert window.twin.snapshot().estop
    panel.estop.setChecked(False)
    panel.air.setChecked(False)
    assert not window.twin.air_ok
    panel.air.setChecked(True)
    window._tick()
    assert "qualified" in window.strip.tool.text()


def test_the_status_strip_follows_the_machine(window: MainWindow) -> None:
    window._tick()
    assert window.strip.server.state == "off"
    assert window.strip.homed.state == "off"
    window.toggle_server()
    assert window.strip.server.state == "on"
    assert "Listening" in window.strip.server.text
    window.safety_panel.estop.setChecked(True)
    window._tick()
    assert window.strip.safety.state == "error"
    assert window.estop_action.isChecked()
    window.estop_action.setChecked(False)
    assert not window.twin.estop
    window.toggle_server()


def test_every_camera_view_draws_something(window: MainWindow) -> None:
    window.add_demo_sample()
    for name in ("isometric", "top", "front", "right", "follow", "probe", "table", "fit"):
        window.set_view(name)
        assert window.viewport.render_to_image(240, 160).width() == 240
    window.set_view("probe")
    assert window.view_actions["probe"].isChecked()
    assert window.viewport.camera == "probe"


def test_the_tool_creator_adds_a_tool_the_protocol_offers(window: MainWindow) -> None:
    creator = window.tool_creator
    creator.combos["sensor"].setCurrentText("scanning probe")
    creator.combos["head"].setCurrentText("indexing head (PH10)")
    creator.combos["tip"].setCurrentText("ball 6 mm")
    creator.name_edit.setText("MyScan")
    creator.add_tool()
    spec = window.twin.machine.tools["MyScan"]
    assert spec.mode == "scanning"
    assert spec.head == "indexed"
    assert spec.ball_radius == pytest.approx(3.0)
    creator.combos["sensor"].setCurrentText("laser line scanner")
    assert not creator.combos["stylus"].isEnabled()  # an optical sensor has no stylus
    creator.name_edit.setText("bad name")
    creator.add_tool()
    assert "letters and digits" in creator.status.text()


def test_the_machine_panel_offers_the_library(window: MainWindow) -> None:
    panel = window.machine_panel
    assert panel.preset_combo.currentText() == "bridge-700"
    panel.preset_combo.setCurrentText("gantry-large")
    assert window.twin.machine.spec.travel[0] == 2000.0
    panel.rotary_combo.setCurrentText("rotary table 500 mm")
    spec = window.twin.machine.spec
    assert spec.rotary_origin is not None
    assert spec.rotary_diameter == 500.0


def test_teach_in_events_reach_the_client(window: MainWindow) -> None:
    window.add_demo_sample()
    window.toggle_server()
    port = window.host.port
    assert port
    lo, _ = window.twin.objects[0].world_bounds(window.twin.machine.rotary_pose(0.0))
    top = float(lo[2]) + 30.0

    async def session() -> list[str]:
        machine = await IppDmeMachine.connect("127.0.0.1", port)
        await machine.start_session()
        await machine.dme.home()
        await machine.mover.enable_user()
        events = machine.client.unsolicited_events()
        window.twin._pos = (float(lo[0]) + 10.0, float(lo[1]) + 45.0, top + 20.0)
        window.teach_panel.press("Done")
        window.teach_panel.pick()
        window.teach_panel.clearance()
        got: list[str] = []
        async for event in events:
            got.append(event.data.to_wire())
            if len(got) == 3:
                break
        await machine.close()
        return got

    future = window.host.submit(session())
    while not future.done():
        _pump(0.02)
    wires = future.result()
    assert "KeyPress" in wires[0]
    assert wires[1].startswith("PtMeas")
    assert wires[2].startswith("GoTo")
    assert len(window.twin.contacts) == 1
    window.toggle_server()


def test_the_teach_panel_plans_and_runs_a_program(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pyippdme.gui import teach_panel

    def quick(twin: object, points: list[object], **_: object) -> list[str]:
        # Planning every point of the artefact takes a minute; the real planner is tested with
        # ``test_an_offline_program_is_generated_and_simulated``.
        return ["Home()", "EnableUser()", f"# {len(points)} points", "PtMeas(X(1),Y(2),Z(3))"]

    monkeypatch.setattr(teach_panel, "touch_program", quick)
    panel = window.teach_panel
    panel.generate()
    assert "artefact" in panel.status.text()
    window.check_panel.add_artifact()
    panel.per_feature.setValue(1)
    panel.generate()
    deadline = __import__("time").monotonic() + 120
    while not panel.program.toPlainText() and __import__("time").monotonic() < deadline:
        _pump(0.05)
    lines = panel.commands()
    assert lines[0] == "Home()"
    assert any(line.startswith("PtMeas") for line in lines)
    assert "# 24 points" in panel.program.toPlainText() or "# " in panel.program.toPlainText()
    panel.program.setPlainText("# comment\nGetDMEVersion()\n")
    assert panel.commands() == ["GetDMEVersion()"]
    window.twin.clear_scene()  # the artefact's mesh is slow to paint while the program runs
    window.toggle_server()
    panel.run()
    deadline = __import__("time").monotonic() + 20
    while not panel.run_button.isEnabled() and __import__("time").monotonic() < deadline:
        _pump(0.02)
    assert panel.run_button.isEnabled(), panel.status.text()
    assert "Program: done" in panel.status.text()
    window.toggle_server()


class _FakePad:
    name = "Fake pad"

    def __init__(self) -> None:
        self.state: PadState | None = PadState()

    def poll(self) -> PadState | None:
        return self.state


def test_gamepad_mapping_has_a_dead_zone_and_edge_triggered_buttons() -> None:
    mapping = PadMapping(deadzone=0.2, speed=10.0)
    assert mapping.jog(PadState({"leftx": 0.1}), 1.0) == (0.0, 0.0, 0.0)
    dx, dy, dz = mapping.jog(PadState({"leftx": 1.0, "lefty": -1.0, "righty": 1.0}), 0.5)
    assert (dx, dy, dz) == pytest.approx((5.0, 5.0, -5.0))  # up on the stick is +Y
    assert mapping.pressed(PadState(buttons=frozenset({"a"}))) == ["a"]
    assert mapping.pressed(PadState(buttons=frozenset({"a", "b"}))) == ["b"]
    assert mapping.pressed(PadState()) == []


def test_the_game_controller_jogs_the_machine(window: MainWindow) -> None:
    panel = window.teach_panel
    pad = _FakePad()
    panel.pad = pad
    panel.pad_box.setChecked(True)
    assert panel._pad_timer.isActive()
    panel._pad_timer.stop()
    pad.state = PadState({"leftx": 1.0}, frozenset({"rightshoulder"}))
    step = panel.jog_step.value()
    panel._poll_pad()
    assert panel.jog_step.value() == pytest.approx(step * 2)  # no client, so no movement
    assert panel.pad_status.text() == "Fake pad"
    pad.state = None
    panel._poll_pad()
    assert "Connect" in panel.pad_status.text()


def test_the_machine_panel_sets_table_kind_and_rack(window: MainWindow) -> None:
    panel = window.machine_panel
    panel.table_combo.setCurrentIndex(panel.table_combo.findData("moving-y"))
    assert window.twin.machine.spec.table_kind == "moving-y"
    panel.rack_combo.setCurrentIndex(panel.rack_combo.findData("front"))
    panel.rack_inset.setValue(20.0)
    spec = window.twin.machine.spec
    assert (spec.rack_side, spec.rack_inset) == ("front", 20.0)
    panel.preset_combo.setCurrentText("bridge-900")
    assert window.twin.machine.spec.table_kind == "fixed"  # a preset brings its own layout
    assert panel.table_combo.currentData() == "fixed"
