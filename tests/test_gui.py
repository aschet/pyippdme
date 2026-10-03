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
pytest.importorskip("PySide6.QtWidgets")
pytest.importorskip("OCP")

from PySide6.QtWidgets import QApplication

from pyippdme import IppDmeMachine
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
    assert "qualified" in window.state_label.text()
