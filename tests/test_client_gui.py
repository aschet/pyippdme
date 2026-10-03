# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""The command client window against a virtual CMM started inside it."""

from __future__ import annotations

import os
import time
from collections.abc import Iterator

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)

from PySide6.QtWidgets import QApplication

from pyippdme.client import commandform, recipes
from pyippdme.gui.client_window import ClientWindow
from pyippdme.gui.icons import icon_names, load_icon
from pyippdme.protocol.parser import parse_method


@pytest.fixture
def window() -> Iterator[ClientWindow]:
    app = QApplication.instance() or QApplication([])
    w = ClientWindow()
    w.show()
    app.processEvents()
    yield w
    w.close()


def _wait(condition, timeout: float = 10.0) -> None:  # type: ignore[no-untyped-def]
    app = QApplication.instance()
    assert app is not None
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        app.processEvents()
        if condition():
            return
        time.sleep(0.01)
    raise AssertionError("timed out")


def _idle(window: ClientWindow) -> bool:
    return not window._polling


def test_every_icon_loads() -> None:
    QApplication.instance() or QApplication([])
    assert len(icon_names()) >= 20
    assert all(not load_icon(name).pixmap(24, 24).isNull() for name in icon_names())


def test_recipes_and_forms_build_valid_commands() -> None:
    lines = [
        recipes.goto_line(1, 2, r=5, sync=True),
        recipes.pt_meas_line((1, 2, 3), (0, 0, 1)),
        *recipes.scan_line_lines((0, 0, 0), (5, 0, 0), (0, 0, 1), 1.0),
        *recipes.scan_circle_lines((0, 0, 0), (5, 0, 0), (0, 0, 1), 90, 0, 5, pitch=1),
        *recipes.speed_lines("ScanPar", 10, 50),
        *recipes.status_lines(),
        commandform.build_command_line("GoTo", {"Positions": "X(1),Y(2)"}),
    ]
    for line in lines:
        parse_method(line)
    with pytest.raises(ValueError, match="required"):
        commandform.build_command_line("GetErrorInfo", {})


def test_dialogs_measure_and_the_status_bar_follows(window: ClientWindow) -> None:
    window.connect_virtual()
    _wait(lambda: window.host.connected and "Connected" in window.connection_label.text())
    window.send("Home()")
    _wait(lambda: "Homed" in window.homed_label.text() and "Not" not in window.homed_label.text())
    window.send("EnableUser()")

    move = window.open_dialog("move")
    move._axes["X"][1].setValue(40.0)  # type: ignore[attr-defined]
    move._axes["Y"][1].setValue(30.0)  # type: ignore[attr-defined]
    move._axes["Z"][1].setValue(20.0)  # type: ignore[attr-defined]
    move.run_button.click()
    _wait(lambda: window.position is not None and abs(window.position[0] - 40.0) < 1e-6)
    assert "40.000" in window.position_label.text()

    point = window.open_dialog("point")
    point.nominal.set_value((40.0, 30.0, 10.0))  # type: ignore[attr-defined]
    point.run_button.click()
    _wait(lambda: window.points.rowCount() >= 1)
    assert "X" in {window.response.item(r, 0).text() for r in range(window.response.rowCount())}  # type: ignore[union-attr]

    line = window.open_dialog("line")
    line.run_button.click()
    _wait(lambda: window.points.rowCount() > 5)

    tools = window.open_dialog("tool")
    _wait(lambda: tools.tools.count() > 0)  # type: ignore[attr-defined]
    _wait(lambda: _idle(window))


def test_errors_show_in_the_status_bar(window: ClientWindow) -> None:
    window.connect_virtual()
    _wait(lambda: window.host.connected and window.homed_label.text() != "")  # session started
    window.send("NoSuchCommand()")
    _wait(lambda: "Error" in window.error_label.text())
    assert "Error" in window.log.toPlainText()
    window.send("GetErrStatusE()")
    _wait(lambda: not window.error_label.text())


def test_optical_dialog_acquires_points_into_the_cloud_view(window: ClientWindow) -> None:
    window.connect_virtual()
    _wait(lambda: window.host.connected and window.homed_label.text() != "")
    dialog = window.open_dialog("optical")
    dialog.info_requested.emit()  # type: ignore[attr-defined]
    _wait(lambda: "raw data" in dialog.info.text())  # type: ignore[attr-defined]
    assert "SocBin" in dialog.info.text()  # type: ignore[attr-defined]
    dialog.start.set_value((0.0, 0.0, 0.0))  # type: ignore[attr-defined]
    dialog.end.set_value((20.0, 0.0, 0.0))  # type: ignore[attr-defined]
    dialog.run_button.click()
    _wait(lambda: len(window.cloud.points) > 10)
    assert window.cloud.points[:, 0].max() == pytest.approx(20.0, abs=0.5)
    image = window.cloud.render_image()
    assert not image.isNull()
    assert "XYZ" not in window._cloud_csv()
    assert window._cloud_csv().startswith("x,y,z")
