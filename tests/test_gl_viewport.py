# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""The OpenGL renderer of the 3D view (skipped where no OpenGL context can be made)."""

from __future__ import annotations

import os
import time
from collections.abc import Iterator

import numpy as np
import pytest

pytest.importorskip("OCP")
pytest.importorskip("PySide6.QtOpenGLWidgets", exc_type=ImportError)

from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication

from pyippdme.gui.glcanvas import perspective
from pyippdme.gui.viewport import Viewport
from pyippdme.twin import DigitalTwin, demo_sample


def _pump(seconds: float = 0.3) -> None:
    app = QApplication.instance()
    assert app is not None
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.005)


@pytest.fixture
def viewport() -> Iterator[Viewport]:
    app = QApplication.instance() or QApplication([])
    twin = DigitalTwin(time_scale=0.0)
    twin.place_sample(demo_sample())
    view = Viewport(twin, renderer="gl")
    view.resize(640, 480)
    view.show()
    app.processEvents()
    _pump()
    yield view
    view.shutdown()
    view.close()


def test_the_perspective_matches_the_projection_of_the_software_view() -> None:
    m = perspective(np.radians(35.0), 4 / 3, 1.0, 1000.0)
    f = 1 / np.tan(np.radians(35.0) / 2)
    point = m @ np.array([30.0, 20.0, -100.0, 1.0])
    assert point[0] / point[3] == pytest.approx(f / (4 / 3) * 30.0 / 100.0)
    assert point[1] / point[3] == pytest.approx(f * 20.0 / 100.0)


def test_auto_uses_software_on_a_headless_platform() -> None:
    app = QApplication.instance() or QApplication([])
    twin = DigitalTwin(time_scale=0.0)
    view = Viewport(twin, renderer="auto")
    expected = "software" if QGuiApplication.platformName() in ("offscreen", "minimal") else "gl"
    assert view.renderer == expected
    with pytest.raises(ValueError, match="unknown renderer"):
        view.set_renderer("vulkan")
    app.processEvents()


@pytest.mark.skipif(
    os.environ.get("QT_QPA_PLATFORM") in (None, "", "offscreen", "minimal"),
    reason="needs a platform with OpenGL (for example xvfb with QT_QPA_PLATFORM=xcb)",
)
def test_the_gl_renderer_draws_the_scene(viewport: Viewport) -> None:
    if viewport.renderer != "gl":
        pytest.skip("no OpenGL context on this machine")
    image = viewport.render_to_image(480, 360)
    assert not image.isNull()
    colors = {image.pixel(x, y) for x in range(0, 480, 12) for y in range(0, 360, 12)}
    assert len(colors) > 8  # background, the machine, the sample, the overlay
    # The window changes while the machine and the camera move; no triangle is re-projected.
    viewport.set_view("top")
    viewport.view._pos = (100.0, 100.0, 50.0)  # type: ignore[attr-defined]
    _pump()
    moved = viewport.render_to_image(480, 360)
    assert moved != image


@pytest.mark.skipif(
    os.environ.get("QT_QPA_PLATFORM") in (None, "", "offscreen", "minimal"),
    reason="needs a platform with OpenGL",
)
def test_switching_the_renderer_back_and_forth(viewport: Viewport) -> None:
    if viewport.renderer != "gl":
        pytest.skip("no OpenGL context on this machine")
    viewport.set_renderer("software")
    assert viewport.renderer == "software"
    assert not viewport.render_to_image(320, 240).isNull()
    viewport.set_renderer("gl")
    _pump()
    assert viewport.renderer == "gl"
    assert not viewport.render_to_image(320, 240).isNull()
