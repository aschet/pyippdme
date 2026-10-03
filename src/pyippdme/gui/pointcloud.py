# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""A small 3D view of a point cloud: orbit, pan and zoom, coloured by height."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPen, QWheelEvent
from PySide6.QtWidgets import QWidget

_MAX_DRAWN = 60_000
_RAMP = np.array(
    [[47, 111, 176], [127, 184, 216], [250, 220, 120], [232, 131, 58], [190, 60, 40]], dtype=float
)


def _colors(values: npt.NDArray[np.float64]) -> npt.NDArray[np.uint8]:
    lo, hi = float(values.min()), float(values.max())
    t = np.zeros_like(values) if hi - lo < 1e-12 else (values - lo) / (hi - lo)
    position = t * (len(_RAMP) - 1)
    index = np.minimum(position.astype(int), len(_RAMP) - 2)
    frac = (position - index)[:, None]
    return np.asarray(_RAMP[index] * (1 - frac) + _RAMP[index + 1] * frac, dtype=np.uint8)


class PointCloudView(QWidget):
    """Shows points of shape (n, 3); drag to orbit, right-drag to pan, wheel to zoom."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.points: npt.NDArray[np.float64] = np.empty((0, 3))
        self.azimuth = -50.0
        self.elevation = 30.0
        self.zoom = 1.0
        self.pan = np.zeros(2)
        self._last: QPointF | None = None
        self.setMinimumSize(240, 200)

    def set_points(self, points: npt.ArrayLike) -> None:
        array = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        self.points = array[np.all(np.isfinite(array), axis=1)]
        self.update()

    def add_points(self, points: npt.ArrayLike) -> None:
        extra = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        self.set_points(np.concatenate([self.points, extra]))

    def clear(self) -> None:
        self.set_points(np.empty((0, 3)))

    def fit(self) -> None:
        self.zoom = 1.0
        self.pan = np.zeros(2)
        self.update()

    # -- projection -------------------------------------------------------------------------

    def _rotation(self) -> npt.NDArray[np.float64]:
        a, e = math.radians(self.azimuth), math.radians(self.elevation)
        rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
        rx = np.array([[1, 0, 0], [0, math.cos(e), -math.sin(e)], [0, math.sin(e), math.cos(e)]])
        return np.asarray(rx.T @ rz.T)

    def project(self, points: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.float64], float]:
        """Screen coordinates (x right, y down, in pixels) and the view's depth axis."""
        if len(points) == 0:
            return np.empty((0, 2)), 1.0
        centre = (self.points.min(axis=0) + self.points.max(axis=0)) / 2 if len(self.points) else 0
        extent = (
            float(np.max(self.points.max(axis=0) - self.points.min(axis=0)))
            if len(self.points)
            else 1.0
        )
        scale = 0.8 * min(self.width(), self.height()) / max(extent, 1e-9) * self.zoom
        view = (points - centre) @ self._rotation().T
        x = self.width() / 2 + self.pan[0] + view[:, 0] * scale
        y = self.height() / 2 + self.pan[1] - view[:, 2] * scale
        return np.column_stack([x, y]), scale

    def render_image(self) -> QImage:
        image = QImage(self.size(), QImage.Format.Format_ARGB32)
        image.fill(self.palette().window().color())
        self._draw(QPainter(image))
        return image

    def paintEvent(self, _event: object) -> None:  # noqa: N802
        self._draw(QPainter(self))

    def _draw(self, p: QPainter) -> None:
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        if len(self.points) == 0:
            p.setPen(self.palette().placeholderText().color())
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No points yet")
            p.end()
            return
        pts = self.points
        if len(pts) > _MAX_DRAWN:
            pts = pts[:: math.ceil(len(pts) / _MAX_DRAWN)]
        screen, _ = self.project(pts)
        depth = (pts @ self._rotation().T)[:, 1]
        order = np.argsort(-depth)  # far points first
        colors = _colors(pts[:, 2])
        size = 2.0 if len(pts) > 5000 else 3.0
        p.setPen(Qt.PenStyle.NoPen)
        for i in order:
            r, g, b = colors[i]
            p.setBrush(QColor(int(r), int(g), int(b)))
            p.drawEllipse(QPointF(screen[i, 0], screen[i, 1]), size / 2, size / 2)
        self._draw_axes(p)
        lo, hi = self.points.min(axis=0), self.points.max(axis=0)
        p.setPen(self.palette().text().color())
        p.drawText(
            8,
            self.height() - 8,
            f"{len(self.points)} points   X {lo[0]:.2f}…{hi[0]:.2f}   Y {lo[1]:.2f}…{hi[1]:.2f}"
            f"   Z {lo[2]:.2f}…{hi[2]:.2f}",
        )
        p.end()

    def _draw_axes(self, p: QPainter) -> None:
        origin = QPointF(34, 34)
        rotation = self._rotation()
        for axis, color in zip(range(3), ("#e74c3c", "#27ae60", "#2f80ed"), strict=True):
            v = rotation @ np.eye(3)[axis]
            p.setPen(QPen(QColor(color), 2))
            p.drawLine(origin, QPointF(origin.x() + v[0] * 22, origin.y() - v[2] * 22))
            p.drawText(QPointF(origin.x() + v[0] * 26 - 3, origin.y() - v[2] * 26 + 4), "XYZ"[axis])

    # -- interaction ------------------------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        self._last = event.position()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._last is None:
            return
        delta = event.position() - self._last
        self._last = event.position()
        if event.buttons() & Qt.MouseButton.RightButton:
            self.pan += (delta.x(), delta.y())
        else:
            self.azimuth -= delta.x() * 0.5
            self.elevation = max(-89.0, min(89.0, self.elevation + delta.y() * 0.5))
        self.update()

    def mouseReleaseEvent(self, _event: QMouseEvent) -> None:  # noqa: N802
        self._last = None

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        self.zoom = max(0.05, min(50.0, self.zoom * (1.0015 ** event.angleDelta().y())))
        self.update()
