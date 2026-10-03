# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""A 3D view of a :class:`~pyippdme.twin.view.SimulationView`, drawn with ``QPainter``.

No OpenGL: the triangles are projected with numpy, sorted back to front and
painted, so the view works everywhere (remote desktops, CI with the offscreen
platform). While the camera is dragged the triangle count is capped; at rest the
full mesh is drawn.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QPointF, Qt, QTimer
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPen, QPolygonF, QWheelEvent
from PySide6.QtWidgets import QWidget

from pyippdme.twin.geometry import Mesh
from pyippdme.twin.twin import DrawItem
from pyippdme.twin.view import SimulationView

_LIGHT = np.array([0.4, -0.5, 0.75])
_LIGHT = _LIGHT / np.linalg.norm(_LIGHT)
#: Triangles painted while the camera moves.
_INTERACTIVE_BUDGET = 6000
_FOV = math.radians(35.0)


def _look_at(eye: NDArray[np.float64], target: NDArray[np.float64]) -> NDArray[np.float64]:
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(right) < 1e-9:
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    m = np.eye(4)
    m[0, :3], m[1, :3], m[2, :3] = right, up, -forward
    m[:3, 3] = -m[:3, :3] @ eye
    return m


#: Camera presets: ``(azimuth, elevation)`` in degrees, looking at the middle of the machine.
VIEW_PRESETS: dict[str, tuple[float, float]] = {
    "isometric": (-55.0, 28.0),
    "front": (-90.0, 0.0),
    "back": (90.0, 0.0),
    "left": (180.0, 0.0),
    "right": (0.0, 0.0),
    "top": (-90.0, 89.0),
}
#: Views that follow the machine: ``follow`` orbits the tool centre point, ``probe`` is a camera
#: mounted on the probe looking at the tip, ``table`` looks down on the table from above.
FOLLOW_VIEWS = ("follow", "probe", "table")


class Viewport(QWidget):
    """Orbit with the left button, pan with the right (or Shift), zoom with the wheel."""

    def __init__(self, view: SimulationView, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.view = view
        self.azimuth = math.radians(-55.0)
        self.elevation = math.radians(28.0)
        self.distance = 2000.0
        self.target = np.zeros(3)
        self.show_machine = True
        self.machine_opacity = 150
        self.show_csys = True
        #: The coordinate system picked in the Coordinates tab; drawn bold.
        self.csy_selected: str | None = None
        self.show_contacts = True
        self.show_clouds = True
        self.setMinimumSize(480, 360)
        self.setMouseTracking(False)
        self._drag: tuple[QPointF, bool] | None = None
        self._interactive = False
        self._cache: dict[str, tuple[bytes, NDArray[np.float64], NDArray[np.float64]]] = {}
        self._normals: dict[int, NDArray[np.float64]] = {}
        self._settle = QTimer(self)
        self._settle.setSingleShot(True)
        self._settle.setInterval(180)
        self._settle.timeout.connect(self._settled)
        self.fit()

    # -- camera ------------------------------------------------------------------------

    def fit(self) -> None:
        lo, hi = self.view.bounds()
        self.target = (np.asarray(lo) + np.asarray(hi)) / 2.0
        self.distance = float(np.linalg.norm(np.asarray(hi) - np.asarray(lo))) * 2.0
        self.update()

    #: Which camera is active: a preset, a follow view, or ``free`` after the mouse moved it.
    camera = "isometric"

    def set_view(self, name: str) -> None:
        """Switch the camera: a name of :data:`VIEW_PRESETS`, :data:`FOLLOW_VIEWS` or ``fit``."""
        if name == "fit":
            self.fit()
            return
        if name in VIEW_PRESETS:
            azimuth, elevation = VIEW_PRESETS[name]
            self.azimuth, self.elevation = math.radians(azimuth), math.radians(elevation)
            self.fit()
        elif name in FOLLOW_VIEWS:
            if name == "follow":
                self.distance = min(self.distance, 400.0)
            if name == "table":
                self.elevation = math.radians(89.0)
                self.fit()
                self.distance *= 0.7
        else:
            raise ValueError(f"unknown view {name!r}")
        self.camera = name
        self.update()

    def _tool_pose(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        snap = self.view.snapshot()
        tcp = np.asarray(snap.position, dtype=float) + np.asarray(snap.world_shift, dtype=float)
        return tcp, np.asarray(snap.tool_axis, dtype=float)

    def focus_on(self, point: tuple[float, float, float], distance: float | None = None) -> None:
        self.target = np.asarray(point, dtype=float)
        if distance is not None:
            self.distance = distance
        self.update()

    def _eye(self) -> NDArray[np.float64]:
        c = math.cos(self.elevation)
        return self._look_target() + self.distance * np.array(
            [c * math.cos(self.azimuth), c * math.sin(self.azimuth), math.sin(self.elevation)]
        )

    def _look_target(self) -> NDArray[np.float64]:
        if self.camera in ("follow", "probe"):
            return self._tool_pose()[0]
        return self.target

    def _view_matrix(self) -> NDArray[np.float64]:
        if self.camera == "probe":
            tcp, axis = self._tool_pose()
            norm = float(np.linalg.norm(axis)) or 1.0
            axis = axis / norm
            side = np.cross(axis, np.array([0.0, 1.0, 0.0]))
            if np.linalg.norm(side) < 1e-6:
                side = np.array([1.0, 0.0, 0.0])
            side = side / np.linalg.norm(side)
            return _look_at(tcp + axis * 60.0 + side * 35.0, tcp)
        return _look_at(self._eye(), self._look_target())

    def _project(
        self, world: NDArray[np.float64], view: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """``(N, 3)`` world points to ``(N, 2)`` pixels and ``(N,)`` depth (negative = behind)."""
        cam = world @ view[:3, :3].T + view[:3, 3]
        depth = -cam[:, 2]
        focal = (self.height() / 2.0) / math.tan(_FOV / 2.0)
        safe = np.where(depth > 1e-6, depth, 1e-6)
        xy = np.empty((len(world), 2))
        xy[:, 0] = self.width() / 2.0 + cam[:, 0] / safe * focal
        xy[:, 1] = self.height() / 2.0 - cam[:, 1] / safe * focal
        return xy, depth

    # -- mouse -------------------------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        pan = bool(
            event.button() == Qt.MouseButton.RightButton
            or event.modifiers() & Qt.KeyboardModifier.ShiftModifier
        )
        self._drag = (event.position(), pan)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._drag is None:
            return
        last, pan = self._drag
        d = event.position() - last
        self._drag = (event.position(), pan)
        if pan:
            scale = self.distance * math.tan(_FOV / 2.0) * 2.0 / max(self.height(), 1)
            view = self._view_matrix()
            right, up = view[0, :3], view[1, :3]
            self.target = self.target - right * d.x() * scale + up * d.y() * scale
        else:
            self.azimuth -= d.x() * 0.008
            self.elevation = max(-1.5, min(1.5, self.elevation + d.y() * 0.008))
        if (
            (pan and self.camera in ("follow", "probe"))
            or self.camera in VIEW_PRESETS
            or self.camera == "table"
        ):
            self.camera = "free"
        self._begin_interaction()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        del event
        self._drag = None

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        del event
        self.fit()

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        steps = event.angleDelta().y() / 120.0
        self.distance = max(20.0, self.distance * (0.88**steps))
        self._begin_interaction()

    def _begin_interaction(self) -> None:
        self._interactive = True
        self._settle.start()
        self.update()

    def _settled(self) -> None:
        self._interactive = False
        self.update()

    # -- drawing -----------------------------------------------------------------------

    def _world_mesh(self, item: DrawItem) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        key = item.pose.tobytes()
        cached = self._cache.get(item.key)
        mesh = item.mesh
        if cached is not None and cached[0] == key + id(mesh).to_bytes(8, "little"):
            return cached[1], cached[2]
        vertices = mesh.vertices @ item.pose[:3, :3].T + item.pose[:3, 3]
        normals = self._face_normals(mesh) @ item.pose[:3, :3].T
        self._cache[item.key] = (key + id(mesh).to_bytes(8, "little"), vertices, normals)
        return vertices, normals

    def _face_normals(self, mesh: Mesh) -> NDArray[np.float64]:
        cached = self._normals.get(id(mesh))
        if cached is None:
            v = mesh.vertices[mesh.faces]
            n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
            length = np.linalg.norm(n, axis=1, keepdims=True)
            cached = n / np.where(length > 1e-12, length, 1.0)
            self._normals[id(mesh)] = cached
        return cached

    def paintEvent(self, event: object) -> None:  # noqa: N802
        del event
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(30, 33, 40))
        try:
            self._paint(painter)
        finally:
            painter.end()

    def render_to_image(self, width: int = 960, height: int = 640) -> QImage:
        """Draw the current view into an image (for screenshots and tests)."""
        image = QImage(width, height, QImage.Format.Format_RGB32)
        old = self.size()
        self.resize(width, height)
        painter = QPainter(image)
        painter.fillRect(image.rect(), QColor(30, 33, 40))
        try:
            self._paint(painter)
        finally:
            painter.end()
        self.resize(old)
        return image

    def _paint(self, painter: QPainter) -> None:
        view = self._view_matrix()
        eye = -view[:3, :3].T @ view[:3, 3]
        items = self.view.draw_items()
        self._cache = {k: v for k, v in self._cache.items() if k in {i.key for i in items}}

        xy_all: list[NDArray[np.float64]] = []
        depth_all: list[NDArray[np.float64]] = []
        rgba_all: list[NDArray[np.int64]] = []
        for item in items:
            if item.kind == "machine" and not self.show_machine:
                continue
            if item.mesh.is_empty:
                continue
            vertices, normals = self._world_mesh(item)
            faces = item.mesh.faces
            if self._interactive and len(faces) > _INTERACTIVE_BUDGET:
                stride = math.ceil(len(faces) / _INTERACTIVE_BUDGET)
                faces, normals = faces[::stride], normals[::stride]
            xy, depth = self._project(vertices, view)
            tri_xy = xy[faces]
            tri_depth = depth[faces]
            visible = np.all(tri_depth > 1.0, axis=1)
            centroid = vertices[faces].mean(axis=1)
            facing = np.einsum("ij,ij->i", normals, eye - centroid)
            # Two-sided: flip normals that point away so both sides are lit.
            lit = np.abs(normals @ _LIGHT) * 0.65 + 0.35
            lit = np.where(facing < 0, np.abs(normals @ _LIGHT) * 0.45 + 0.3, lit)
            base = np.asarray(item.color) * 255.0
            rgb = np.clip(lit[:, None] * base[None, :], 0, 255).astype(np.int64)
            alpha = self.machine_opacity if item.kind == "machine" else 255
            rgba = np.concatenate([rgb, np.full((len(rgb), 1), alpha, dtype=np.int64)], axis=1)
            xy_all.append(tri_xy[visible])
            depth_all.append(tri_depth[visible].mean(axis=1))
            rgba_all.append(rgba[visible])
        if xy_all:
            tri_xy_all = np.concatenate(xy_all)
            order = np.argsort(-np.concatenate(depth_all))
            colors = np.concatenate(rgba_all)
            tri_xy_all = tri_xy_all[order]
            colors = colors[order]
            painter.setPen(Qt.PenStyle.NoPen)
            last = None
            for tri, c in zip(tri_xy_all, colors, strict=True):
                key = (int(c[0]), int(c[1]), int(c[2]), int(c[3]))
                if key != last:
                    painter.setBrush(QColor(*key))
                    last = key
                painter.drawPolygon(QPolygonF([QPointF(float(p[0]), float(p[1])) for p in tri]))
        self._paint_overlay(painter, view)

    def _line(self, painter: QPainter, view: NDArray[np.float64], a: object, b: object) -> None:
        xy, depth = self._project(np.asarray([a, b], dtype=float), view)
        if np.all(depth > 1.0):
            painter.drawLine(QPointF(*xy[0]), QPointF(*xy[1]))

    def _paint_csys(self, painter: QPainter, view: NDArray[np.float64], length: float) -> None:
        """Draw a triad and the name of every coordinate system of the chain."""
        shift = np.asarray(self.view.snapshot().world_shift, dtype=float)
        colors = (QColor(230, 70, 70), QColor(80, 200, 90), QColor(80, 130, 240))
        placed: list[tuple[str, NDArray[np.float64]]] = []
        for frame in self.view.csy_frames():
            m = np.asarray(frame.matrix, dtype=float)
            origin = m[:3, 3] + shift
            at_machine = float(np.linalg.norm(m - np.eye(4))) < 1e-9
            if at_machine and frame.name != "MachineCsy" and not frame.active:
                continue  # it coincides with the machine system: one triad is enough
            bold = frame.active or frame.name == self.csy_selected
            if frame.name != "MachineCsy":
                for i, color in enumerate(colors):
                    painter.setPen(QPen(color, 3 if bold else 1.5))
                    tip = origin + m[:3, i] * length * 0.8
                    self._line(painter, view, tuple(origin), tuple(tip))
            suffix = " (active)" if frame.active else ""
            placed.append((frame.name + suffix, origin))
        painter.setPen(QPen(QColor(235, 235, 120)))
        for i, (label, origin) in enumerate(placed):
            xy, depth = self._project(np.asarray([origin], dtype=float), view)
            if depth[0] > 1.0:
                painter.drawText(QPointF(xy[0, 0] + 6, xy[0, 1] + 14 + 12 * (i % 2)), label)

    def _paint_overlay(self, painter: QPainter, view: NDArray[np.float64]) -> None:
        lo, hi = self.view.bounds()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(QPen(QColor(120, 130, 150, 160), 1, Qt.PenStyle.DashLine))
        corners = [
            (x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])
        ]
        for i, a in enumerate(corners):
            for j, b in enumerate(corners):
                if i < j and sum(p != q for p, q in zip(a, b, strict=True)) == 1:
                    self._line(painter, view, a, b)
        axis_len = max(hi[0] - lo[0], 1.0) * 0.12
        for vec, color, label in (
            ((axis_len, 0, 0), QColor(230, 70, 70), "X"),
            ((0, axis_len, 0), QColor(80, 200, 90), "Y"),
            ((0, 0, axis_len), QColor(80, 130, 240), "Z"),
        ):
            painter.setPen(QPen(color, 2))
            self._line(painter, view, (0, 0, 0), vec)
            xy, depth = self._project(np.asarray([vec], dtype=float), view)
            if depth[0] > 1.0:
                painter.drawText(QPointF(xy[0, 0] + 4, xy[0, 1] - 4), label)

        if self.show_csys:
            self._paint_csys(painter, view, axis_len)
        if self.show_clouds:
            for cloud in list(self.view.clouds):
                self._points(painter, view, cloud, None)
        if self.show_contacts and self.view.contacts:
            self._points(
                painter, view, np.asarray(list(self.view.contacts)), QColor(255, 80, 60), 3.0
            )

        xy, depth = self._project(np.asarray([self._tool_pose()[0]], dtype=float), view)
        if depth[0] > 1.0:
            x, y = float(xy[0, 0]), float(xy[0, 1])
            painter.setPen(QPen(QColor(255, 220, 60), 1.5))
            painter.drawLine(QPointF(x - 9, y), QPointF(x + 9, y))
            painter.drawLine(QPointF(x, y - 9), QPointF(x, y + 9))
        painter.setPen(QColor(150, 160, 175))
        painter.drawText(10, self.height() - 10, f"Camera: {self.camera}")

    def _points(
        self,
        painter: QPainter,
        view: NDArray[np.float64],
        points: NDArray[np.float64],
        color: QColor | None,
        size: float = 1.6,
    ) -> None:
        if len(points) == 0:
            return
        if len(points) > 80_000:
            points = points[:: math.ceil(len(points) / 80_000)]
        # Measured points belong to the table, which slides on a machine with a moving table.
        points = points + np.asarray(self.view.snapshot().world_shift, dtype=float)
        xy, depth = self._project(points, view)
        ok = depth > 1.0
        xy, z = xy[ok], points[ok, 2]
        if color is not None:
            pen = QPen(color, size)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawPoints([QPointF(float(p[0]), float(p[1])) for p in xy])
            return
        lo, hi = float(z.min()), float(z.max())
        bucket = np.minimum(((z - lo) / max(hi - lo, 1e-9) * 7.99).astype(int), 7)
        for b in range(8):
            t = b / 7.0
            painter.setPen(
                QPen(
                    QColor(int(40 + 215 * t), int(200 - 120 * abs(t - 0.5)), int(255 - 215 * t)),
                    size,
                )
            )
            painter.drawPoints([QPointF(float(p[0]), float(p[1])) for p in xy[bucket == b]])
