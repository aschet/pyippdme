# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""The OpenGL surface of :class:`~pyippdme.gui.viewport.Viewport`.

The triangles of every mesh are uploaded to the graphics card once and drawn with a pose
matrix, so moving the machine or the camera costs a few uniforms instead of a projection and
sort of every triangle in numpy. No package beyond PySide6 is needed (``QOpenGLShaderProgram``
and ``QOpenGLBuffer``). The shaders are GLSL 1.10 (and valid GLSL ES 1.00), which every desktop
driver and the legacy profile of macOS understand. Lines, text and the coordinate system
triads stay in ``QPainter`` on top, as in the software view. If the context or the shaders
cannot be made, the canvas reports :attr:`GLCanvas.failed` and the viewport falls back to
software rendering.
"""

from __future__ import annotations

import math
import weakref
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QMatrix4x4, QOpenGLContext, QPainter, QSurfaceFormat, QVector3D, QVector4D
from PySide6.QtOpenGL import QOpenGLBuffer, QOpenGLShader, QOpenGLShaderProgram
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QWidget

from pyippdme.twin.geometry import Mesh
from pyippdme.twin.twin import DrawItem

if TYPE_CHECKING:
    from pyippdme.gui.viewport import Viewport

__all__ = ["GLCanvas", "perspective"]

# OpenGL constants; PySide6 does not export them.
_GL_POINTS = 0x0000
_GL_TRIANGLES = 0x0004
_GL_FLOAT = 0x1406
_GL_DEPTH_TEST = 0x0B71
_GL_BLEND = 0x0BE2
_GL_SRC_ALPHA = 0x0302
_GL_ONE_MINUS_SRC_ALPHA = 0x0303
_GL_COLOR_BUFFER_BIT = 0x4000
_GL_DEPTH_BUFFER_BIT = 0x0100
_GL_VERTEX_PROGRAM_POINT_SIZE = 0x8642
_FLOAT = 4

_MESH_VERTEX = """
#ifdef GL_ES
precision mediump float;
#endif
attribute vec3 a_pos;
attribute vec3 a_normal;
uniform mat4 u_mvp;
uniform mat4 u_model;
varying vec3 v_normal;
void main() {
    mat3 rotation = mat3(u_model[0].xyz, u_model[1].xyz, u_model[2].xyz);
    v_normal = rotation * a_normal;
    gl_Position = u_mvp * vec4(a_pos, 1.0);
}
"""

# Two-sided like the software view: the side that faces away is a little darker.
_MESH_FRAGMENT = """
#ifdef GL_ES
precision mediump float;
#endif
uniform vec4 u_color;
uniform vec3 u_light;
varying vec3 v_normal;
void main() {
    float d = abs(dot(normalize(v_normal), u_light));
    float lit = gl_FrontFacing ? d * 0.65 + 0.35 : d * 0.45 + 0.30;
    gl_FragColor = vec4(u_color.rgb * lit, u_color.a);
}
"""

_POINT_VERTEX = """
#ifdef GL_ES
precision mediump float;
#endif
attribute vec3 a_pos;
attribute vec3 a_color;
uniform mat4 u_mvp;
uniform vec3 u_offset;
uniform float u_size;
varying vec3 v_color;
void main() {
    v_color = a_color;
    gl_Position = u_mvp * vec4(a_pos + u_offset, 1.0);
    gl_PointSize = u_size;
}
"""

_POINT_FRAGMENT = """
#ifdef GL_ES
precision mediump float;
#endif
varying vec3 v_color;
void main() {
    gl_FragColor = vec4(v_color, 1.0);
}
"""

#: A cloud with more points than this is thinned for drawing.
_MAX_POINTS = 400_000


def perspective(fov: float, aspect: float, near: float, far: float) -> NDArray[np.float64]:
    """Return the OpenGL perspective matrix for a vertical field of view in radians."""
    f = 1.0 / math.tan(fov / 2.0)
    m = np.zeros((4, 4))
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2.0 * far * near / (near - far)
    m[3, 2] = -1.0
    return m


def _qmatrix(m: NDArray[np.float64]) -> QMatrix4x4:
    return QMatrix4x4(*[float(v) for v in m.reshape(16)])


class _Uploaded:
    """A mesh on the graphics card: interleaved position and face normal per vertex."""

    def __init__(self, mesh: Mesh, normals: NDArray[np.float64]) -> None:
        corners = mesh.vertices[mesh.faces].reshape(-1, 3)
        face_normals = np.repeat(normals, 3, axis=0)
        data = np.concatenate([corners, face_normals], axis=1).astype(np.float32)
        self.mesh = mesh  # keeps the mesh alive, so its id() cannot be reused while cached
        self.count = len(corners)
        self.buffer = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        self.buffer.create()
        self.buffer.bind()
        raw = data.tobytes()
        self.buffer.allocate(raw, len(raw))
        self.buffer.release()

    def destroy(self) -> None:
        self.buffer.destroy()


class _Points:
    """A set of points with colours on the graphics card."""

    def __init__(self, xyz: NDArray[np.float64], rgb: NDArray[np.float64]) -> None:
        data = np.concatenate([xyz, rgb], axis=1).astype(np.float32)
        self.count = len(data)
        self.buffer = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        self.buffer.create()
        self.buffer.bind()
        raw = data.tobytes()
        self.buffer.allocate(raw, len(raw))
        self.buffer.release()

    def destroy(self) -> None:
        self.buffer.destroy()


def _ramp(z: NDArray[np.float64]) -> NDArray[np.float64]:
    """Colour points by height like the software view: blue (low) to orange (high)."""
    lo, hi = float(z.min()), float(z.max())
    t = np.minimum(np.floor((z - lo) / max(hi - lo, 1e-9) * 7.99), 7) / 7.0
    return np.stack(
        [(40 + 215 * t) / 255, (200 - 120 * np.abs(t - 0.5)) / 255, (255 - 215 * t) / 255], axis=1
    )


class GLCanvas(QOpenGLWidget):
    """Draws the scene of a :class:`~pyippdme.gui.viewport.Viewport` with OpenGL."""

    #: The context or a shader could not be made; the text says why.
    failed = Signal(str)

    def __init__(self, owner: Viewport, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # A weak reference: owner and canvas must not form a cycle, or the garbage collector may
        # destroy the widget on another thread, which Qt does not allow for OpenGL widgets.
        self._owner = weakref.ref(owner)
        fmt = QSurfaceFormat()
        fmt.setDepthBufferSize(24)
        fmt.setSamples(4)
        self.setFormat(fmt)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._mesh_program: QOpenGLShaderProgram | None = None
        self._point_program: QOpenGLShaderProgram | None = None
        self._meshes: dict[int, _Uploaded] = {}
        self._contacts: tuple[int, _Points | None] = (-1, None)
        self._clouds: dict[int, tuple[int, _Points]] = {}
        self._ok = False

    @property
    def owner(self) -> Viewport:
        """The viewport this canvas draws for."""
        owner = self._owner()
        if owner is None:
            raise RuntimeError("the viewport of this canvas is gone")
        return owner

    # -- setup -------------------------------------------------------------------------

    def initializeGL(self) -> None:  # noqa: N802
        context = QOpenGLContext.currentContext()
        if context is None or not context.isValid():
            self.failed.emit("no OpenGL context")
            return
        self._funcs = context.functions()
        self._funcs.initializeOpenGLFunctions()
        # Free the buffers while their context still exists (see the QOpenGLWidget docs).
        context.aboutToBeDestroyed.connect(self.release)
        mesh = self._program(_MESH_VERTEX, _MESH_FRAGMENT)
        points = self._program(_POINT_VERTEX, _POINT_FRAGMENT)
        if mesh is None or points is None:
            return
        self._mesh_program, self._point_program = mesh, points
        self._ok = True

    def _program(self, vertex: str, fragment: str) -> QOpenGLShaderProgram | None:
        program = QOpenGLShaderProgram(self)
        stage = QOpenGLShader.ShaderTypeBit
        if not (
            program.addShaderFromSourceCode(stage.Vertex, vertex)
            and program.addShaderFromSourceCode(stage.Fragment, fragment)
            and program.link()
        ):
            self.failed.emit(program.log())
            return None
        return program

    @property
    def ok(self) -> bool:
        """Whether the context and the shaders work."""
        return self._ok and self.isValid()

    # -- drawing -----------------------------------------------------------------------

    def paintGL(self) -> None:  # noqa: N802
        owner = self._owner()
        if not self._ok or owner is None:
            return
        f = self._funcs
        f.glClearColor(30 / 255, 33 / 255, 40 / 255, 1.0)
        f.glClear(_GL_COLOR_BUFFER_BIT | _GL_DEPTH_BUFFER_BIT)
        view = owner._view_matrix()
        eye = -view[:3, :3].T @ view[:3, 3]
        aspect = max(self.width(), 1) / max(self.height(), 1)
        projection = perspective(owner.fov, aspect, 1.0, max(5000.0, float(owner.distance) * 8.0))
        vp = projection @ view
        items = owner.view.draw_items()
        shift = np.asarray(owner.view.snapshot().world_shift, dtype=float)
        self._forget({id(i.mesh) for i in items})

        f.glEnable(_GL_DEPTH_TEST)
        opaque = [i for i in items if i.kind != "machine" and not i.mesh.is_empty]
        glass = [
            i for i in items if i.kind == "machine" and owner.show_machine and not i.mesh.is_empty
        ]
        self._draw_meshes(opaque, vp, 255)
        # The translucent machine goes last, far parts first, without writing depth.
        glass.sort(key=lambda i: -self._distance(i, eye))
        f.glEnable(_GL_BLEND)
        f.glBlendFunc(_GL_SRC_ALPHA, _GL_ONE_MINUS_SRC_ALPHA)
        f.glDepthMask(False)
        self._draw_meshes(glass, vp, owner.machine_opacity)
        f.glDepthMask(True)
        f.glDisable(_GL_BLEND)
        self._draw_points(vp, shift)
        f.glDisable(_GL_DEPTH_TEST)

        painter = QPainter(self)
        try:
            owner._paint_overlay(painter, view, points=False)
        finally:
            painter.end()

    @staticmethod
    def _distance(item: DrawItem, eye: NDArray[np.float64]) -> float:
        """Return how far the middle of the item's box is from the eye (to sort the glass)."""
        lo, hi = item.mesh.bounds()
        centre = item.pose[:3, :3] @ ((lo + hi) / 2.0) + item.pose[:3, 3]
        return float(np.linalg.norm(centre - eye))

    def _draw_meshes(self, items: list[DrawItem], vp: NDArray[np.float64], alpha: int) -> None:
        program = self._mesh_program
        if program is None or not items:
            return
        from pyippdme.gui.viewport import LIGHT

        f = self._funcs
        program.bind()
        program.setUniformValue(
            program.uniformLocation("u_light"), QVector3D(*[float(v) for v in LIGHT])
        )
        a_pos = program.attributeLocation("a_pos")
        a_normal = program.attributeLocation("a_normal")
        for item in items:
            uploaded = self._upload(item.mesh)
            r, g, b = item.color
            program.setUniformValue(
                program.uniformLocation("u_color"), QVector4D(r, g, b, alpha / 255.0)
            )
            program.setUniformValue(program.uniformLocation("u_model"), _qmatrix(item.pose))
            program.setUniformValue(program.uniformLocation("u_mvp"), _qmatrix(vp @ item.pose))
            uploaded.buffer.bind()
            program.enableAttributeArray(a_pos)
            program.enableAttributeArray(a_normal)
            program.setAttributeBuffer(a_pos, _GL_FLOAT, 0, 3, 6 * _FLOAT)
            program.setAttributeBuffer(a_normal, _GL_FLOAT, 3 * _FLOAT, 3, 6 * _FLOAT)
            f.glDrawArrays(_GL_TRIANGLES, 0, uploaded.count)
            program.disableAttributeArray(a_pos)
            program.disableAttributeArray(a_normal)
            uploaded.buffer.release()
        program.release()

    def _upload(self, mesh: Mesh) -> _Uploaded:
        cached = self._meshes.get(id(mesh))
        if cached is None:
            cached = _Uploaded(mesh, self.owner._face_normals(mesh))
            self._meshes[id(mesh)] = cached
        return cached

    def _forget(self, live: set[int]) -> None:
        """Free what is no longer drawn (a part that was removed, a replaced tool)."""
        for key in [k for k in self._meshes if k not in live]:
            self._meshes.pop(key).destroy()

    def _draw_points(self, vp: NDArray[np.float64], shift: NDArray[np.float64]) -> None:
        program = self._point_program
        owner = self.owner
        if program is None:
            return
        batches: list[tuple[_Points, float]] = []
        if owner.show_clouds:
            live = {id(c) for c in owner.view.clouds}
            for key in [k for k in self._clouds if k not in live]:
                self._clouds.pop(key)[1].destroy()
            for cloud in list(owner.view.clouds):
                if len(cloud):
                    batches.append((self._cloud(cloud), 3.0))
        if owner.show_contacts and owner.view.contacts:
            batches.append((self._contact_points(), 5.0))
        if not batches:
            return
        f = self._funcs
        f.glEnable(_GL_VERTEX_PROGRAM_POINT_SIZE)
        program.bind()
        program.setUniformValue(program.uniformLocation("u_mvp"), _qmatrix(vp))
        program.setUniformValue(
            program.uniformLocation("u_offset"), QVector3D(*[float(v) for v in shift])
        )
        a_pos = program.attributeLocation("a_pos")
        a_color = program.attributeLocation("a_color")
        for points, size in batches:
            program.setUniformValue(program.uniformLocation("u_size"), float(size))
            points.buffer.bind()
            program.enableAttributeArray(a_pos)
            program.enableAttributeArray(a_color)
            program.setAttributeBuffer(a_pos, _GL_FLOAT, 0, 3, 6 * _FLOAT)
            program.setAttributeBuffer(a_color, _GL_FLOAT, 3 * _FLOAT, 3, 6 * _FLOAT)
            f.glDrawArrays(_GL_POINTS, 0, points.count)
            program.disableAttributeArray(a_pos)
            program.disableAttributeArray(a_color)
            points.buffer.release()
        program.release()

    def _cloud(self, cloud: NDArray[np.float64]) -> _Points:
        key = id(cloud)
        cached = self._clouds.get(key)
        if cached is not None and cached[0] == len(cloud):
            return cached[1]
        if cached is not None:
            cached[1].destroy()
        xyz = np.asarray(cloud, dtype=float)
        if len(xyz) > _MAX_POINTS:
            xyz = xyz[:: math.ceil(len(xyz) / _MAX_POINTS)]
        points = _Points(xyz, _ramp(xyz[:, 2]))
        self._clouds[key] = (len(cloud), points)
        return points

    def _contact_points(self) -> _Points:
        contacts = self.owner.view.contacts
        count, cached = self._contacts
        if cached is not None and count == len(contacts):
            return cached
        if cached is not None:
            cached.destroy()
        xyz = np.asarray(list(contacts), dtype=float)
        red = np.tile(np.array([255, 80, 60]) / 255.0, (len(xyz), 1))
        points = _Points(xyz, red)
        self._contacts = (len(contacts), points)
        return points

    def release(self) -> None:
        """Free the buffers; called when the context goes away, or before the canvas is dropped."""
        if not self.isValid():
            return
        self.makeCurrent()
        try:
            for uploaded in self._meshes.values():
                uploaded.destroy()
            self._meshes.clear()
            for _, points in self._clouds.values():
                points.destroy()
            self._clouds.clear()
            if self._contacts[1] is not None:
                self._contacts[1].destroy()
            self._contacts = (-1, None)
        finally:
            self.doneCurrent()
