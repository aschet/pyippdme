# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""The digital twin: machine, scene, sensors and a protocol server that drives them.

:class:`DigitalTwin` has no GUI and no Qt. It builds a
:class:`~pyippdme.simulation.virtual_cmm.VirtualCMM` whose seams
(:class:`~pyippdme.server.motion.MotionModel`,
:class:`~pyippdme.server.surface.SampleSurface`,
:class:`~pyippdme.server.surface.RawSensor`) are backed by a machine with
limits, timed motion and collision checking, CAD samples and fixtures, an
ISO 10360 style measuring error and a line scanner. Whatever a client does over
the wire shows up in :meth:`DigitalTwin.snapshot`, :meth:`DigitalTwin.draw_items`
and the listener events, which is all a front end needs (see
:class:`pyippdme.twin.view.SimulationView`).
"""

from __future__ import annotations

import math
import random
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from pyippdme.protocol.network import TCP_NETWORK, Network
from pyippdme.server.motion import MotionRequest
from pyippdme.simulation.virtual_cmm import VirtualCMM
from pyippdme.twin import cad, geometry
from pyippdme.twin.backend import TwinBackend
from pyippdme.twin.geometry import Matrix, Mesh
from pyippdme.twin.machine import MachineModel
from pyippdme.twin.motion import TwinMotion
from pyippdme.twin.objects import SceneObject, ToolShapes, primitive_fixture, tool_pose
from pyippdme.twin.sensor import LineScanner, LineScannerSpec
from pyippdme.twin.spec import ToolSpec
from pyippdme.types.csy import CsyStore, InMemoryCsyStore
from pyippdme.types.pointcloud import PointCloudSet
from pyippdme.types.vec3 import Vec3, add, normalize, scale

Listener = Callable[["TwinEvent"], None]

#: A steel sample: 11.5 um/m/K.
STEEL_CTE = 11.5e-6
_STEP = 1.5
_MAX_STEPS = 600


@dataclass(frozen=True, slots=True)
class TwinEvent:
    """Something that happened in the twin; ``kind`` tells what, ``data`` the details.

    Kinds: ``connect``, ``disconnect``, ``line_in``, ``line_out``, ``contact``,
    ``collision``, ``scan``, ``scene`` (objects added/removed/moved), ``machine``
    (machine model replaced), ``reset``.
    """

    kind: str
    data: dict[str, Any] = field(default_factory=dict)
    time: float = field(default_factory=time.time)


@dataclass(frozen=True, slots=True)
class TwinSnapshot:
    """The machine state at one moment, as a front end shows it."""

    position: Vec3
    rotary: float
    tool_name: str
    tool_axis: Vec3
    homed: bool
    moving: bool
    connected: bool
    peer: str
    error: str
    session_active: bool
    machine_class: str


@dataclass(frozen=True, slots=True)
class DrawItem:
    """One body to draw: ``mesh`` is in its own frame, ``pose`` places it in machine coordinates."""

    key: str
    name: str
    mesh: Mesh
    pose: Matrix
    color: tuple[float, float, float]
    kind: str  # machine, sample, fixture, tool


class TwinSurface:
    """Probing against the scene's CAD shapes with measuring noise and thermal drift."""

    def __init__(self, twin: DigitalTwin) -> None:
        self._twin = twin

    def intersect(self, origin: Vec3, direction: Vec3) -> Vec3 | None:
        twin = self._twin
        unit = normalize(direction)
        hit = twin.cast(origin, unit)
        if hit is None:
            return None
        point, _distance, obj = hit
        if twin.noise_enabled:
            sigma = twin.machine.spec.accuracy.sigma_mm(float(np.linalg.norm(point)))
            point = add(point, scale(unit, twin.rng.gauss(0.0, sigma)))
        drift = STEEL_CTE * (twin.temperature - 20.0)
        if drift and obj is not None:
            ref = obj.pose[:3, 3]
            point = (
                point[0] + drift * (point[0] - ref[0]),
                point[1] + drift * (point[1] - ref[1]),
                point[2] + drift * (point[2] - ref[2]),
            )
        twin.record_contact(point)
        return point


class DigitalTwin:
    """A virtual CMM with a physical model; create the server with :meth:`create_server`."""

    def __init__(
        self,
        machine: MachineModel | None = None,
        *,
        seed: int | None = None,
        time_scale: float = 1.0,
        scanner: LineScannerSpec | None = None,
    ) -> None:
        self.machine = machine or MachineModel.default()
        #: 1.0 plays moves in real time, 4.0 four times faster, 0 makes them instant.
        self.time_scale = time_scale
        #: Share of the maximum axis speed the operator allows (the speed knob).
        self.speed_override = 1.0
        self.noise_enabled = True
        #: Part temperature in degrees Celsius; 20 is the reference (no thermal error).
        self.temperature = 20.0
        self.rng = random.Random(seed)  # noqa: S311 # nosec B311 (simulation noise)
        self.lock = threading.RLock()
        self.server: VirtualCMM | None = None
        self.objects: list[SceneObject] = []
        self.contacts: deque[Vec3] = deque(maxlen=20_000)
        self.clouds: deque[np.ndarray] = deque(maxlen=8)
        self.last_error = ""
        self.scene_version = 0
        self._tool_shapes = ToolShapes()
        self._tool_bounds: dict[ToolSpec, tuple[np.ndarray, np.ndarray]] = {}
        self._listeners: list[Listener] = []
        self._pos: Vec3 = (0.0, 0.0, 0.0)
        self._rotary = 0.0
        self._published = 0.0
        self._connected = ""
        self.scanner = LineScanner(
            lambda o, d: self._scan_cast(o, d),
            scanner,
            seed=seed,
            noise=lambda: self.noise_enabled,
            on_acquired=self._record_cloud,
        )

    # -- listeners ---------------------------------------------------------------------

    def add_listener(self, listener: Listener) -> None:
        self._listeners.append(listener)

    def remove_listener(self, listener: Listener) -> None:
        self._listeners.remove(listener)

    def emit(self, kind: str, **data: Any) -> None:
        event = TwinEvent(kind, data)
        for listener in list(self._listeners):
            listener(event)

    # -- server ------------------------------------------------------------------------

    def create_server(
        self, *, csy_store: CsyStore | None = None, network: Network = TCP_NETWORK
    ) -> VirtualCMM:
        """Build the protocol server of this twin (use ``await server.start(host, port)``)."""
        server = VirtualCMM(
            csy_store=csy_store if csy_store is not None else InMemoryCsyStore(),
            network=network,
            backend=TwinBackend(self),
            sample_surface=TwinSurface(self),
            raw_sensor=self.scanner,
            motion=TwinMotion(self),
            on_line_received=lambda line: self.emit("line_in", line=line),
            on_line_sent=lambda line: self.emit("line_out", line=line),
            on_connect=self._on_connect,
            on_disconnect=self._on_disconnect,
        )
        self.server = server
        return server

    def _on_connect(self, peer: str) -> None:
        self._connected = peer
        self.emit("connect", peer=peer)

    def _on_disconnect(self, peer: str) -> None:
        self._connected = ""
        self.emit("disconnect", peer=peer)

    # -- scene -------------------------------------------------------------------------

    def _scene_changed(self, kind: str = "scene") -> None:
        self.scene_version += 1
        self.emit(kind)

    def set_machine(self, machine: MachineModel) -> None:
        with self.lock:
            self.machine = machine
            self._tool_bounds.clear()
        self._scene_changed("machine")

    def add_object(self, obj: SceneObject) -> SceneObject:
        with self.lock:
            self.objects.append(obj)
        self._scene_changed()
        return obj

    def remove_object(self, obj: SceneObject) -> None:
        with self.lock:
            self.objects = [o for o in self.objects if o.id != obj.id]
        self._scene_changed()

    def load_sample(self, path: str | Path, *, replace: bool = True) -> SceneObject:
        """Read a CAD file and put the sample on the table at the middle of the volume."""
        obj = SceneObject.from_file(path, "sample")
        return self.place_sample(obj, replace=replace)

    def rest_on_table(self, obj: SceneObject) -> None:
        """Rest ``obj`` on the rotary table (if it is mounted there) or the table, centred."""
        spec = self.machine.spec
        if spec.rotary_origin is not None and obj.on_rotary:
            top = max(
                (cad.bounding_box(b.shape)[1][2] for b in self.machine.bodies if b.spec.rotates),
                default=spec.table_top_z,
            )
            obj.drop_to(top, spec.rotary_origin[0], spec.rotary_origin[1])
        else:
            obj.drop_to(spec.table_top_z, spec.travel[0] / 2, spec.travel[1] / 2)

    def place_sample(self, obj: SceneObject, *, replace: bool = True) -> SceneObject:
        """Rest ``obj`` on the table (or the rotary table if the machine has one)."""
        obj.on_rotary = self.machine.spec.rotary_origin is not None
        self.rest_on_table(obj)
        with self.lock:
            if replace and obj.kind == "sample":
                self.objects = [o for o in self.objects if o.kind != "sample"]
            self.objects.append(obj)
        self._scene_changed()
        return obj

    def set_pose(self, obj: SceneObject, pose: Matrix) -> None:
        with self.lock:
            obj.pose = pose
        self._scene_changed()

    def add_fixture(self, kind: str, size: Vec3, at: Vec3) -> SceneObject:
        obj = primitive_fixture(kind, size)
        obj.pose = geometry.translation(*at)
        return self.add_object(obj)

    def clear_scene(self) -> None:
        with self.lock:
            self.objects = []
        self._scene_changed()

    def clear_measurements(self) -> None:
        self.contacts.clear()
        self.clouds.clear()
        self.emit("reset")

    # -- measuring ---------------------------------------------------------------------

    def _current_rotary(self) -> float:
        state = self.server.active_state if self.server else None
        return state.rotary_table.position if state is not None else self._rotary

    def cast(self, origin: Vec3, direction: Vec3) -> tuple[Vec3, float, SceneObject | None] | None:
        """Nearest hit of a ray on any visible object, in machine coordinates."""
        with self.lock:
            rotary = self.machine.rotary_pose(self._current_rotary())
            best: tuple[Vec3, float, SceneObject | None] | None = None
            for obj in self.objects:
                if not obj.visible:
                    continue
                inverse = np.asarray(np.linalg.inv(obj.world_pose(rotary)), dtype=np.float64)
                o = geometry.apply(inverse, np.asarray([origin], dtype=np.float64))[0]
                d = inverse[:3, :3] @ np.asarray(direction)
                d = d / np.linalg.norm(d)
                hit = obj.caster.first_hit(
                    (float(o[0]), float(o[1]), float(o[2])), (float(d[0]), float(d[1]), float(d[2]))
                )
                if hit is not None and (best is None or hit[1] < best[1]):
                    w = geometry.apply(obj.world_pose(rotary), np.asarray([hit[0]]))[0]
                    best = ((float(w[0]), float(w[1]), float(w[2])), hit[1], obj)
            return best

    def _scan_cast(self, origin: Vec3, direction: Vec3) -> tuple[Vec3, float] | None:
        hit = self.cast(origin, direction)
        return None if hit is None else (hit[0], hit[1])

    def _record_cloud(self, cloud: PointCloudSet) -> None:
        points = np.array(
            [[p.x, p.y, p.z] for c in cloud.point_clouds for ps in c.point_sets for p in ps.points]
        ).reshape(-1, 3)
        self.clouds.append(points)
        self.emit("scan", points=len(points))

    def jog(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> bool:
        """Move the machine by hand (the jog box); False if no client session or out of volume."""
        state = self.server.active_state if self.server else None
        if state is None or not state.mover.user_enabled:
            return False
        x, y, z = state.cart_cmm.position
        target = (x + dx, y + dy, z + dz)
        if any(not 0.0 <= v <= t for v, t in zip(target, self.machine.spec.travel, strict=True)):
            return False
        state.cart_cmm.position = target
        return True

    def record_contact(self, point: Vec3) -> None:
        self.contacts.append(point)
        self.emit("contact", point=point)

    # -- motion support ----------------------------------------------------------------

    def publish_motion(self, position: Vec3, rotary: float) -> None:
        self._pos = position
        self._rotary = rotary
        self._published = time.monotonic()

    def report_collision(self, what: str, at: Vec3) -> None:
        self.last_error = f"Collision with {what}"
        self.emit("collision", what=what, at=at)

    def _tool_axis(self, tool_name: str) -> Vec3:
        state = self.server.active_state if self.server else None
        if state is not None:
            aligned = state.tool.alignment.get(tool_name)
            if aligned is not None:
                return aligned[0]
        return (0.0, 0.0, 1.0)

    def _bounds_of_tool(self, spec: ToolSpec) -> tuple[np.ndarray, np.ndarray]:
        if spec not in self._tool_bounds:
            lo, hi = cad.bounding_box(self._tool_shapes.shape(spec, (0, 0, 0), (0, 0, 1)))
            self._tool_bounds[spec] = (np.asarray(lo), np.asarray(hi))
        return self._tool_bounds[spec]

    @staticmethod
    def _aabb(lo: np.ndarray, hi: np.ndarray, m: Matrix) -> tuple[np.ndarray, np.ndarray]:
        import itertools

        corners = np.array(list(itertools.product(*zip(lo, hi, strict=True))))
        w = geometry.apply(m, corners)
        return w.min(axis=0), w.max(axis=0)

    @staticmethod
    def _overlap(a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]) -> bool:
        return bool(np.all(a[0] <= b[1] + 0.05) and np.all(b[0] <= a[1] + 0.05))

    def first_collision(self, request: MotionRequest, *, probing: bool) -> tuple[Vec3, str | None]:
        """Walk the move in small steps; return where the machine stops and what it hit."""
        spec = self.machine.spec
        tool = self.machine.tool(request.tool_name)
        axis = self._tool_axis(request.tool_name)
        delta = tuple(e - s for s, e in zip(request.start, request.end, strict=True))
        length = math.sqrt(sum(d * d for d in delta))
        steps = max(1, min(_MAX_STEPS, math.ceil(length / max(_STEP, tool.ball_radius))))
        previous = request.start
        with self.lock:
            # What the tool already touches at the start (it rests on a probed surface) does not
            # stop it from moving away; the exemption ends once it is clear of that obstacle.
            ignored = self._collisions(
                request.start, request.rotary_start, tool, axis, probing, spec.travel
            )
            for i in range(1, steps + 1):
                f = i / steps
                pos = (
                    request.start[0] + delta[0] * f,
                    request.start[1] + delta[1] * f,
                    request.start[2] + delta[2] * f,
                )
                rotary = request.rotary_start + (request.rotary_end - request.rotary_start) * f
                hits = self._collisions(pos, rotary, tool, axis, probing, spec.travel)
                fresh = hits - ignored
                if fresh:
                    return previous, sorted(fresh)[0]
                ignored &= hits
                previous = pos
        return request.end, None

    def _collisions(
        self,
        pos: Vec3,
        rotary: float,
        tool: ToolSpec,
        axis: Vec3,
        probing: bool,
        travel: Vec3,
    ) -> set[str]:
        del travel
        found: set[str] = set()
        rot_pose = self.machine.rotary_pose(rotary)
        if probing:
            # The protocol puts the TCP on the surface being touched (the tip radius is taken
            # as zero), so the stem starts one ball diameter above it and the tip is the sensor.
            a = np.asarray(normalize(axis))
            lift = 2.0 * tool.ball_radius + 0.05
            pos = (pos[0] + a[0] * lift, pos[1] + a[1] * lift, pos[2] + a[2] * lift)
        tool_pose_m = tool_pose(pos, axis)
        lo, hi = self._bounds_of_tool(tool)
        tool_box = self._aabb(lo, hi, tool_pose_m)
        tool_shape = None
        # Stylus against the table, rotary table and every sample/fixture.
        for body in self.machine.bodies:
            if not body.spec.collides or body.spec.moves_with:
                continue
            pose = self.machine.body_pose(body, pos, rotary)
            blo, bhi = cad.bounding_box(body.shape)
            if not self._overlap(tool_box, self._aabb(np.array(blo), np.array(bhi), pose)):
                continue
            tool_shape = tool_shape or self._tool_shapes.shape(
                tool, pos, axis, with_ball=not probing
            )
            if cad.min_distance(tool_shape, cad.moved(body.shape, pose)) < 0.01:
                found.add(body.spec.name)
        for obj in self.objects:
            world = obj.world_pose(rot_pose)
            obox = obj.world_bounds(rot_pose)
            if not obj.visible:
                continue
            if self._overlap(tool_box, obox):
                tool_shape = tool_shape or self._tool_shapes.shape(
                    tool, pos, axis, with_ball=not probing
                )
                if cad.min_distance(tool_shape, cad.moved(obj.shape, world)) < 0.01:
                    found.add(obj.name)
            # Carrying parts of the machine against the sample and fixtures.
            for body in self.machine.bodies:
                if not body.spec.collides or not body.spec.moves_with:
                    continue
                pose = self.machine.body_pose(body, pos, rotary)
                blo, bhi = cad.bounding_box(body.shape)
                if self._overlap(self._aabb(np.array(blo), np.array(bhi), pose), obox) and (
                    cad.min_distance(cad.moved(body.shape, pose), cad.moved(obj.shape, world))
                    < 0.01
                ):
                    found.add(f"{body.spec.name} / {obj.name}")
        return found

    # -- presentation ------------------------------------------------------------------

    def snapshot(self) -> TwinSnapshot:
        state = self.server.active_state if self.server else None
        moving = time.monotonic() - self._published < 0.25
        tool_name = "RefTool"
        homed = False
        error = self.last_error
        session = False
        machine_class = ""
        position, rotary = self._pos, self._rotary
        if state is not None:
            tool_name = state.tool.active_name
            homed = state.homed
            session = state.session_active
            machine_class = str(state.machine_class)
            if state.active_error is not None:
                error = str(state.active_error)
            elif not moving:
                error = ""
            if not moving:
                position = state.cart_cmm.position
                rotary = state.rotary_table.position
                self._pos, self._rotary = position, rotary
        return TwinSnapshot(
            position=position,
            rotary=rotary,
            tool_name=tool_name,
            tool_axis=self._tool_axis(tool_name),
            homed=homed,
            moving=moving,
            connected=bool(self._connected),
            peer=self._connected,
            error=error,
            session_active=session,
            machine_class=machine_class,
        )

    def draw_items(self) -> list[DrawItem]:
        """Everything to draw now, with current poses."""
        snap = self.snapshot()
        rotary = self.machine.rotary_pose(snap.rotary)
        items = [
            DrawItem(
                f"machine:{i}:{b.spec.name}",
                b.spec.name,
                b.mesh,
                self.machine.body_pose(b, snap.position, snap.rotary),
                b.color,
                "machine",
            )
            for i, b in enumerate(self.machine.bodies)
        ]
        with self.lock:
            items.extend(
                DrawItem(f"{o.kind}:{o.id}", o.name, o.mesh, o.world_pose(rotary), o.color, o.kind)
                for o in self.objects
                if o.visible
            )
        tool = self.machine.tool(snap.tool_name)
        items.append(
            DrawItem(
                "tool",
                tool.name,
                self._tool_shapes.mesh(tool),
                tool_pose(snap.position, snap.tool_axis),
                tool.color,
                "tool",
            )
        )
        return items

    def bounds(self) -> tuple[Vec3, Vec3]:
        """Return the measuring volume, for framing the view."""
        t = self.machine.spec.travel
        return (0.0, 0.0, 0.0), (t[0], t[1], t[2])
