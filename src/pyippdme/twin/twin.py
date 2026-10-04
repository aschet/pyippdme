# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""The digital twin: machine, scene, tools, sensors and a protocol server that drives them.

:class:`DigitalTwin` has no GUI and no Qt. It builds a
:class:`~pyippdme.simulation.virtual_cmm.VirtualCMM` whose seams
(:class:`~pyippdme.server.motion.MotionModel`,
:class:`~pyippdme.server.motion.ToolHandler`,
:class:`~pyippdme.server.surface.SampleSurface`,
:class:`~pyippdme.server.surface.RawSensor`,
:class:`~pyippdme.server.backend.MachineBackend`) are backed by a machine with
limits, timed motion and collision checking, CAD samples and fixtures, probe
systems with their own measuring behaviour, and a line scanner. Whatever a client does
over the wire shows up in :meth:`DigitalTwin.snapshot`, :meth:`DigitalTwin.draw_items`
and the listener events, which is all a front end needs (see
:class:`pyippdme.twin.view.SimulationView`).

How the machine behaves, briefly: it must be homed with the air supply on and the
emergency stop released; moves run on the speeds and accelerations of the protocol
(``GoToPar``/``PtMeasPar``/``ScanPar``) within the machine's limits; a tool is qualified on
a reference sphere (per head position on an indexing head) or its points carry a systematic
error; kinematic probes have three-lobe pre-travel; a crash stops the machine and a
breakaway module detaches; ``ChangeTool`` drives to the rack.
"""

from __future__ import annotations

import asyncio
import itertools
import math
import random
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from pyippdme.protocol.errors import ErrorCode, ErrorSeverity, ServerError
from pyippdme.protocol.network import TCP_NETWORK, Network
from pyippdme.server.motion import MotionError, MotionRequest, ProbeRequest, ProbeResult
from pyippdme.server.registry import CommandRegistry
from pyippdme.simulation import DEFAULT_COMMAND_CLASSES
from pyippdme.simulation.virtual_cmm import VirtualCMM
from pyippdme.twin import cad, geometry
from pyippdme.twin.backend import TwinBackend
from pyippdme.twin.depthbuffer import DepthBuffer
from pyippdme.twin.features import Result, evaluate_features
from pyippdme.twin.geometry import Matrix, Mesh
from pyippdme.twin.machine import MachineModel
from pyippdme.twin.motion import TwinMotion
from pyippdme.twin.objects import SceneObject, primitive_fixture
from pyippdme.twin.optical import OpticalSensor, OpticalSpec
from pyippdme.twin.planning import (
    collision_steps,
    free_steps,
    plan_homing,
    plan_qualification,
    plan_safe_path,
    plan_tool_change,
    point_along,
    profile_position,
    profile_time,
    travel_time,
)
from pyippdme.twin.spec import ToolSpec
from pyippdme.twin.toolmath import OPTICAL_MODES, drop, head_rotation, reach_orientation, tip_offset
from pyippdme.twin.tools import ToolKit, ToolModel
from pyippdme.types.csy import (
    CSY_CHAIN,
    CoordinateTransform,
    CsyContext,
    CsyStore,
    InMemoryCsyStore,
    chain_matrix,
)
from pyippdme.types.obb import Obb
from pyippdme.types.pointcloud import MeasPoint, PointCloud, PointCloudSet, PointSet
from pyippdme.types.vec3 import Vec3, add, norm, normalize, scale, sub

Listener = Callable[["TwinEvent"], None]

#: A steel sample: 11.5 um/m/K.
STEEL_CTE = 11.5e-6
#: Faults the machine can have, by name: severity, error code and text of what moving raises.
FAULTS: dict[str, tuple[ErrorSeverity, str, str]] = {
    "axis_position_error": (
        ErrorSeverity.CRITICAL,
        ErrorCode.AXIS_POSITION_ERROR,
        "Axis position error: an axis does not follow its command",
    ),
    "axis_not_active": (
        ErrorSeverity.CRITICAL,
        ErrorCode.AXIS_NOT_ACTIVE,
        "Axis not active: a drive is switched off",
    ),
    "scale_read_head": (
        ErrorSeverity.FATAL,
        ErrorCode.SCALE_READ_HEAD_FAILURE,
        "Scale read head failure",
    ),
    "controller_link": (
        ErrorSeverity.FATAL,
        ErrorCode.CONTROLLER_COMMUNICATIONS_FAILURE,
        "No communication with the controller",
    ),
}
#: Expansion of the glass-ceramic scales of the machine, per kelvin.
SCALE_CTE = 8.0e-6
_FRAME = 1.0 / 60.0
#: Seconds a stylus-module change takes at the rack (TP20 modules: about 6 s with the moves).
_MODULE_CHANGE_DWELL = 2.0


@dataclass(frozen=True, slots=True)
class TwinEvent:
    """Something that happened in the twin; ``kind`` tells what, ``data`` the details.

    Kinds: ``connect``, ``disconnect``, ``line_in``, ``line_out``, ``contact``,
    ``collision``, ``scan``, ``scene`` (objects added/removed/moved), ``machine``
    (machine model replaced), ``tools`` (tool definitions edited), ``safety`` (e-stop or
    air pressure), ``qualified``, ``tool_change``, ``reset``.
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
    tool_mode: str
    tool_axis: Vec3
    head_position: tuple[float, float] | None
    homed: bool
    moving: bool
    connected: bool
    peer: str
    error: str
    session_active: bool
    machine_class: str
    estop: bool
    air_ok: bool
    qualified: bool
    detached: bool
    changing_tool: str
    #: How far the table frame is displaced in the world, for a machine with a moving table.
    world_shift: Vec3 = (0.0, 0.0, 0.0)
    #: The coordinate system the client works in (``SetCoordSystem``).
    active_csy: str = "MachineCsy"


@dataclass(frozen=True, slots=True)
class CsyFrame:
    """One coordinate system of the chain of 6.5.1, placed in machine coordinates."""

    name: str
    #: Maps points of this system to machine coordinates.
    matrix: Matrix
    #: The client works in this system (``SetCoordSystem``).
    active: bool
    #: What the client set with ``SetCsyTransformation``/``LoadCoordSystem``, if anything.
    placement: CoordinateTransform | None


@dataclass(frozen=True, slots=True)
class DrawItem:
    """One body to draw: ``mesh`` is in its own frame, ``pose`` places it in machine coordinates."""

    key: str
    name: str
    mesh: Mesh
    pose: Matrix
    color: tuple[float, float, float]
    kind: str  # machine, sample, fixture, tool, stored


@dataclass(frozen=True, slots=True)
class ToolPlacement:
    """A tool in the machine: where its pivot is for a tool centre point and head orientation."""

    name: str
    spec: ToolSpec
    model: ToolModel
    rotation: Matrix
    axis: Vec3
    pivot: Vec3
    orientation: tuple[float, float] | None


class TwinSurface:
    """Probing against the scene's CAD shapes through :meth:`DigitalTwin.measure`."""

    def __init__(self, twin: DigitalTwin) -> None:
        self._twin = twin

    def intersect(self, origin: Vec3, direction: Vec3) -> Vec3 | None:
        return self._twin.measure(origin, direction)


class TwinToolHandler:
    """``ChangeTool`` drives to the rack; ``ReQualify`` measures the reference sphere."""

    def __init__(self, twin: DigitalTwin) -> None:
        self._twin = twin

    async def change_tool(
        self, current: str, target: str, position: Vec3, cancel: asyncio.Event
    ) -> None:
        await self._twin.run_tool_change(current, target, self._twin.to_machine(position), cancel)

    async def requalify(self, tool_name: str, cancel: asyncio.Event) -> None:
        await self._twin.run_qualification(tool_name, cancel)

    def alignment_volume(self, tool_name: str) -> tuple[Vec3, float] | None:
        return self._twin.alignment_volume(tool_name)

    def collision_volume(self, tool_name: str) -> list[Obb] | None:
        return self._twin.collision_volume(tool_name)

    def is_calibrated(self, tool_name: str) -> bool:
        return self._twin.is_calibrated(tool_name)

    def avr_radius(self, tool_name: str) -> float:
        return self._twin.avr_radius(tool_name)

    def avr_offsets(self, tool_name: str) -> Vec3 | None:
        return self._twin.avr_offsets(tool_name)

    def reach(
        self, tool_name: str, primary: Vec3, secondary: Vec3 | None
    ) -> tuple[Vec3, Vec3 | None]:
        return self._twin.reach_alignment(tool_name, primary, secondary)


class ClientSensor:
    """Gives the optical sensor machine coordinates and the client its answers in its own CSY."""

    def __init__(self, twin: DigitalTwin) -> None:
        self._twin = twin

    def acquire(
        self,
        acq_name: str,
        acquisition_type: str,
        positions: list[Vec3],
        directions: list[Vec3],
    ) -> PointCloudSet | None:
        twin = self._twin
        context = twin.csy_context()
        cloud = twin.scanner.acquire(
            acq_name,
            acquisition_type,
            [twin.to_machine(p, context) for p in positions],
            [twin.to_machine_direction(d, context) for d in directions],
        )
        if cloud is None or np.allclose(context.matrix(), np.eye(4)):
            return cloud
        return PointCloudSet(
            tuple(
                PointCloud(
                    tuple(
                        PointSet(
                            twin.to_client_direction(ps.normal, context),
                            twin.to_client_direction(ps.direction, context),
                            tuple(
                                MeasPoint(*twin.to_client((p.x, p.y, p.z), context), p.q)
                                for p in ps.points
                            ),
                        )
                        for ps in pc.point_sets
                    )
                )
                for pc in cloud.point_clouds
            )
        )


class TwinServer(VirtualCMM):
    """The protocol server of a :class:`DigitalTwin`; its tools are known while it listens.

    The protocol's tool catalog is process-wide, so the twin's tools are registered when the
    server starts and removed when it closes.
    """

    def __init__(self, twin: DigitalTwin, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.twin = twin

    async def start(self, host: str = "127.0.0.1", port: int = 0) -> int:
        self.twin.toolkit.register()
        return await super().start(host, port)

    async def close(self) -> None:
        await super().close()
        self.twin.toolkit.unregister()


class DigitalTwin:
    """A virtual CMM with a physical model; create the server with :meth:`create_server`."""

    def __init__(
        self,
        machine: MachineModel | None = None,
        *,
        seed: int | None = None,
        time_scale: float = 1.0,
        scanner: OpticalSpec | None = None,
    ) -> None:
        self.machine = machine or MachineModel.default()
        self.toolkit = ToolKit(self.machine.tools)
        #: 1.0 plays moves in real time, 4.0 four times faster, 0 makes them instant.
        self.time_scale = time_scale
        #: Share of the maximum axis speed the operator allows (the speed knob).
        self.speed_override = 1.0
        self.noise_enabled = True
        #: Part temperature in degrees Celsius; 20 is the reference (no thermal error).
        self.temperature = 20.0
        #: Air temperature in the room; the scales of the machine follow it.
        self.ambient_temperature = 20.0
        #: Linear expansion of the part and of the machine scales, per kelvin.
        self.part_cte = STEEL_CTE
        self.scale_cte = SCALE_CTE
        #: Whether an unqualified tool carries a systematic error (and ``ReQualify`` is needed).
        self.require_qualification = True
        #: Safety: emergency stop pressed, and whether the air supply is within range.
        self.estop = False
        #: Whether the last collision found was only the tip ball touching the part.
        self.last_hit_was_touch = False
        #: Machine faults that are switched on (see :data:`FAULTS`); each fails what moves.
        self.faults: set[str] = set()
        self.air_ok = True
        self.rng = random.Random(seed)  # noqa: S311 # nosec B311 (simulation noise)
        self._seed = seed or 0
        self.lock = threading.RLock()
        self.server: VirtualCMM | None = None
        self.objects: list[SceneObject] = []
        self.contacts: deque[Vec3] = deque(maxlen=20_000)
        #: Measured points by measurement mode (``touch``, ``head_touch``, ``scanning``, ``laser``).
        self.points: dict[str, deque[Vec3]] = {}
        self.clouds: deque[np.ndarray] = deque(maxlen=8)
        #: ``(tool, head position)`` pairs that are qualified; a module that broke away.
        self.qualified: set[tuple[str, tuple[float, float] | None]] = set()
        self.detached: set[str] = set()
        self.last_error = ""
        self.scene_version = 0
        self._listeners: list[Listener] = []
        self._pos: Vec3 = (0.0, 0.0, 0.0)
        self._rotary = 0.0
        self._published = 0.0
        self._connected = ""
        self._display_tool: str | None = None
        self._changing = ""
        self._buffers: dict[tuple[object, ...], DepthBuffer] = {}
        self.scanner = OpticalSensor(
            self.depth_buffer,
            scanner,
            seed=seed,
            noise=lambda: self.noise_enabled,
            on_acquired=self._record_cloud,
        )
        self.scanner.spec_provider = self._optical_spec

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
        self,
        *,
        csy_store: CsyStore | None = None,
        network: Network = TCP_NETWORK,
        command_classes: Sequence[Callable[[CommandRegistry], None]] = DEFAULT_COMMAND_CLASSES,
        on_line_received: Callable[[str], None] | None = None,
        on_line_sent: Callable[[str], None] | None = None,
        on_connect: Callable[[str], None] | None = None,
        on_disconnect: Callable[[str], None] | None = None,
        max_pending: int | None = None,
    ) -> VirtualCMM:
        """Build the protocol server of this twin (use ``await server.start(host, port)``).

        The hooks are called after the twin has seen the line or the connection itself.
        ``max_pending`` makes the server delay the acknowledgement of a command while that many
        commands wait (5.4.3); by default it acknowledges at once.
        """

        def received(line: str) -> None:
            self.emit("line_in", line=line)
            if on_line_received is not None:
                on_line_received(line)

        def sent(line: str) -> None:
            self.emit("line_out", line=line)
            if on_line_sent is not None:
                on_line_sent(line)

        def connected(peer: str) -> None:
            self._on_connect(peer)
            if on_connect is not None:
                on_connect(peer)

        def disconnected(peer: str) -> None:
            self._on_disconnect(peer)
            if on_disconnect is not None:
                on_disconnect(peer)

        server = TwinServer(
            self,
            csy_store=csy_store if csy_store is not None else InMemoryCsyStore(),
            network=network,
            command_classes=command_classes,
            backend=TwinBackend(self),
            sample_surface=TwinSurface(self),
            raw_sensor=ClientSensor(self),
            motion=TwinMotion(self),
            tool_handler=TwinToolHandler(self),
            on_line_received=received,
            on_line_sent=sent,
            on_connect=connected,
            on_disconnect=disconnected,
            max_pending=max_pending,
        )
        self.server = server
        return server

    def _on_connect(self, peer: str) -> None:
        self._connected = peer
        self.emit("connect", peer=peer)

    def _on_disconnect(self, peer: str) -> None:
        self._connected = ""
        self.emit("disconnect", peer=peer)

    @property
    def state(self) -> Any:
        return self.server.active_state if self.server else None

    # -- coordinate systems --------------------------------------------------------------

    def csy_context(self) -> CsyContext:
        """Return the client's coordinate system: what it activated and how it sits in the machine.

        A client commands and reads positions in the coordinate system it activated
        (``SetCoordSystem``), placed by the transformations it set (``SetCsyTransformation``,
        ``LoadCoordSystem``) along the chain of :data:`pyippdme.types.csy.CSY_CHAIN`. The twin
        works in machine coordinates and converts at the protocol boundary. The rotary table's
        own system follows its angle while ``EnableRotaryTableVarCsy`` is on.
        """
        state = self.state
        if state is None:
            return CsyContext()
        var = None
        if self.machine.spec.rotary_origin is not None and state.rotary_table.var_csy_enabled:
            var = self.machine.rotary_pose(state.rotary_table.position)
        return CsyContext(state.cart_cmm.active_csy, state.cart_cmm.csy_transformations, var)

    def csy_frames(self) -> list[CsyFrame]:
        """Return every coordinate system of the chain with its place in the machine."""
        context = self.csy_context()
        return [
            CsyFrame(
                name,
                chain_matrix(name, context.transforms, context.rotary_var),
                name == context.active,
                context.transforms.get(name),
            )
            for name in CSY_CHAIN
        ]

    def to_csy(self, point: Vec3, name: str) -> Vec3:
        """Convert a machine point to the coordinate system ``name`` of the chain."""
        context = self.csy_context()
        return CsyContext(name, context.transforms, context.rotary_var).to_client(point)

    def to_machine(self, point: Vec3, context: CsyContext | None = None) -> Vec3:
        """Convert a point of the client's coordinate system to machine coordinates."""
        return (context or self.csy_context()).to_machine(point)

    def to_client(self, point: Vec3, context: CsyContext | None = None) -> Vec3:
        """Convert a machine point to the client's coordinate system."""
        return (context or self.csy_context()).to_client(point)

    def to_machine_direction(self, vector: Vec3, context: CsyContext | None = None) -> Vec3:
        """Convert a direction (probing direction, tool axis) to machine coordinates."""
        return (context or self.csy_context()).direction_to_machine(vector)

    def to_client_direction(self, vector: Vec3, context: CsyContext | None = None) -> Vec3:
        """Convert a machine direction to the client's coordinate system."""
        return (context or self.csy_context()).direction_to_client(vector)

    def client_error(self, error: MotionError, context: CsyContext | None = None) -> MotionError:
        """Return a motion error with its stopping position in the client's coordinates."""
        return MotionError(
            error.severity,
            error.number,
            error.cause,
            error.text,
            self.to_client(error.stopped_at, context),
        )

    # -- scene -------------------------------------------------------------------------

    def _scene_changed(self, kind: str = "scene") -> None:
        self.scene_version += 1
        self.emit(kind)

    def set_machine(self, machine: MachineModel) -> None:
        with self.lock:
            self.machine = machine
            self.toolkit.unregister()
            self.toolkit = ToolKit(machine.tools)
            if self.server is not None:
                self.toolkit.register()
        self._scene_changed("machine")

    def set_tool_spec(self, spec: ToolSpec) -> None:
        """Add or change a tool definition; the protocol sees it at once."""
        with self.lock:
            self.machine.tools[spec.name] = spec
            self.toolkit.specs[spec.name] = spec
            self.qualified = {q for q in self.qualified if q[0] != spec.name}
            if self.server is not None:
                self.toolkit.register()
        self._scene_changed("tools")

    def remove_tool(self, name: str) -> None:
        with self.lock:
            self.machine.tools.pop(name, None)
            self.toolkit.specs.pop(name, None)
            if self.server is not None:
                self.toolkit.register()
        self._scene_changed("tools")

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
        return self.place_sample(SceneObject.from_file(path, "sample"), replace=replace)

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

    def jog(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> bool:
        """Move the machine by hand (the jog box); False if it is not allowed or out of volume."""
        state = self.state
        if state is None or not state.mover.user_enabled or self.estop or not self.air_ok:
            return False
        context = self.csy_context()
        x, y, z = self.to_machine(state.cart_cmm.position, context)
        target = (x + dx, y + dy, z + dz)
        if any(not 0.0 <= v <= t for v, t in zip(target, self.machine.spec.travel, strict=True)):
            return False
        state.cart_cmm.position = self.to_client(target, context)
        return True

    def pick_point(
        self, direction: Vec3 = (0.0, 0.0, -1.0), reach: float = 60.0
    ) -> tuple[Vec3, Vec3] | None:
        """Touch the part by hand: probe along ``direction`` from here, up to ``reach`` mm.

        This is what the user does at the jog box. Returns the measured point and the surface
        normal (machine coordinates), moves the machine to the touch and records the point, or
        returns ``None`` if the part is not within reach.
        """
        origin = self._pos
        hit = self.cast(origin, direction)
        if hit is None or hit[1] > reach:
            return None
        point = self.measure(origin, direction, mode=self.toolkit.spec(self.tool_name()).mode)
        if point is None:
            return None
        normal = self.surface_normal(hit[0], scale(normalize(direction), -1.0))
        state = self.state
        if state is not None:
            state.cart_cmm.position = self.to_client(
                add(
                    hit[0],
                    scale(normalize(direction), -self.toolkit.spec(self.tool_name()).ball_radius),
                )
            )
        return point, normal

    # -- planning ----------------------------------------------------------------------

    def segment_is_free(self, a: Vec3, b: Vec3) -> bool:
        """Whether the active tool can move from ``a`` to ``b`` (machine coordinates) unhindered."""
        request = MotionRequest(
            "Plan", a, b, self._rotary, self._rotary, True, self.tool_name(), asyncio.Event()
        )
        _, hit = self.first_collision(request, probing=False)
        return hit is None

    def plan_collision_free(self, target: Vec3, start: Vec3 | None = None) -> list[Vec3] | None:
        """Plan a collision-free path (machine coordinates) from ``start`` (default: here)."""
        spec = self.machine.spec
        upper = (spec.travel[0], spec.travel[1], spec.travel[2])
        return plan_safe_path(start or self._pos, target, self.segment_is_free, upper)

    async def go_safely(self, target: Vec3, cancel: asyncio.Event | None = None) -> list[Vec3]:
        """Drive to ``target`` (machine coordinates) along a planned collision-free path.

        Raises a protocol ``ServerError`` when the machine is not ready, and ``1011`` "unable to
        move" when no free path is found. Returns the path that was driven.
        """
        self.check_ready("GoTo")
        path = self.plan_collision_free(target)
        if path is None:
            raise ServerError(
                ErrorSeverity.ERROR,
                ErrorCode.UNABLE_TO_MOVE,
                "GoTo",
                "No collision-free path to the target was found",
            )
        params = self.protocol_parameters()
        event = cancel or asyncio.Event()
        await self.run_path(path, params["go_speed"], params["go_accel"], event, "GoTo")
        state = self.state
        if state is not None:
            state.cart_cmm.position = self.to_client(path[-1])
        return path

    def clear_scene(self) -> None:
        with self.lock:
            self.objects = []
        self._scene_changed()

    def clear_measurements(self) -> None:
        self.contacts.clear()
        self.clouds.clear()
        self.points.clear()
        self.emit("reset")

    # -- safety ------------------------------------------------------------------------

    def check_ready(self, cause: str) -> None:
        """Raise the protocol error for a machine that must not move or measure now."""
        if self.estop:
            raise ServerError(
                ErrorSeverity.CRITICAL, ErrorCode.EMERGENCY_STOP, cause, "Emergency stop is active"
            )
        if not self.air_ok:
            raise ServerError(
                ErrorSeverity.CRITICAL,
                ErrorCode.AIR_PRESSURE_OUT_OF_RANGE,
                cause,
                "Air pressure out of range",
            )
        for fault in FAULTS:
            if fault in self.faults:
                severity, code, text = FAULTS[fault]
                raise ServerError(severity, code, cause, text)

    def set_estop(self, pressed: bool) -> None:
        """Press or release the emergency stop; pressing it loses the reference (re-home)."""
        self.estop = pressed
        if pressed:
            self._lose_reference()
        self.last_error = "Emergency stop" if pressed else ""
        self.emit("safety", estop=pressed, air_ok=self.air_ok)

    def set_fault(self, kind: str, on: bool = True) -> None:
        """Switch a machine fault on or off (see :data:`FAULTS`).

        A fault stays until it is cleared. A fatal one (a failing scale, no link to the
        controller) also loses the reference, so the machine has to be homed again.
        """
        if kind not in FAULTS:
            raise ValueError(f"unknown fault {kind!r}; choose from {', '.join(FAULTS)}")
        if on:
            self.faults.add(kind)
            if FAULTS[kind][0] >= ErrorSeverity.FATAL:
                self._lose_reference()
            self.last_error = FAULTS[kind][2]
        else:
            self.faults.discard(kind)
            self.last_error = ""
        self.emit("safety", estop=self.estop, air_ok=self.air_ok)

    def set_air_ok(self, ok: bool) -> None:
        """Fail or restore the air supply; failing it brakes the machine and loses the reference."""
        self.air_ok = ok
        if not ok:
            self._lose_reference()
        self.last_error = "" if ok else "Air pressure out of range"
        self.emit("safety", estop=self.estop, air_ok=ok)

    def _lose_reference(self) -> None:
        if self.state is not None:
            self.state.homed = False

    # -- tools -------------------------------------------------------------------------

    def tool_name(self) -> str:
        if self._display_tool is not None:
            return self._display_tool
        state = self.state
        return state.tool.active_name if state is not None else "RefTool"

    def _tool_axis(self, tool_name: str) -> Vec3:
        state = self.state
        if state is not None:
            aligned = state.tool.alignment.get(tool_name)
            if aligned is not None:
                return self.to_machine_direction(aligned[0])  # alignment is in the client's CSY
        return (0.0, 0.0, 1.0) if state is None else self.to_machine_direction((0.0, 0.0, 1.0))

    def placement(self, tcp: Vec3, name: str | None = None) -> ToolPlacement:
        """Where the tool's pivot is when its tool centre point is at ``tcp``."""
        name = name or self.tool_name()
        spec = self.toolkit.spec(name)
        model = self.toolkit.model(name)
        rotation, axis, orientation = head_rotation(spec, self._tool_axis(name))
        offset = rotation[:3, :3] @ np.asarray(model.tip_offset)
        pivot = (tcp[0] - float(offset[0]), tcp[1] - float(offset[1]), tcp[2] - float(offset[2]))
        return ToolPlacement(name, spec, model, rotation, axis, pivot, orientation)

    def reach_alignment(
        self, name: str, primary: Vec3, secondary: Vec3 | None
    ) -> tuple[Vec3, Vec3 | None]:
        """Return the orientation the head of tool ``name`` gets for a requested one (client CSY).

        An angle beyond the range of the head is ``2505``; an indexing head snaps to its step, so
        the answer may differ from the request (``AlignTool`` then checks it against alpha).
        """
        spec = self.toolkit.spec(name)
        context = self.csy_context()
        try:
            reached = reach_orientation(spec, self.to_machine_direction(primary, context))
        except ValueError as error:
            raise ServerError(
                ErrorSeverity.ERROR, ErrorCode.ANGLE_OUT_OF_RANGE, "AlignTool", str(error)
            ) from None
        return self.to_client_direction(reached, context), secondary

    def alignment_volume(self, name: str) -> tuple[Vec3, float] | None:
        """Return the sphere that holds the tool in every alignment (``Tool.AlignmentVolume``).

        Its centre is the head's pivot, as a vector from the tool's reference point (in machine
        coordinates); its radius reaches the farthest point of the probe and stylus.
        """
        placement = self.placement(self._pos, name)
        centre = tuple(p - t for p, t in zip(placement.pivot, self._pos, strict=True))
        return (float(centre[0]), float(centre[1]), float(centre[2])), placement.model.reach

    def collision_volume(self, name: str) -> list[Obb] | None:
        """Oriented boxes along the tool's axes that cover it (``Tool.CollisionVolume``)."""
        placement = self.placement(self._pos, name)
        rotation = placement.rotation[:3, :3]
        tip = np.asarray(placement.model.tip_offset)
        axes = tuple(
            (float(rotation[0, k]), float(rotation[1, k]), float(rotation[2, k])) for k in range(3)
        )
        boxes = []
        for lo, hi in placement.model.collision_boxes():
            centre = rotation @ ((lo + hi) / 2.0 - tip)
            half = (hi - lo) / 2.0
            boxes.append(
                Obb(
                    (float(centre[0]), float(centre[1]), float(centre[2])),
                    (float(half[0]), float(half[1]), float(half[2])),
                    axes,  # type: ignore[arg-type]
                )
            )
        return boxes

    def head_shift(self, tcp: Vec3, placement: ToolPlacement) -> Vec3:
        """How far the quill is from where it was built for (reference tool, vertical head)."""
        ref = self.toolkit.reference_drop()
        return (
            placement.pivot[0] - tcp[0],
            placement.pivot[1] - tcp[1],
            placement.pivot[2] - (tcp[2] + ref),
        )

    def is_calibrated(self, name: str) -> bool:
        """Whether ``Tool.Alignment`` and ``Tool.AvrOffsets`` can answer (tactile: qualified)."""
        spec = self.toolkit.spec(name)
        return (
            spec.mode in OPTICAL_MODES
            or not self.require_qualification
            or any(q[0] == name for q in self.qualified)
        )

    def avr_radius(self, name: str) -> float:
        """Return the radius of the stylus ball; zero for optical tools (``AvrRadius``)."""
        spec = self.toolkit.spec(name)
        return 0.0 if spec.mode in OPTICAL_MODES or spec.mode == "none" else spec.ball_radius

    def avr_offsets(self, name: str) -> Vec3:
        """Return the tool centre point relative to the head pivot (``Tool.AvrOffsets``)."""
        return tip_offset(self.toolkit.spec(name))

    def is_qualified(self, name: str, orientation: tuple[float, float] | None) -> bool:
        return (name, orientation) in self.qualified

    def _optical_spec(self) -> OpticalSpec | None:
        """Return the active tool's optical sensor as a spec, if it has one."""
        spec = self.toolkit.spec(self.tool_name())
        kind = {"laser": "line", "point_laser": "point", "area": "area", "camera": "camera"}.get(
            spec.mode
        )
        if kind is None:
            return None
        return OpticalSpec(
            kind=kind,
            standoff=spec.standoff,
            depth_range=spec.depth_range,
            line_width=spec.line_width,
            field_height=spec.field_height,
            points_per_line=spec.points_per_line,
            line_pitch=spec.line_pitch,
            noise_mm=spec.noise_um * 1e-3,
            dropout=spec.dropout,
            max_angle_deg=spec.max_angle_deg,
            triangulation_deg=spec.triangulation_deg,
            point_type=spec.point_type,
            grid_pitch=spec.grid_pitch,
        )

    def depth_buffer(self, direction: Vec3) -> DepthBuffer:
        """Rasterise the scene along ``direction`` (towards the scene), cached per scene state."""
        rotary = round(self._current_rotary(), 6)
        key = (self.scene_version, rotary, tuple(round(c, 4) for c in normalize(direction)))
        buffer = self._buffers.get(key)
        if buffer is None:
            with self.lock:
                rot = self.machine.rotary_pose(rotary)
                vertices: list[np.ndarray] = []
                faces: list[np.ndarray] = []
                base = 0
                for obj in self.objects:
                    if not obj.visible or obj.fine_mesh.is_empty:
                        continue
                    mesh = obj.fine_mesh
                    vertices.append(geometry.apply(obj.world_pose(rot), mesh.vertices))
                    faces.append(mesh.faces + base)
                    base += len(mesh.vertices)
            buffer = DepthBuffer(
                np.vstack(vertices) if vertices else np.zeros((0, 3)),
                np.vstack(faces) if faces else np.zeros((0, 3), dtype=np.int64),
                np.asarray(direction, dtype=float),
            )
            if len(self._buffers) >= 4:
                self._buffers.pop(next(iter(self._buffers)))
            self._buffers[key] = buffer
        return buffer

    # -- measuring ---------------------------------------------------------------------

    def _current_rotary(self) -> float:
        state = self.state
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

    def measure(self, origin: Vec3, direction: Vec3, *, mode: str | None = None) -> Vec3 | None:
        """Probe along the ray from ``origin`` against ``direction`` and return the measured point.

        The point carries the errors of the machine and the tool: the length-dependent
        measuring error, probe repeatability, pre-travel lobing, stylus bending, a
        systematic error while the tool is not qualified, and thermal expansion.
        """
        unit = normalize(direction)
        hit = self.cast(origin, unit)
        if hit is None:
            return None
        point, _distance, obj = hit
        name = self.tool_name()
        spec = self.toolkit.spec(name)
        if name in self.detached:
            raise ServerError(
                ErrorSeverity.CRITICAL,
                ErrorCode.PROBE_NOT_ARMED,
                "PtMeas",
                "The stylus module broke away in a crash; change the tool to re-seat it",
            )
        if self.noise_enabled:
            placement = self.placement(self._pos, name)
            point = self._apply_probe_errors(point, unit, spec, placement)
        point = self.apply_thermal(point, obj)
        self.record_contact(point, mode or spec.mode)
        return point

    def apply_thermal(self, point: Vec3, obj: SceneObject | None) -> Vec3:
        """Return ``point`` as the machine reports it with its thermal errors (6.5.2, 6.24.2).

        The part grows with its own temperature about its origin; the server undoes that with the
        ``Part.Temperature`` and ``Part.XpanCoefficient`` the client set (the coefficient is zero
        until it does). The scales grow with the room temperature and read short; the server
        corrects them with the scale temperatures it knows (``UpdateScaleTemperatures``,
        ``SetScaleTemperatures``; 20 until then).
        """
        state = self.state
        x, y, z = point
        if obj is not None:
            ref = obj.pose[:3, 3]
            grown = 1.0 + self.part_cte * (self.temperature - 20.0)
            known_temperature = 20.0
            known_cte = 0.0
            if state is not None:
                known_temperature = state.part.properties.get("Part.Temperature", 20.0)
                known_cte = state.part.properties.get("Part.XpanCoefficient", 0.0) * 1e-6
            factor = grown / (1.0 + known_cte * (known_temperature - 20.0))
            if abs(factor - 1.0) > 1e-12:
                x, y, z = (
                    float(ref[0]) + (x - ref[0]) * factor,
                    float(ref[1]) + (y - ref[1]) * factor,
                    float(ref[2]) + (z - ref[2]) * factor,
                )
        scales = []
        for value, axis in zip((x, y, z), "XYZ", strict=True):
            known = state.mover.scale_temperatures.get(axis, 20.0) if state is not None else 20.0
            shrunk = (1.0 + self.scale_cte * (known - 20.0)) / (
                1.0 + self.scale_cte * (self.ambient_temperature - 20.0)
            )
            scales.append(value * shrunk)
        return (scales[0], scales[1], scales[2])

    def _apply_probe_errors(
        self, point: Vec3, unit: Vec3, spec: ToolSpec, placement: ToolPlacement
    ) -> Vec3:
        """Machine and probe errors on a probed point, along the probing direction."""
        accuracy = self.machine.spec.accuracy
        sigma = accuracy.sigma_mm(float(np.linalg.norm(point)))
        sigma *= 1.0 + spec.shaft_length / 150.0  # a longer stylus bends more
        error = self.rng.gauss(0.0, sigma)
        if spec.repeatability_um:
            error += self.rng.gauss(0.0, spec.repeatability_um * 1e-3)
        if spec.lobing_um:
            axis = np.asarray(placement.axis)
            u = np.asarray(unit)
            along = float(u @ axis)
            lateral = u - along * axis
            sin_phi = float(np.linalg.norm(lateral))
            if sin_phi > 1e-9:
                helper = (
                    np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
                )
                e1 = np.cross(axis, helper)
                e1 /= np.linalg.norm(e1)
                e2 = np.cross(axis, e1)
                theta = math.atan2(float(lateral @ e2), float(lateral @ e1))
                error += spec.lobing_um * 1e-3 * sin_phi * math.cos(3.0 * theta)
        if self.require_qualification and not self.is_qualified(spec.name, placement.orientation):
            seed = f"{self._seed}:{spec.name}:{placement.orientation}"
            error += random.Random(seed).gauss(0.0, spec.unqualified_um * 1e-3)  # noqa: S311 # nosec B311
        return add(point, scale(unit, error))

    def record_contact(self, point: Vec3, mode: str = "touch") -> None:
        self.contacts.append(point)
        self.points.setdefault(mode, deque(maxlen=200_000)).append(point)
        self.emit("contact", point=point, mode=mode)

    def record_scan_points(self, points: list[Vec3], mode: str = "scanning") -> None:
        bucket = self.points.setdefault(mode, deque(maxlen=200_000))
        bucket.extend(points)
        self.emit("scan", points=len(points), mode=mode)

    def _record_cloud(self, cloud: PointCloudSet) -> None:
        points = np.array(
            [[p.x, p.y, p.z] for c in cloud.point_clouds for ps in c.point_sets for p in ps.points]
        ).reshape(-1, 3)
        self.clouds.append(points)
        mode = self.toolkit.spec(self.tool_name()).mode
        mode = mode if mode in OPTICAL_MODES else "laser"
        bucket = self.points.setdefault(mode, deque(maxlen=200_000))
        bucket.extend((float(row[0]), float(row[1]), float(row[2])) for row in points)
        self.emit("scan", points=len(points), mode=mode)

    def artifact(self) -> SceneObject | None:
        """Return the first object in the scene that carries an artefact description."""
        return next((o for o in self.objects if o.artifact and len(o.artifact.features) > 1), None)

    def evaluate(self, obj: SceneObject | None = None) -> dict[str, list[Result]]:
        """Evaluate the points measured so far, per mode, against a check artefact."""
        obj = obj or self.artifact()
        if obj is None or obj.artifact is None:
            return {}
        rotary = self.machine.rotary_pose(self._current_rotary())
        data = obj.artifact.placed(obj.world_pose(rotary))
        accuracy = self.machine.spec.accuracy
        out: dict[str, list[Result]] = {}
        for mode, points in self.points.items():
            if not points:
                continue
            array = np.asarray(list(points), dtype=np.float64)
            results = evaluate_features(data.features, data.lengths, array, accuracy)
            if results:
                out[mode] = results
        return out

    def reference_sphere(self) -> tuple[Vec3, float, SceneObject] | None:
        """Centre and radius of the first qualification sphere, in machine coordinates."""
        rotary = self.machine.rotary_pose(self._current_rotary())
        for obj in self.objects:
            if obj.artifact is None or obj.artifact.reference is None:
                continue
            data = obj.artifact.placed(obj.world_pose(rotary))
            feature = next(f for f in data.features if f.name == data.reference)
            center = getattr(feature, "center", None)
            radius = getattr(feature, "radius", None)
            if center is not None and radius is not None:
                return (float(center[0]), float(center[1]), float(center[2])), float(radius), obj
        return None

    # -- motion ------------------------------------------------------------------------

    def protocol_parameters(self) -> dict[str, float]:
        """Speeds and distances the client set through ``GoToPar``/``PtMeasPar``/``ScanPar``."""
        spec = self.machine.spec
        values = {
            "go_speed": spec.max_speed,
            "go_accel": spec.acceleration,
            "probe_speed": spec.probing_speed,
            "approach": 5.0,
            "search": 2.0,
            "scan_speed": spec.scanning_speed,
        }
        state = self.state
        if state is not None:
            p = state.tool.parameters
            values["go_speed"] = min(p.go_to_par["Speed"].value, spec.max_speed)
            values["go_accel"] = min(p.go_to_par["Accel"].value, spec.acceleration)
            values["probe_speed"] = p.pt_meas_par["Speed"].value
            values["approach"] = p.pt_meas_par["Approach"].value
            values["search"] = p.pt_meas_par["Search"].value
            values["scan_speed"] = min(p.scan_par["Speed"].value, spec.scanning_speed)
        return values

    def publish_motion(self, position: Vec3, rotary: float) -> None:
        self._pos = position
        self._rotary = rotary
        self._published = time.monotonic()

    def report_touch(self, what: str, at: Vec3) -> tuple[str, str]:
        """Note that the tip touched ``what`` during a move; return the error code and its text.

        A touch probe triggers (illegal touch, 1001); a measuring probe is pushed beyond its
        range (head error excessive force, 2001). Neither breaks the stylus away.
        """
        spec = self.toolkit.spec(self.tool_name())
        if spec.mode in ("scanning", "head_touch"):
            code, text = (
                ErrorCode.HEAD_ERROR_EXCESSIVE_FORCE,
                f"Excessive force on the head at {what}",
            )
        else:
            code, text = ErrorCode.ILLEGAL_TOUCH, f"The probe touched {what} during a move"
        self.last_error = text
        self.emit("touch", what=what, at=at)
        return str(code), text

    def report_collision(self, what: str, at: Vec3) -> None:
        name = self.tool_name()
        self.last_error = f"Collision with {what}"
        if self.toolkit.spec(name).breakaway:
            self.detached.add(name)
            self.last_error += "; the stylus module broke away"
        self.emit("collision", what=what, at=at)

    async def run_leg(
        self,
        start: Vec3,
        end: Vec3,
        speed: float,
        accel: float,
        cancel: asyncio.Event,
        rotary: tuple[float, float] | None = None,
        cause: str = "Move",
        v_start: float = 0.0,
        v_end: float = 0.0,
    ) -> Vec3:
        """Move in a straight line on a trapezoidal profile; returns where the machine stopped.

        The leg starts and ends at rest unless ``v_start``/``v_end`` say otherwise (the approach
        of a probing move ends at the probing speed, the search starts at it).
        """
        rotary = rotary or (self._rotary, self._rotary)
        distance = norm(sub(end, start))
        duration = profile_time(distance, speed, accel, v_start, v_end)
        if rotary[0] != rotary[1]:
            duration = max(
                duration,
                travel_time(
                    abs(rotary[1] - rotary[0]),
                    self.machine.spec.rotary_speed,
                    self.machine.spec.rotary_speed * 4.0,
                ),
            )
        position = start
        if self.time_scale > 0.0 and duration > 0.0:
            loop = asyncio.get_running_loop()
            started = loop.time()
            while not cancel.is_set():
                self._raise_if_unready(cause, position)
                t = (loop.time() - started) * self.time_scale
                if t >= duration:
                    break
                along = (
                    profile_position(t, distance, speed, accel, v_start, v_end) / distance
                    if distance
                    else 0.0
                )
                position = add(start, scale(sub(end, start), min(along, 1.0)))
                self.publish_motion(position, rotary[0] + (rotary[1] - rotary[0]) * t / duration)
                await asyncio.sleep(_FRAME)
            if cancel.is_set():
                self.publish_motion(position, rotary[0])
                return position
        self._raise_if_unready(cause, position)
        self.publish_motion(end, rotary[1])
        return end

    def _raise_if_unready(self, cause: str, position: Vec3) -> None:
        try:
            self.check_ready(cause)
        except ServerError as error:
            self.publish_motion(position, self._rotary)
            raise MotionError(
                error.severity, error.number, error.cause, error.text, position
            ) from None

    async def dwell(self, seconds: float, cancel: asyncio.Event, cause: str = "Move") -> None:
        """Stand still for ``seconds`` of machine time (shortened by the playback speed)."""
        if self.time_scale <= 0.0 or seconds <= 0.0:
            return
        loop = asyncio.get_running_loop()
        end = loop.time() + seconds / self.time_scale
        while loop.time() < end and not cancel.is_set():
            self._raise_if_unready(cause, self._pos)
            self.publish_motion(self._pos, self._rotary)
            await asyncio.sleep(_FRAME)

    async def run_path(
        self, waypoints: list[Vec3], speed: float, accel: float, cancel: asyncio.Event, cause: str
    ) -> None:
        for a, b in itertools.pairwise(waypoints):
            await self.run_leg(a, b, speed, accel, cancel, cause=cause)

    def surface_normal(self, point: Vec3, hint: Vec3) -> Vec3:
        """Estimate the surface normal at ``point`` from three nearby rays; faces along ``hint``."""
        h = np.asarray(normalize(hint))
        helper = np.array([1.0, 0.0, 0.0]) if abs(h[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        t1 = np.cross(h, helper)
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(h, t1)
        p = np.asarray(point)
        hits = []
        for offset in (np.zeros(3), t1 * 0.2, t2 * 0.2):
            origin = p + offset + h * 2.0
            hit = self.cast(
                (float(origin[0]), float(origin[1]), float(origin[2])), (-h[0], -h[1], -h[2])
            )
            if hit is None:
                return normalize(hint)
            hits.append(np.asarray(hit[0]))
        n = np.cross(hits[1] - hits[0], hits[2] - hits[0])
        length = float(np.linalg.norm(n))
        if length < 1e-12:
            return normalize(hint)
        n = n / length
        if float(n @ h) < 0:
            n = -n
        return (float(n[0]), float(n[1]), float(n[2]))

    @property
    def position(self) -> Vec3:
        """Where the tool centre point last was, in machine coordinates."""
        return self._pos

    async def run_probe(self, request: ProbeRequest, start: Vec3) -> ProbeResult:
        """Run the probing cycle of ``PtMeas`` (6.12.1): search, trigger, overtravel, retract.

        From the approach position the machine moves towards the surface at the search speed
        for up to ``approach + search``. The ball triggers when it touches the surface, which
        for a tilted surface is later along the probing line than for a normal one; the machine
        runs on for its stopping distance, the point is latched at the trigger and compensated
        for the ball radius along the probing direction, and the machine retracts. If nothing
        is touched the machine returns to the approach position and the command fails with 1006.
        """
        cause = request.cause
        self.check_ready(cause)
        name = request.tool_name
        spec = self.toolkit.spec(name)
        if name in self.detached:
            raise ServerError(
                ErrorSeverity.CRITICAL,
                ErrorCode.PROBE_NOT_ARMED,
                cause,
                "The stylus module broke away in a crash; change the tool to re-seat it",
            )
        unit = request.direction
        toward = scale(unit, -1.0)
        speed = max(spec.touch_speed or request.speed, 1e-3) * (
            1.0 if spec.touch_speed else self.speed_override
        )
        accel = max(request.accel, 1e-3)
        radius = spec.ball_radius
        reach = request.approach - radius + request.search
        hit = self.cast(start, toward)
        trigger = None
        if hit is not None:
            normal = self.surface_normal(hit[0], unit)
            cos_theta = max(float(np.dot(normal, unit)), 0.05)
            trigger = max(hit[1] - radius / cos_theta, 0.0)
        found = trigger is not None and trigger <= reach + 1e-6
        travel = trigger if found and trigger is not None else reach
        end = add(start, scale(toward, travel))
        # The stylus and probe body must not hit anything on the way; the ball is the sensor.
        stop, collided = self.first_collision(
            MotionRequest(
                cause, start, end, self._rotary, self._rotary, True, name, request.cancel
            ),
            probing=True,
        )
        if collided is not None:
            await self.run_leg(start, stop, speed, accel, request.cancel, cause=cause)
            self.report_collision(collided, stop)
            raise MotionError(
                ErrorSeverity.CRITICAL,
                ErrorCode.COLLISION,
                cause,
                f"Collision with {collided}",
                stop,
            )
        go = self.protocol_parameters()
        # The search runs at the probing speed from the approach point, without a stop, to the
        # trigger; the machine then runs on for its stopping distance (overtravel, the stylus
        # deflects meanwhile) and comes to rest (Figure 29).
        over = min(speed * speed / (2.0 * accel), 0.5)
        runs_to = add(end, scale(toward, over)) if found and hit is not None else end
        position = await self.run_leg(
            start, runs_to, speed, accel, request.cancel, cause=cause, v_start=speed, v_end=0.0
        )
        if request.cancel.is_set():
            return ProbeResult(start, position)
        if not found or hit is None:
            await self.run_leg(
                position, start, go["go_speed"], go["go_accel"], request.cancel, cause=cause
            )
            raise ServerError(
                ErrorSeverity.ERROR, ErrorCode.SURFACE_NOT_FOUND, cause, "Surface not found"
            )
        stopped = runs_to
        # The point latched at the trigger, compensated along the probing direction.
        centre = end
        measured = add(centre, scale(toward, radius))
        if self.noise_enabled:
            measured = self._apply_probe_errors(measured, unit, spec, self.placement(centre, name))
        obj = hit[2]
        measured = self.apply_thermal(measured, obj)
        self.record_contact(measured, spec.mode)
        rest = start if request.retract < 0 else add(centre, scale(unit, request.retract))
        # The retract runs at the GoTo speed (Figure 29: the speed goes negative up to V_goto).
        await self.run_leg(
            stopped, rest, go["go_speed"], go["go_accel"], request.cancel, cause=cause
        )
        return ProbeResult(measured, rest)

    async def run_home(self, cancel: asyncio.Event) -> None:
        """Referencing: raise Z, move X and Y to their reference marks, lower Z."""
        try:
            self.check_ready("Home")
        except ServerError as error:
            raise ServerError(
                ErrorSeverity.CRITICAL, ErrorCode.ERROR_DURING_HOME, "Home", error.text
            ) from None
        spec = self.machine.spec
        speed, accel = min(spec.max_speed * 0.2, 100.0), spec.acceleration * 0.5
        path = plan_homing(self._pos, spec.travel)
        await self.run_path(path, speed * self.speed_override, accel, cancel, "Home")
        self.publish_motion(path[-1], 0.0)

    async def run_tool_change(
        self, current: str, target: str, position: Vec3, cancel: asyncio.Event
    ) -> None:
        """Drive to the rack, put the old module back, take the new one, return."""
        self.check_ready("ChangeTool")
        self.detached.discard(target)  # a module that broke away is seated again by the change
        if current == target:
            return
        spec = self.machine.spec
        slots = self.machine.rack_slots()
        old = self.toolkit.spec(current)
        new = self.toolkit.spec(target)
        if self.time_scale <= 0.0:
            self._pos = position
            return
        self._changing = target
        try:
            stages = plan_tool_change(
                position,
                slots.get(old.rack_key) if old.mode != "none" else None,
                slots.get(new.rack_key) if new.mode != "none" else None,
                spec.travel,
                dwell=_MODULE_CHANGE_DWELL,
            )
            speed, accel = spec.max_speed * 0.6 * self.speed_override, spec.acceleration
            for stage in stages:
                await self.run_path(list(stage.waypoints), speed, accel, cancel, "ChangeTool")
                await self.dwell(stage.dwell, cancel, "ChangeTool")
                if stage.action == "release":
                    self._display_tool = "NoTool"
                    self.emit("tool_change", stage="released", tool=current)
                elif stage.action == "take":
                    self._display_tool = target
                    self.emit("tool_change", stage="taken", tool=target)
            self._display_tool = target
        finally:
            self._changing = ""
            self._display_tool = None

    async def run_qualification(self, name: str, cancel: asyncio.Event) -> None:
        """Qualify a tool on the reference sphere: five touches, then its errors are known."""
        self.check_ready("ReQualify")
        spec = self.toolkit.spec(name)
        if spec.mode == "none":
            raise ServerError(
                ErrorSeverity.CRITICAL, ErrorCode.TOOL_NOT_DEFINED, "ReQualify", "Tool not defined"
            )
        if spec.mode == "laser":
            raise ServerError(
                ErrorSeverity.ERROR,
                ErrorCode.PROBE_TYPE_NOT_ALLOWED,
                "ReQualify",
                "A laser scanner is not qualified on a sphere",
            )
        if name in self.detached:
            raise ServerError(
                ErrorSeverity.CRITICAL,
                ErrorCode.PROBE_NOT_ARMED,
                "ReQualify",
                "The stylus module broke away; change the tool to re-seat it",
            )
        sphere = self.reference_sphere()
        if sphere is None:
            raise ServerError(
                ErrorSeverity.ERROR,
                ErrorCode.SURFACE_NOT_FOUND,
                "ReQualify",
                "There is no reference sphere in the measuring volume",
            )
        center, radius, _ = sphere
        if self.time_scale > 0.0:
            params = self.protocol_parameters()
            path = plan_qualification(self._pos, center, radius, spec.ball_radius)
            await self.run_path(
                path,
                max(params["probe_speed"], 5.0),
                self.machine.spec.acceleration * 0.25,
                cancel,
                "ReQualify",
            )
            await self.dwell(0.5, cancel, "ReQualify")
        placement = self.placement(self._pos, name)
        self.qualified.add((name, placement.orientation))
        self.emit("qualified", tool=name, orientation=placement.orientation)

    # -- collisions --------------------------------------------------------------------

    @staticmethod
    def _overlap(a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]) -> bool:
        return bool(np.all(a[0] <= b[1] + 0.05) and np.all(b[0] <= a[1] + 0.05))

    @staticmethod
    def _aabb(lo: np.ndarray, hi: np.ndarray, m: Matrix) -> tuple[np.ndarray, np.ndarray]:
        import itertools

        corners = np.array(list(itertools.product(*zip(lo, hi, strict=True))))
        w = geometry.apply(m, corners)
        return w.min(axis=0), w.max(axis=0)

    @staticmethod
    def _box_gap(a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]) -> float:
        """Distance bound between two boxes: the largest gap along any axis (0 if they overlap)."""
        return float(max(0.0, np.max(a[0] - b[1]), np.max(b[0] - a[1])))

    def first_collision(self, request: MotionRequest, *, probing: bool) -> tuple[Vec3, str | None]:
        """Walk the move in small steps; return where the machine stops and what it hit.

        The step is adaptive: the clearance to the nearest obstacle bounds how far the tool can
        move without touching anything, so the walk skips ahead through free space.
        """
        tool = self.toolkit.spec(request.tool_name)
        delta = tuple(e - s for s, e in zip(request.start, request.end, strict=True))
        length = math.sqrt(sum(d * d for d in delta))
        steps = collision_steps(length, tool.ball_radius)
        step_length = length / steps if steps else 0.0
        offset: Vec3 | None = None
        if probing and length > 0.0:
            offset = scale(normalize(delta), -(tool.ball_radius + 0.05))  # type: ignore[arg-type]
        rotating = request.rotary_end != request.rotary_start
        previous = request.start
        with self.lock:
            # What the tool already touches at the start (it rests on a probed surface) does not
            # stop it from moving away; the exemption ends once it is clear of that obstacle.
            ignored, _ = self._collisions(
                request.start, request.rotary_start, request.tool_name, offset
            )
            i = 1
            while i <= steps:
                f = i / steps
                pos = point_along(request.start, request.end, f)
                rotary = request.rotary_start + (request.rotary_end - request.rotary_start) * f
                hits, clearance = self._collisions(pos, rotary, request.tool_name, offset)
                fresh = hits - ignored
                if fresh:
                    name = sorted(fresh)[0]
                    # Where does the contact begin? The step that found it may already have
                    # pushed the stylus in; the first point of contact says what touched first.
                    low, high = f - 1.0 / steps if steps else 0.0, f
                    for _ in range(10):
                        mid = (low + high) / 2.0
                        mid_hits, _ = self._collisions(
                            point_along(request.start, request.end, mid),
                            request.rotary_start
                            + (request.rotary_end - request.rotary_start) * mid,
                            request.tool_name,
                            offset,
                        )
                        if mid_hits - ignored:
                            high = mid
                        else:
                            low = mid
                    self.last_hit_was_touch = self._tip_only(
                        point_along(request.start, request.end, high),
                        request.rotary_start + (request.rotary_end - request.rotary_start) * high,
                        request.tool_name,
                        offset,
                        name,
                    )
                    return previous, name
                ignored &= hits
                previous = pos
                i += 1 if rotating or hits else free_steps(clearance, step_length)
        return request.end, None

    def _tip_only(
        self, tcp: Vec3, rotary: float, tool_name: str, probing_offset: Vec3 | None, name: str
    ) -> bool:
        """Whether only the tip ball of the stylus touches the object ``name`` (a probe trigger)."""
        obj = next((o for o in self.objects if o.name == name), None)
        if obj is None:
            return False  # a part of the machine: the tool hit more than a surface
        if probing_offset is not None:
            tcp = add(tcp, probing_offset)
        placement = self.placement(tcp, tool_name)
        fixed, turning = placement.model.placed(placement.pivot, placement.rotation, with_tip=False)
        body = cad.make_compound([fixed, turning])
        world = obj.world_pose(self.machine.rotary_pose(rotary))
        return bool(cad.min_distance(body, cad.moved(obj.shape, world)) >= 0.01)

    def _collisions(
        self, tcp: Vec3, rotary: float, tool_name: str, probing_offset: Vec3 | None
    ) -> tuple[set[str], float]:
        """Names of what the tool or machine touches at ``tcp``, and the clearance to the rest."""
        found: set[str] = set()
        clearance = math.inf
        rot_pose = self.machine.rotary_pose(rotary)
        if probing_offset is not None:
            # The protocol puts the TCP on the surface being touched (the tip radius is taken as
            # zero); at contact the ball centre sits one ball radius back along the probing
            # direction, so the stylus is placed there and just touches the surface.
            tcp = add(tcp, probing_offset)
        placement = self.placement(tcp, tool_name)
        pivot = placement.pivot
        box = placement.model.aabb(pivot, placement.rotation)
        shapes: tuple[cad.Shape, cad.Shape] | None = None

        def tool_shape() -> cad.Shape:
            nonlocal shapes
            if shapes is None:
                shapes = placement.model.placed(pivot, placement.rotation)
            return cad.make_compound(list(shapes))

        def check(
            name: str, other_box: tuple[np.ndarray, np.ndarray], shape: cad.Shape, pose: Matrix
        ) -> None:
            nonlocal clearance
            gap = self._box_gap(box, other_box)
            if gap > 0.05:
                clearance = min(clearance, gap)
                return
            distance = cad.min_distance(tool_shape(), cad.moved(shape, pose))
            clearance = min(clearance, distance)
            if distance < 0.01:
                found.add(name)

        shift = self.head_shift(tcp, placement)
        for body in self.machine.bodies:
            if not body.spec.collides or self.machine.carried_axes(body):
                continue
            pose = self.machine.body_pose(body, tcp, rotary, shift)
            blo, bhi = body.bounds
            check(body.spec.name, self._aabb(np.array(blo), np.array(bhi), pose), body.shape, pose)
        for obj in self.objects:
            if not obj.visible:
                continue
            world = obj.world_pose(rot_pose)
            obox = obj.world_bounds(rot_pose)
            check(obj.name, obox, obj.shape, world)
            # Carrying parts of the machine against the sample and fixtures.
            for body in self.machine.bodies:
                if not body.spec.collides or not self.machine.carried_axes(body):
                    continue
                pose = self.machine.body_pose(body, tcp, rotary, shift)
                blo, bhi = body.bounds
                bbox = self._aabb(np.array(blo), np.array(bhi), pose)
                gap = self._box_gap(bbox, obox)
                if gap > 0.05:
                    clearance = min(clearance, gap)
                    continue
                distance = cad.min_distance(
                    cad.moved(body.shape, pose), cad.moved(obj.shape, world)
                )
                clearance = min(clearance, distance)
                if distance < 0.01:
                    found.add(f"{body.spec.name} / {obj.name}")
        return found, clearance

    # -- presentation ------------------------------------------------------------------

    def snapshot(self) -> TwinSnapshot:
        state = self.state
        moving = time.monotonic() - self._published < 0.25
        tool_name = self.tool_name()
        homed = False
        error = self.last_error
        session = False
        machine_class = ""
        position, rotary = self._pos, self._rotary
        if state is not None:
            homed = state.homed
            session = state.session_active
            machine_class = str(state.machine_class)
            if state.active_error is not None:
                error = str(state.active_error)
            elif not moving and not self.estop and self.air_ok:
                error = ""
            if not moving:
                position = self.to_machine(state.cart_cmm.position)
                rotary = state.rotary_table.position
                self._pos, self._rotary = position, rotary
        spec = self.toolkit.spec(tool_name)
        placement = self.placement(position, tool_name)
        return TwinSnapshot(
            position=position,
            rotary=rotary,
            tool_name=tool_name,
            tool_mode=spec.mode,
            tool_axis=placement.axis,
            head_position=placement.orientation,
            homed=homed,
            moving=moving,
            connected=bool(self._connected),
            peer=self._connected,
            error=error,
            session_active=session,
            machine_class=machine_class,
            estop=self.estop,
            air_ok=self.air_ok,
            qualified=(not self.require_qualification)
            or self.is_qualified(tool_name, placement.orientation),
            detached=tool_name in self.detached,
            changing_tool=self._changing,
            world_shift=self.machine.world_shift(position),
            active_csy=self.state.cart_cmm.active_csy if self.state is not None else "MachineCsy",
        )

    def draw_items(self) -> list[DrawItem]:
        """Everything to draw now, with current poses."""
        snap = self.snapshot()
        rotary = self.machine.rotary_pose(snap.rotary)
        placement = self.placement(snap.position, snap.tool_name)
        shift = self.head_shift(snap.position, placement)
        items = [
            DrawItem(
                f"machine:{i}:{b.spec.name}",
                b.spec.name,
                b.mesh,
                self.machine.body_pose(b, snap.position, snap.rotary, shift),
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
        base = geometry.translation(*placement.pivot)
        for part in placement.model.parts:
            pose = base @ placement.rotation if part.rotates else base
            color = part.color
            if part.tip and snap.detached:
                color = (0.3, 0.3, 0.3)
            items.append(DrawItem(f"tool:{part.name}", part.name, part.mesh, pose, color, "tool"))
        items.extend(self._stored_tool_items(placement.spec.rack_key))
        if any(snap.world_shift):
            lift = geometry.translation(*snap.world_shift)
            items = [replace(item, pose=lift @ item.pose) for item in items]
        return items

    def _stored_tool_items(self, mounted_key: str) -> list[DrawItem]:
        """Return the probe modules waiting in the rack, standing on their ports, tip up."""
        slots = self.machine.rack_slots()
        flip = geometry.rotation((1.0, 0.0, 0.0), 180.0)
        items: list[DrawItem] = []
        shown: set[str] = set()
        for spec in self.toolkit.specs.values():
            key = spec.rack_key
            if spec.mode == "none" or key in shown or key not in slots or key == mounted_key:
                continue
            shown.add(key)
            sx, sy, sz = slots[key]
            pose = geometry.translation(sx, sy, sz) @ flip
            for part in self.toolkit.model(spec.name).parts:
                if part.rotates:
                    items.append(
                        DrawItem(
                            f"stored:{key}:{part.name}",
                            part.name,
                            part.mesh,
                            pose,
                            part.color,
                            "stored",
                        )
                    )
        return items

    def bounds(self) -> tuple[Vec3, Vec3]:
        """Return the measuring volume, for framing the view."""
        t = self.machine.spec.travel
        dx, dy, dz = self.machine.world_shift(self._pos)
        return (dx, dy, dz), (t[0] + dx, t[1] + dy, t[2] + dz)

    def tool_drop(self, name: str) -> float:
        return drop(self.toolkit.spec(name))
