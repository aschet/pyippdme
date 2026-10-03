# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""A multi-feature check artefact for evaluating every measurement mode of the twin.

Real CMMs are verified with calibrated artefacts (ISO 10360): a test sphere for
the probing error, a step gauge or gauge blocks for the length measuring error,
ring gauges, ball plates, hole plates, and purpose-made test pieces. Commercial
"CMM check" artefacts combine a step gauge, a setting ring and a sphere. This one
puts a comparable set on a plate: a qualification sphere, a ring gauge, a pin, a
cone, a five-block step gauge, a stepped wedge with 15/30/45 degree faces, a ball
plate, a freeform dome for scanning, and a groove. Its geometry is generated,
its nominal features are known exactly, and :class:`ArtifactData` carries
measurement plans so that touch, scanning, head-touch and laser data can be taken
and evaluated with :mod:`pyippdme.twin.features`.

The plate's lower front-left corner is the artefact origin; its top face is at
``z = 25``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from pyippdme.twin import cad, geometry
from pyippdme.twin.features import Cone, Cylinder, Feature, Length, Plane, Sphere
from pyippdme.twin.geometry import Matrix
from pyippdme.twin.objects import SceneObject
from pyippdme.types.vec3 import Vec3

PLATE = (360.0, 220.0, 25.0)
TOP = PLATE[2]
SPHERE_CENTER = (40.0, 45.0, TOP + 60.0 + 12.5)
SPHERE_RADIUS = 12.5
RING = (140.0, 45.0)  # centre x, y; bore radius 25, outer radius 45, height 20
PIN = (225.0, 45.0)  # radius 12.5, height 40
CONE = (310.0, 45.0)  # base radius 18, top radius 9, height 35
STEP_X0 = 20.0
STEP_PITCH = 40.0
STEP_THICK = 10.0
STEP_Y = 115.0
STEP_WIDTH = 20.0
STEP_HEIGHT = 40.0
WEDGE = (250.0, 105.0)  # x, y origin; 30 mm wide in y
WEDGE_DEGREES = (15.0, 30.0, 45.0)
BALLS = [(30.0, 175.0), (80.0, 175.0), (30.0, 205.0), (80.0, 205.0)]
BALL_RADIUS = 7.5
DOME = (160.0, 180.0)  # footprint radius 40, sphere radius 150, peak 10 mm above the plate
DOME_R = 150.0
GROOVE = (260.0, 180.0)  # block 60 x 40 x 25 on the plate; groove 6 wide, 10 deep, along y


@dataclass(frozen=True, slots=True)
class PlanPoint:
    """A touch point: where the surface is and its normal pointing out of the material."""

    feature: str
    point: Vec3
    normal: Vec3
    #: Distance to approach from along the normal; ``None`` uses the program's default.
    clearance: float | None = None


@dataclass(frozen=True, slots=True)
class ScanLine:
    """A straight scan path: start, end, probing direction and point spacing."""

    start: Vec3
    end: Vec3
    normal: Vec3
    step: float


@dataclass(frozen=True, slots=True)
class ScanCircle:
    """A circular scan path around ``center``."""

    center: Vec3
    start: Vec3
    normal: Vec3
    delta: float
    step: float


@dataclass(frozen=True, slots=True)
class Sweep:
    """Control points for ``DataAcquire(..., Sweep, ...)`` and the sensor's ``primary`` vector."""

    positions: tuple[Vec3, ...]
    #: Anti-parallel to the tool axis, away from the surface (the sensor looks along -primary).
    direction: Vec3


@dataclass(slots=True)
class ArtifactData:
    """What an artefact knows about itself, in its own coordinates (see :meth:`placed`)."""

    features: list[Feature]
    lengths: list[Length]
    touch: list[PlanPoint] = field(default_factory=list)
    lines: list[ScanLine] = field(default_factory=list)
    circles: list[ScanCircle] = field(default_factory=list)
    sweeps: list[Sweep] = field(default_factory=list)
    #: Name of the qualification sphere feature, if any.
    reference: str | None = None
    #: Height of the artefact's base plate top and the box around the whole artefact.
    top: float = 0.0
    extent: tuple[Vec3, Vec3] = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

    def placed(self, pose: Matrix) -> ArtifactData:
        """Return the same data in machine coordinates, for an artefact placed with ``pose``."""

        def point(p: Vec3) -> Vec3:
            v = pose[:3, :3] @ np.asarray(p) + pose[:3, 3]
            return (float(v[0]), float(v[1]), float(v[2]))

        def direction(d: Vec3) -> Vec3:
            v = pose[:3, :3] @ np.asarray(d)
            return (float(v[0]), float(v[1]), float(v[2]))

        return ArtifactData(
            [f.transformed(pose) for f in self.features],
            self.lengths,
            [
                PlanPoint(t.feature, point(t.point), direction(t.normal), t.clearance)
                for t in self.touch
            ],
            [
                ScanLine(point(s.start), point(s.end), direction(s.normal), s.step)
                for s in self.lines
            ],
            [
                ScanCircle(point(c.center), point(c.start), direction(c.normal), c.delta, c.step)
                for c in self.circles
            ],
            [
                Sweep(tuple(point(p) for p in s.positions), direction(s.direction))
                for s in self.sweeps
            ],
            self.reference,
            self.top + float(pose[2, 3]),
            self._placed_extent(pose),
        )

    def _placed_extent(self, pose: Matrix) -> tuple[Vec3, Vec3]:
        lo, hi = self.extent
        corners = np.array(
            [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
        )
        world = corners @ pose[:3, :3].T + pose[:3, 3]
        mn, mx = world.min(axis=0), world.max(axis=0)
        return (float(mn[0]), float(mn[1]), float(mn[2])), (
            float(mx[0]),
            float(mx[1]),
            float(mx[2]),
        )


def _unit(*v: float) -> Vec3:
    n = math.sqrt(sum(a * a for a in v))
    return (v[0] / n, v[1] / n, v[2] / n)


def _hemisphere(center: Vec3, radius: float, count: int) -> list[tuple[Vec3, Vec3]]:
    """``count`` points spread over the upper hemisphere (golden spiral), with outward normals."""
    out = []
    for i in range(count):
        z = 1.0 - (i + 0.5) / count * 0.9  # from the pole to about 25 degrees above the equator
        phi = i * math.pi * (3.0 - math.sqrt(5.0))
        r = math.sqrt(1.0 - z * z)
        n = (r * math.cos(phi), r * math.sin(phi), z)
        out.append(
            (
                (center[0] + n[0] * radius, center[1] + n[1] * radius, center[2] + n[2] * radius),
                n,
            )
        )
    return out


def _wedge_top() -> list[tuple[float, float]]:
    """Top outline of the stepped wedge as (x, height above the plate), left to right."""
    x, z = 0.0, 10.0
    outline = [(x, z)]
    for degrees in WEDGE_DEGREES:
        x += 28.0 * math.cos(math.radians(degrees))
        z += 28.0 * math.sin(math.radians(degrees))
        outline.append((x, z))
    return outline


def build_shape() -> cad.Shape:
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCone
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt

    parts = [cad.rounded_box(*PLATE, radius=3.0)]
    # Qualification sphere on a post.
    parts.append(cad.make_cylinder(5.0, 60.0, (SPHERE_CENTER[0], SPHERE_CENTER[1], TOP)))
    parts.append(cad.make_sphere(SPHERE_RADIUS, SPHERE_CENTER))
    # Ring gauge (bore), pin, cone.
    parts.append(
        cad.cut(
            cad.make_cylinder(45.0, 20.0, (RING[0], RING[1], TOP)),
            cad.make_cylinder(25.0, 30.0, (RING[0], RING[1], TOP - 5.0)),
        )
    )
    parts.append(cad.make_cylinder(12.5, 40.0, (PIN[0], PIN[1], TOP)))
    parts.append(
        BRepPrimAPI_MakeCone(
            gp_Ax2(gp_Pnt(CONE[0], CONE[1], TOP), gp_Dir(0, 0, 1)), 18.0, 9.0, 35.0
        ).Shape()
    )
    # Step gauge: five blocks.
    for k in range(5):
        origin = (STEP_X0 + STEP_PITCH * k, STEP_Y - STEP_WIDTH / 2, TOP)
        parts.append(cad.make_box(STEP_THICK, STEP_WIDTH, STEP_HEIGHT, origin))
    # Stepped wedge: 15, 30 and 45 degree faces.
    outline = _wedge_top()
    profile = [(0.0, 0.0), (outline[-1][0], 0.0), *reversed(outline)]
    wedge = cad.make_prism_xz([(x + WEDGE[0], z + TOP) for x, z in profile], 30.0)
    parts.append(cad.moved(wedge, geometry.translation(0.0, WEDGE[1], 0.0)))
    # Ball plate.
    for bx, by in BALLS:
        parts.append(cad.make_cylinder(3.0, 30.0, (bx, by, TOP)))
        parts.append(cad.make_sphere(BALL_RADIUS, (bx, by, TOP + 30.0 + BALL_RADIUS * 0.5)))
    # Freeform dome.
    sphere = cad.make_sphere(DOME_R, (DOME[0], DOME[1], TOP + 10.0 - DOME_R))
    parts.append(cad.common(sphere, cad.make_cylinder(40.0, 40.0, (DOME[0], DOME[1], TOP))))
    # Groove block.
    block = cad.make_box(60.0, 40.0, 25.0, (GROOVE[0] - 30.0, GROOVE[1] - 20.0, TOP))
    groove = cad.make_box(6.0, 50.0, 10.0, (GROOVE[0] - 3.0, GROOVE[1] - 25.0, TOP + 15.0))
    parts.append(cad.cut(block, groove))
    shape = parts[0]
    for part in parts[1:]:
        shape = cad.fuse(shape, part)
    return shape


def build_data() -> ArtifactData:
    features: list[Feature] = []
    lengths: list[Length] = []
    touch: list[PlanPoint] = []
    up = np.array([0.0, 0.0, 1.0])

    def add_points(name: str, points: list[tuple[Vec3, Vec3]]) -> None:
        touch.extend(PlanPoint(name, p, n) for p, n in points)

    # Qualification sphere.
    features.append(Sphere("reference sphere", np.asarray(SPHERE_CENTER), SPHERE_RADIUS))
    add_points("reference sphere", _hemisphere(SPHERE_CENTER, SPHERE_RADIUS, 25))

    # Ring gauge and pin.
    z_ring = TOP + 10.0
    angles = [i * math.pi / 4 for i in range(8)]
    features.append(
        Cylinder("ring gauge", np.array([RING[0], RING[1], TOP]), up, 25.0, True, 2.0, 18.0)
    )
    add_points(
        "ring gauge",
        [
            (
                (RING[0] + 25.0 * math.cos(a), RING[1] + 25.0 * math.sin(a), z_ring),
                (-math.cos(a), -math.sin(a), 0.0),
            )
            for a in angles
        ],
    )
    features.append(Cylinder("pin", np.array([PIN[0], PIN[1], TOP]), up, 12.5, False, 5.0, 35.0))
    add_points(
        "pin",
        [
            (
                (PIN[0] + 12.5 * math.cos(a), PIN[1] + 12.5 * math.sin(a), TOP + 20.0),
                (math.cos(a), math.sin(a), 0.0),
            )
            for a in angles
        ],
    )
    # Cone.
    features.append(Cone("cone", np.array([CONE[0], CONE[1], TOP]), up, 18.0, 9.0, 35.0))
    slope = math.atan2(18.0 - 9.0, 35.0)
    r_mid = 13.5
    add_points(
        "cone",
        [
            (
                (CONE[0] + r_mid * math.cos(a), CONE[1] + r_mid * math.sin(a), TOP + 17.5),
                _unit(
                    math.cos(a) * math.cos(slope), math.sin(a) * math.cos(slope), math.sin(slope)
                ),
            )
            for a in angles
        ],
    )

    # Step gauge faces: left faces look to -x, right faces to +x.
    z_mid = TOP + STEP_HEIGHT / 2
    for k in range(5):
        x_left = STEP_X0 + STEP_PITCH * k
        for side, x, nx in (("left", x_left, -1.0), ("right", x_left + STEP_THICK, 1.0)):
            name = f"step {k + 1} {side}"
            features.append(
                Plane(
                    name,
                    np.array([x, STEP_Y, z_mid]),
                    np.array([nx, 0.0, 0.0]),
                    STEP_WIDTH / 2,
                    STEP_HEIGHT / 2,
                )
            )
            add_points(
                name,
                [
                    ((x, STEP_Y + dy, z_mid + dz), (nx, 0.0, 0.0))
                    for dy in (-6.0, 6.0)
                    for dz in (2.0, 14.0)
                ],
            )
    for k in range(1, 5):
        lengths.append(
            Length(
                f"step gauge {STEP_PITCH * k:g} mm",
                "step 1 left",
                f"step {k + 1} left",
                STEP_PITCH * k,
            )
        )
    for k in range(5):
        lengths.append(
            Length(
                f"step {k + 1} thickness", f"step {k + 1} left", f"step {k + 1} right", STEP_THICK
            )
        )

    # Stepped wedge faces.
    outline = _wedge_top()
    for i, degrees in enumerate(WEDGE_DEGREES):
        (xa, za), (xb, zb) = outline[i], outline[i + 1]
        mid = np.array([WEDGE[0] + (xa + xb) / 2, WEDGE[1] + 15.0, TOP + (za + zb) / 2])
        rad = math.radians(degrees)
        normal = (-math.sin(rad), 0.0, math.cos(rad))
        along = (math.cos(rad), 0.0, math.sin(rad))
        name = f"wedge {degrees:g} deg"
        features.append(Plane(name, mid, np.asarray(normal), 14.0, 15.0))
        add_points(
            name,
            [
                (
                    (
                        float(mid[0] + along[0] * s),
                        float(mid[1] + dy),
                        float(mid[2] + along[2] * s),
                    ),
                    normal,
                )
                for s in (-8.0, 8.0)
                for dy in (-8.0, 8.0)
            ],
        )

    # Ball plate.
    for i, (bx, by) in enumerate(BALLS):
        center = (bx, by, TOP + 30.0 + BALL_RADIUS * 0.5)
        features.append(Sphere(f"ball {i + 1}", np.asarray(center), BALL_RADIUS))
        add_points(f"ball {i + 1}", _hemisphere(center, BALL_RADIUS, 9))

    # Dome: a cap of a large sphere.
    dome_center = np.array([DOME[0], DOME[1], TOP + 10.0 - DOME_R])
    features.append(Sphere("dome", dome_center, DOME_R, up, 40.0))
    dome_points = []
    for r in (0.0, 15.0, 30.0):
        for a in (i * math.pi / 3 for i in range(6 if r else 1)):
            x, y = DOME[0] + r * math.cos(a), DOME[1] + r * math.sin(a)
            z = float(dome_center[2]) + math.sqrt(DOME_R**2 - r**2)
            n = _unit(
                x - float(dome_center[0]), y - float(dome_center[1]), z - float(dome_center[2])
            )
            dome_points.append(((x, y, z), n))
    add_points("dome", dome_points)

    # Groove: two walls.
    gz = TOP + 15.0
    for side, x, nx in (("left wall", GROOVE[0] - 3.0, 1.0), ("right wall", GROOVE[0] + 3.0, -1.0)):
        name = f"groove {side}"
        features.append(
            Plane(name, np.array([x, GROOVE[1], gz + 5.0]), np.array([nx, 0.0, 0.0]), 20.0, 5.0)
        )
        touch.extend(
            PlanPoint(name, (x, GROOVE[1] + dy, gz + 5.0), (nx, 0.0, 0.0), 2.5)
            for dy in (-10.0, 10.0)
        )
    lengths.append(Length("groove width", "groove left wall", "groove right wall", 6.0))

    data = ArtifactData(
        features,
        lengths,
        touch,
        reference="reference sphere",
        top=TOP,
        extent=((0.0, 0.0, 0.0), (PLATE[0], PLATE[1], 110.0)),
    )
    # Scans: lines over the dome; circles on the pin and in the bore.
    z_dome = TOP + 10.0 - 4.0
    for dy in (-20.0, 0.0, 20.0):
        data.lines.append(
            ScanLine(
                (DOME[0] - 36.0, DOME[1] + dy, z_dome),
                (DOME[0] + 36.0, DOME[1] + dy, z_dome),
                (0.0, 0.0, 1.0),
                2.0,
            )
        )
    data.circles.append(
        ScanCircle(
            (PIN[0], PIN[1], TOP + 20.0),
            (PIN[0] + 13.5, PIN[1], TOP + 20.0),
            (0.0, 0.0, 1.0),
            360.0,
            10.0,
        )
    )
    data.circles.append(
        ScanCircle(
            (RING[0], RING[1], z_ring),
            (RING[0] + 24.0, RING[1], z_ring),
            (0.0, 0.0, 1.0),
            360.0,
            10.0,
        )
    )
    # Laser: down-looking sweeps at the sensor's stand-off above the plate.
    for y in (25.0, 55.0):
        data.sweeps.append(Sweep(((10.0, y, TOP + 60.0), (350.0, y, TOP + 60.0)), (0.0, 0.0, 1.0)))
    return data


def build_check_artifact(name: str = "Check artefact") -> SceneObject:
    """Build the check artefact as a scene object; ``artifact`` holds the nominal data."""
    obj = SceneObject.from_shape(name, "sample", build_shape(), source="builtin check artefact")
    obj.artifact = build_data()
    obj.color = (0.55, 0.6, 0.65)
    return obj


def build_reference_sphere(name: str = "Reference sphere") -> SceneObject:
    """Build a qualification sphere (25 mm) on a post and a small base, to qualify tools on."""
    base = cad.rounded_box(60.0, 60.0, 15.0, (-30.0, -30.0, 0.0), 2.0)
    post = cad.make_cylinder(5.0, 70.0, (0.0, 0.0, 15.0))
    ball = cad.make_sphere(12.5, (0.0, 0.0, 15.0 + 70.0 + 6.0))
    shape = cad.fuse(cad.fuse(base, post), ball)
    obj = SceneObject.from_shape(name, "fixture", shape, source="builtin reference sphere")
    center = (0.0, 0.0, 91.0)
    obj.artifact = ArtifactData(
        [Sphere("reference sphere", np.asarray(center), 12.5)],
        [],
        [PlanPoint("reference sphere", p, n) for p, n in _hemisphere(center, 12.5, 25)],
        reference="reference sphere",
    )
    obj.color = (0.85, 0.35, 0.3)
    return obj
