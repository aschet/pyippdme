# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""The motion maths and path planners of the twin, as plain functions.

Nothing here needs CAD, Qt or a running simulation: every function takes numbers and
returns numbers or waypoints. :class:`~pyippdme.twin.twin.DigitalTwin` executes what
these functions plan, and a simulation of your own can use them the same way
(for a different UI framework, a different machine model, or a test).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pyippdme.types.vec3 import Vec3, add, scale

#: The step the collision walk takes along a move at most, in millimetres, and its cap.
COLLISION_STEP = 1.5
COLLISION_MAX_STEPS = 600


# -- velocity profile -------------------------------------------------------------------


def travel_time(distance: float, vmax: float, accel: float) -> float:
    """Duration of a trapezoidal (or, for short moves, triangular) velocity profile."""
    if distance <= 0.0:
        return 0.0
    ramp = vmax * vmax / accel
    if distance >= ramp:
        return distance / vmax + vmax / accel
    return 2.0 * math.sqrt(distance / accel)


def travelled(t: float, distance: float, vmax: float, accel: float) -> float:
    """Distance covered ``t`` seconds into the profile of :func:`travel_time`."""
    total = travel_time(distance, vmax, accel)
    if t >= total:
        return distance
    t_acc = min(vmax / accel, total / 2.0)
    v_peak = accel * t_acc
    if t < t_acc:
        return 0.5 * accel * t * t
    if t > total - t_acc:
        rest = total - t
        return distance - 0.5 * accel * rest * rest
    return 0.5 * accel * t_acc * t_acc + v_peak * (t - t_acc)


def _phases(
    distance: float, vmax: float, accel: float, v_start: float, v_end: float
) -> tuple[float, float, float, float, float, float]:
    """Split a leg into accelerate, cruise and decelerate: ``(vp, t_acc, d_acc, t_cru, d_cru, t_dec)``.

    The leg may start and end at any speed (``v_start``, ``v_end``): the machine does not stop
    at the approach point of a probing move, it slows down to the probing speed and goes on.
    """
    v_start, v_end = min(v_start, vmax), min(v_end, vmax)
    peak = math.sqrt(max((2.0 * accel * distance + v_start**2 + v_end**2) / 2.0, 0.0))
    vp = min(vmax, peak)
    if vp < max(v_start, v_end) - 1e-12:
        # Too short to reach the other speed: one uniform acceleration or deceleration.
        vp = max(v_start, v_end)
    d_acc = max((vp**2 - v_start**2) / (2.0 * accel), 0.0)
    d_dec = max((vp**2 - v_end**2) / (2.0 * accel), 0.0)
    if d_acc + d_dec > distance:  # only the degenerate case above gets here
        d_acc, d_dec = (distance, 0.0) if v_start < v_end else (0.0, distance)
    d_cru = max(distance - d_acc - d_dec, 0.0)
    t_acc = (vp - v_start) / accel if d_acc > 0 else 0.0
    t_dec = (vp - v_end) / accel if d_dec > 0 else 0.0
    t_cru = d_cru / vp if vp > 0 else 0.0
    return vp, t_acc, d_acc, t_cru, d_cru, t_dec


def profile_time(
    distance: float, vmax: float, accel: float, v_start: float = 0.0, v_end: float = 0.0
) -> float:
    """Duration of a leg that starts at ``v_start`` and ends at ``v_end`` (both 0: see travel_time)."""
    if distance <= 0.0:
        return 0.0
    _, t_acc, _, t_cru, _, t_dec = _phases(distance, vmax, accel, v_start, v_end)
    return t_acc + t_cru + t_dec


def profile_position(
    t: float,
    distance: float,
    vmax: float,
    accel: float,
    v_start: float = 0.0,
    v_end: float = 0.0,
) -> float:
    """Distance covered ``t`` seconds into the leg of :func:`profile_time`."""
    if distance <= 0.0 or t <= 0.0:
        return 0.0
    vp, t_acc, d_acc, t_cru, d_cru, t_dec = _phases(distance, vmax, accel, v_start, v_end)
    if t >= t_acc + t_cru + t_dec:
        return distance
    v0 = min(v_start, vmax)
    if t < t_acc:
        return v0 * t + 0.5 * (vp - v0) / t_acc * t * t if t_acc > 0 else 0.0
    if t < t_acc + t_cru:
        return d_acc + vp * (t - t_acc)
    rest = t - t_acc - t_cru
    slope = (vp - min(v_end, vmax)) / t_dec if t_dec > 0 else 0.0
    return d_acc + d_cru + vp * rest - 0.5 * slope * rest * rest


def point_along(start: Vec3, end: Vec3, fraction: float) -> Vec3:
    """The point ``fraction`` (0..1) of the way from ``start`` to ``end``."""
    return (
        start[0] + (end[0] - start[0]) * fraction,
        start[1] + (end[1] - start[1]) * fraction,
        start[2] + (end[2] - start[2]) * fraction,
    )


def path_length(waypoints: list[Vec3]) -> float:
    return sum(math.dist(a, b) for a, b in zip(waypoints, waypoints[1:], strict=False))


def path_duration(waypoints: list[Vec3], vmax: float, accel: float) -> float:
    """Time to drive through ``waypoints`` leg by leg, stopping at each (as the twin does)."""
    return sum(
        travel_time(math.dist(a, b), vmax, accel)
        for a, b in zip(waypoints, waypoints[1:], strict=False)
    )


# -- collision walk -----------------------------------------------------------------------


def collision_steps(length: float, ball_radius: float = 0.0) -> int:
    """How many samples the collision walk takes along a move of ``length`` millimetres."""
    return max(1, min(COLLISION_MAX_STEPS, math.ceil(length / max(COLLISION_STEP, ball_radius))))


def free_steps(clearance: float, step_length: float, margin: float = 0.5) -> int:
    """How many samples can be skipped when the nearest obstacle is ``clearance`` away.

    The tool cannot cross more than the distance it moves, so while the clearance exceeds
    the distance covered by ``n`` steps, those steps cannot collide.
    """
    if not math.isfinite(clearance) or step_length <= 0.0:
        return 1
    return max(1, int((clearance - margin) / step_length))


# -- planners ---------------------------------------------------------------------------


def plan_homing(position: Vec3, travel: Vec3, reference: Vec3 = (0.0, 0.0, 0.0)) -> list[Vec3]:
    """Referencing: raise Z to the top, move X and Y to the reference marks, then lower Z."""
    x, y, z = position
    top = travel[2]
    return [(x, y, z), (x, y, top), (reference[0], reference[1], top), reference]


@dataclass(frozen=True, slots=True)
class ToolChangeStage:
    """One part of a tool change: drive ``waypoints``, then stand for ``dwell`` seconds."""

    #: ``release`` (put the old module back), ``take`` (pick up the new one) or ``return``.
    action: str
    waypoints: tuple[Vec3, ...]
    dwell: float


def plan_tool_change(
    position: Vec3,
    old_slot: Vec3 | None,
    new_slot: Vec3 | None,
    travel: Vec3,
    *,
    safe_z: float = 250.0,
    dock_z: float = 40.0,
    dwell: float = 2.0,
) -> list[ToolChangeStage]:
    """Plan a stylus/probe module change at the rack.

    Rise to a safe height, drive over the old module's port, dock and release it, move
    to the new module's port, dock and take it, and return to ``position``. A tool
    without a port (no tool, a manual swap) skips its half.
    """
    safe = min(travel[2], max(position[2], safe_z))
    here = (position[0], position[1], safe)
    stages: list[ToolChangeStage] = []
    path: list[Vec3] = [position, here]
    for action, slot in (("release", old_slot), ("take", new_slot)):
        if slot is None:
            continue
        sx, sy = slot[0], slot[1]
        stages.append(ToolChangeStage(action, (*path, (sx, sy, safe), (sx, sy, dock_z)), dwell))
        path = [(sx, sy, dock_z), (sx, sy, safe)]
    stages.append(ToolChangeStage("return", (*path, here, position), 0.0))
    return stages


#: Directions of the five touches that qualify a tool on a sphere: top and four sides.
QUALIFICATION_DIRECTIONS: tuple[Vec3, ...] = (
    (0.0, 0.0, 1.0),
    (1.0, 0.0, 0.0),
    (-1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, -1.0, 0.0),
)


def plan_qualification(
    position: Vec3, center: Vec3, radius: float, ball_radius: float, *, standoff: float = 6.0
) -> list[Vec3]:
    """Plan the touches that qualify a tool on a reference sphere.

    Go above the sphere, then for each direction approach to ``standoff`` from the
    contact, touch (the ball centre is ``radius + ball_radius`` from the sphere centre),
    and retract; finally return above the sphere.
    """
    reach = radius + ball_radius
    above = (center[0], center[1], center[2] + reach + 25.0)
    path: list[Vec3] = [position, above]
    for direction in QUALIFICATION_DIRECTIONS:
        touch = add(center, scale(direction, reach))
        away = add(center, scale(direction, reach + standoff))
        path += [away, touch, away]
    path.append(above)
    return path
