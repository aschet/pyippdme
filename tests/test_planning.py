# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""The path planners and motion maths are plain functions: no CAD, no simulation needed."""

from __future__ import annotations

import subprocess
import sys

import pytest

from pyippdme.twin import planning, toolmath
from pyippdme.twin.spec import DEFAULT_TOOLS, PRESETS, ToolSpec


def test_pure_modules_import_without_opencascade() -> None:
    code = (
        "import sys; sys.modules['OCP'] = None; "
        "import pyippdme.twin.planning, pyippdme.twin.toolmath, pyippdme.twin.spec, "
        "pyippdme.twin.features, pyippdme.twin.optical, pyippdme.twin.depthbuffer; "
        "print('ok')"
    )
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.stdout.strip() == "ok", result.stderr


def test_trapezoid_profile_covers_the_distance_in_the_planned_time() -> None:
    duration = planning.travel_time(500.0, 100.0, 200.0)
    assert duration == pytest.approx(500.0 / 100.0 + 100.0 / 200.0)
    assert planning.travelled(duration, 500.0, 100.0, 200.0) == pytest.approx(500.0)
    samples = [planning.travelled(duration * f / 20, 500.0, 100.0, 200.0) for f in range(21)]
    assert samples == sorted(samples)  # never goes backwards
    # Short moves never reach the top speed: a triangular profile.
    assert planning.travel_time(1.0, 500.0, 100.0) == pytest.approx(2 * (1.0 / 100.0) ** 0.5)


def test_free_steps_skip_what_the_tool_cannot_cross() -> None:
    assert planning.free_steps(float("inf"), 1.5) == 1
    assert planning.free_steps(31.0, 1.5) == 20
    assert planning.free_steps(0.2, 1.5) == 1


def test_homing_raises_z_before_moving_xy() -> None:
    path = planning.plan_homing((100.0, 200.0, 50.0), (700.0, 700.0, 600.0))
    assert path == [(100.0, 200.0, 50.0), (100.0, 200.0, 600.0), (0.0, 0.0, 600.0), (0.0, 0.0, 0.0)]


def test_tool_change_visits_the_old_port_then_the_new_one_and_returns() -> None:
    stages = planning.plan_tool_change(
        (300.0, 300.0, 100.0), (50.0, 650.0, 20.0), (130.0, 650.0, 20.0), (700.0, 700.0, 600.0)
    )
    assert [s.action for s in stages] == ["release", "take", "return"]
    assert stages[0].waypoints[-1][:2] == (50.0, 650.0)
    assert stages[1].waypoints[-1][:2] == (130.0, 650.0)
    assert stages[-1].waypoints[-1] == (300.0, 300.0, 100.0)
    assert all(s.dwell > 0 for s in stages[:2])
    # No old module (no tool mounted): only take and return.
    only_take = planning.plan_tool_change(
        (0.0, 0.0, 0.0), None, (130.0, 650.0, 20.0), (700.0, 700.0, 600.0)
    )
    assert [s.action for s in only_take] == ["take", "return"]


def test_qualification_touches_the_sphere_from_five_directions() -> None:
    path = planning.plan_qualification((0.0, 0.0, 0.0), (100.0, 100.0, 100.0), 12.5, 1.5)
    touches = [
        p for p in path if abs(planning.path_length([p, (100.0, 100.0, 100.0)]) - 14.0) < 1e-9
    ]
    assert len(touches) == 5


def test_head_rotation_snaps_an_indexing_head_and_ignores_a_fixed_mount() -> None:
    fixed = DEFAULT_TOOLS["RefTool"]
    indexed = DEFAULT_TOOLS["IndexedTP200"]
    continuous = DEFAULT_TOOLS["RevoScan"]
    tilted = (0.2, 0.0, 0.98)
    assert toolmath.head_rotation(fixed, tilted)[1] == (0.0, 0.0, 1.0)
    _, axis, position = toolmath.head_rotation(indexed, tilted)
    assert position is not None
    assert position[0] % indexed.index_step == 0.0  # snapped to the 7.5 degree grid
    _, axis_c, position_c = toolmath.head_rotation(continuous, tilted)
    assert position_c is None
    assert axis_c[0] == pytest.approx(0.2 / (0.2**2 + 0.98**2) ** 0.5, rel=1e-6)
    assert axis != axis_c


def test_tip_offset_of_a_star_and_pivot_of_a_tool() -> None:
    star = DEFAULT_TOOLS["StarXP"]
    assert toolmath.tip_offset(star)[0] == star.arm_length
    pivot = toolmath.pivot_for(star, (100.0, 100.0, 100.0), (0.0, 0.0, 1.0))
    # The pivot sits above and behind the tip by the stylus geometry.
    assert pivot[0] == pytest.approx(100.0 - star.arm_length)
    assert pivot[2] > 100.0


def test_tool_spec_is_editable_data_with_validation() -> None:
    spec = ToolSpec("MyTool", ball_radius=2.0, shaft_length=25.0)
    assert toolmath.drop(spec) == pytest.approx(14.0 + 28.0 + 25.0)
    with pytest.raises(ValueError, match="mode"):
        ToolSpec("x", mode="telepathy")
    with pytest.raises(ValueError, match="star"):
        ToolSpec("x", tip="+x")
    assert {t.mode for t in DEFAULT_TOOLS.values()} >= {
        "touch",
        "head_touch",
        "scanning",
        "laser",
        "point_laser",
        "area",
        "camera",
    }
    assert PRESETS["bridge-700"].accuracy.sigma_mm(300.0) < 0.001
