# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""A moving table, and where the tool rack stands."""

from __future__ import annotations

import pytest

pytest.importorskip("OCP")

from pyippdme.twin import DigitalTwin, MachineModel
from pyippdme.twin.spec import PRESETS, RACK_SIDES, MachineSpec, manifest_to_toml, parse_manifest
from pyippdme.twin.toolmath import rack_slots


def _y_of(twin: DigitalTwin, key: str, y: float) -> float:
    twin._pos = (100.0, y, 50.0)
    pose = {i.name: i.pose for i in twin.draw_items()}[key]
    return float(pose[1, 3])


def test_a_fixed_table_stays_and_the_bridge_moves() -> None:
    twin = DigitalTwin(MachineModel.default("bridge-700"), time_scale=0.0)
    assert _y_of(twin, "table", 0.0) == _y_of(twin, "table", 400.0) == 0.0
    assert _y_of(twin, "bridge", 400.0) == pytest.approx(400.0)


def test_a_moving_table_slides_and_the_bridge_stands() -> None:
    twin = DigitalTwin(MachineModel.default("moving-table-600"), time_scale=0.0)
    assert _y_of(twin, "bridge", 0.0) == _y_of(twin, "bridge", 400.0) == 0.0
    assert _y_of(twin, "table", 400.0) == pytest.approx(-400.0)
    assert _y_of(twin, "rack", 400.0) == pytest.approx(-400.0)  # what stands on it goes along
    # The tool point stays at one world Y, whatever the machine Y is.
    snap = twin.snapshot()
    assert snap.position[1] + snap.world_shift[1] == pytest.approx(0.0)


def test_a_moving_table_gives_the_same_collisions_as_a_fixed_one() -> None:
    fixed = DigitalTwin(MachineModel.default("bridge-700"), time_scale=0.0)
    moving = DigitalTwin(
        MachineModel.default(PRESETS["bridge-700"].with_(table_kind="moving-y")), time_scale=0.0
    )
    for position in ((100.0, 100.0, -5.0), (100.0, 100.0, 300.0), (300.0, 650.0, -8.0)):
        assert fixed._collisions(position, 0.0, "RefTool", None) == moving._collisions(
            position, 0.0, "RefTool", None
        )
    assert moving.machine.carried_axes(
        next(b for b in moving.machine.bodies if b.spec.name == "bridge")
    ) == ("y",)


def test_the_rack_stands_where_the_spec_says() -> None:
    spec = PRESETS["bridge-700"]
    keys = ["a", "b", "c"]
    back = rack_slots(spec, keys)
    assert {p[1] for p in back.values()} == {spec.travel[1] - 45.0}
    front = rack_slots(spec.with_(rack_side="front", rack_inset=10.0), keys)
    assert {p[1] for p in front.values()} == {10.0}
    right = rack_slots(spec.with_(rack_side="right"), keys)
    assert {p[0] for p in right.values()} == {spec.travel[0] - 45.0}
    assert len({p[1] for p in right.values()}) == 3  # the row runs along Y
    free = rack_slots(spec.with_(rack_origin=(10.0, 20.0, -5.0)), keys)
    assert free["a"] == (10.0, 20.0, -5.0)
    for side in RACK_SIDES:
        model = MachineModel.default(spec.with_(rack_side=side))
        assert any(b.spec.name == "rack" for b in model.bodies)


def test_table_kind_and_rack_survive_the_machine_file() -> None:
    spec = PRESETS["moving-table-600"].with_(rack_origin=(1.0, 2.0, 3.0))
    manifest = parse_manifest(_toml(manifest_to_toml(_manifest(spec))))
    assert manifest.spec.table_kind == "moving-y"
    assert manifest.spec.rack_side == "left"
    assert manifest.spec.rack_inset == -40.0
    assert manifest.spec.rack_origin == (1.0, 2.0, 3.0)
    with pytest.raises(ValueError, match="table_kind"):
        MachineSpec(table_kind="rotating")
    with pytest.raises(ValueError, match="rack_side"):
        MachineSpec(rack_side="top")


def _toml(text: str) -> dict[str, object]:
    import tomllib

    return tomllib.loads(text)


def _manifest(spec: MachineSpec):  # type: ignore[no-untyped-def]
    from pyippdme.twin.spec import MachineManifest

    return MachineManifest(spec)
