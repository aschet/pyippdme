# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for coordinate-system transformation math, persistence, and commands."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pyippdme import IppDmeClient
from pyippdme.exceptions import IppDmeServerError
from pyippdme.protocol.ast import BasicName, DataPayload, Items, Number, String
from pyippdme.protocol.commands import CommandName
from pyippdme.types.csy import CoordinateTransform, FileCsyStore, InMemoryCsyStore
from pyippdme.types.vec3 import Vec3


def _numbers(data: DataPayload) -> dict[str, float]:
    assert isinstance(data, Items)
    result = {}
    for item in data.values:
        assert len(item.args) == 1
        value = item.args[0]
        assert isinstance(value, Number)
        result[item.name] = value.value
    return result


def test_identity_transform_is_a_no_op() -> None:
    t = CoordinateTransform.identity()
    assert t.apply((1.0, 2.0, 3.0)) == (1.0, 2.0, 3.0)


def test_psi_and_phi_are_normalized_modulo_360() -> None:
    t = CoordinateTransform(0, 0, 0, 0, 370.0, -10.0)
    assert t.psi == pytest.approx(10.0)
    assert t.phi == pytest.approx(350.0)


def test_apply_translates_and_rotates() -> None:
    t = CoordinateTransform(10, 0, 0, 0, 0, 90)
    x, y, z = t.apply((1.0, 0.0, 0.0))
    assert x == pytest.approx(10.0)
    assert y == pytest.approx(1.0)
    assert z == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("theta", "psi", "phi"),
    [(0, 0, 0), (45, 90, 180), (180, 270, 45), (90, 0, 0)],
)
def test_apply_then_inverse_round_trips(theta: float, psi: float, phi: float) -> None:
    t = CoordinateTransform(3, -4, 5, theta, psi, phi)
    point = (7.0, -2.0, 11.0)
    assert t.inverse().apply(t.apply(point)) == pytest.approx(point)


def test_is_theta_in_range() -> None:
    assert CoordinateTransform(0, 0, 0, 0, 0, 0).is_theta_in_range
    assert CoordinateTransform(0, 0, 0, 180, 0, 0).is_theta_in_range
    assert not CoordinateTransform(0, 0, 0, 180.1, 0, 0).is_theta_in_range
    assert not CoordinateTransform(0, 0, 0, -0.1, 0, 0).is_theta_in_range


async def test_in_memory_csy_store_round_trips() -> None:
    store = InMemoryCsyStore()
    t = CoordinateTransform(1, 2, 3, 10, 20, 30)
    assert await store.load("Fixture") is None
    await store.save("Fixture", t)
    assert await store.load("Fixture") == t
    assert await store.names() == ("Fixture",)
    assert await store.delete("Fixture")
    assert await store.load("Fixture") is None
    assert not await store.delete("Fixture")


async def test_file_csy_store_round_trips(tmp_path: object) -> None:
    store = FileCsyStore(tmp_path)  # type: ignore[arg-type]
    t = CoordinateTransform(1, 2, 3, 10, 20, 30)
    await store.save("Fixture", t)
    loaded = await store.load("Fixture")
    assert loaded == t
    assert await store.names() == ("Fixture",)


async def test_file_csy_store_rejects_path_traversal_names(tmp_path: object) -> None:
    store = FileCsyStore(tmp_path)  # type: ignore[arg-type]
    t = CoordinateTransform.identity()
    with pytest.raises(ValueError, match="not a valid"):
        await store.save("../escape", t)
    with pytest.raises(ValueError, match="not a valid"):
        await store.save("a/b", t)


async def test_set_and_get_csy_transformation(started_client: IppDmeClient) -> None:
    await started_client.call(
        CommandName.SET_CSY_TRANSFORMATION,
        BasicName("PartCsy"),
        Number.of(1),
        Number.of(2),
        Number.of(3),
        Number.of(10),
        Number.of(20),
        Number.of(30),
    )
    (data,) = await started_client.call(CommandName.GET_CSY_TRANSFORMATION, BasicName("PartCsy"))
    values = _numbers(data)
    assert values == {"X0": 1, "Y0": 2, "Z0": 3, "Theta": 10, "Psi": 20, "Phi": 30}


async def test_get_csy_transformation_unset_raises_1013(started_client: IppDmeClient) -> None:
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(CommandName.GET_CSY_TRANSFORMATION, BasicName("SensorCsy"))
    assert excinfo.value.error.number == "1013"


async def test_set_csy_transformation_rejects_unknown_csy(started_client: IppDmeClient) -> None:
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(
            CommandName.SET_CSY_TRANSFORMATION,
            BasicName("MachineCsy"),  # not one of the LIVE_TRANSFORM_NAMES
            *(Number.of(0) for _ in range(6)),
        )
    assert excinfo.value.error.number == "0505"


async def test_set_csy_transformation_rejects_theta_out_of_range(
    started_client: IppDmeClient,
) -> None:
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(
            CommandName.SET_CSY_TRANSFORMATION,
            BasicName("PartCsy"),
            Number.of(0),
            Number.of(0),
            Number.of(0),
            Number.of(181),
            Number.of(0),
            Number.of(0),
        )
    assert excinfo.value.error.number == "1007"


async def test_named_csy_transformation_save_get_delete(started_client: IppDmeClient) -> None:
    await started_client.call(
        CommandName.SAVE_NAMED_CSY_TRANSFORMATION,
        String("Fixture1"),
        Number.of(1),
        Number.of(2),
        Number.of(3),
        Number.of(0),
        Number.of(0),
        Number.of(0),
    )
    (data,) = await started_client.call(
        CommandName.GET_NAMED_CSY_TRANSFORMATION, String("Fixture1")
    )
    assert _numbers(data) == {"X0": 1, "Y0": 2, "Z0": 3, "Theta": 0, "Psi": 0, "Phi": 0}

    names = await started_client.call(CommandName.ENUM_COORD_SYSTEMS)
    assert [d.to_wire() for d in names] == ['"Fixture1"']

    await started_client.call(CommandName.DELETE_COORD_SYSTEM, String("Fixture1"))
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(CommandName.DELETE_COORD_SYSTEM, String("Fixture1"))
    assert excinfo.value.error.number == "1013"


async def test_save_active_and_load_coord_system(started_client: IppDmeClient) -> None:
    await started_client.call(
        CommandName.SET_CSY_TRANSFORMATION,
        BasicName("PartCsy"),
        Number.of(5),
        Number.of(6),
        Number.of(7),
        Number.of(0),
        Number.of(0),
        Number.of(0),
    )
    await started_client.call(CommandName.SAVE_ACTIVE_COORD_SYSTEM, String("MyPart"))

    # Overwrite the live PartCsy transform, then reload the saved one.
    await started_client.call(
        CommandName.SET_CSY_TRANSFORMATION,
        BasicName("PartCsy"),
        Number.of(0),
        Number.of(0),
        Number.of(0),
        Number.of(0),
        Number.of(0),
        Number.of(0),
    )
    await started_client.call(CommandName.LOAD_COORD_SYSTEM, String("MyPart"))
    (data,) = await started_client.call(CommandName.GET_CSY_TRANSFORMATION, BasicName("PartCsy"))
    assert _numbers(data)["X0"] == 5


async def test_load_coord_system_unknown_raises_1013(started_client: IppDmeClient) -> None:
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(CommandName.LOAD_COORD_SYSTEM, String("DoesNotExist"))
    assert excinfo.value.error.number == "1013"


# -- the standard's formula (6.5.1, Figure 12 text) ---------------------------------------


def _standard_parent_to_child(p: Vec3, t: Vec3, phi: float, theta: float, psi: float) -> Vec3:
    """P' = A(phi) . B(theta) . C(psi) . (p - t), exactly as printed in the standard."""
    f, th, ps = (math.radians(a) for a in (phi, theta, psi))
    a = np.array([[math.cos(f), math.sin(f), 0], [-math.sin(f), math.cos(f), 0], [0, 0, 1]])
    b = np.array([[1, 0, 0], [0, math.cos(th), math.sin(th)], [0, -math.sin(th), math.cos(th)]])
    c = np.array([[math.cos(ps), math.sin(ps), 0], [-math.sin(ps), math.cos(ps), 0], [0, 0, 1]])
    v = a @ b @ c @ (np.asarray(p) - np.asarray(t))
    return (float(v[0]), float(v[1]), float(v[2]))


@pytest.mark.parametrize(
    ("theta", "psi", "phi"), [(0, 0, 90), (90, 0, 0), (50, 20, 70), (120, 300, 10), (180, 45, 270)]
)
def test_transformation_follows_the_formula_of_the_standard(
    theta: float, psi: float, phi: float
) -> None:
    t = CoordinateTransform(3.0, -4.0, 5.0, theta, psi, phi)
    point = (10.0, 20.0, 30.0)
    expected = _standard_parent_to_child(point, (3.0, -4.0, 5.0), phi, theta, psi)
    assert t.parent_to_child(point) == pytest.approx(expected)
    assert t.apply(expected) == pytest.approx(point)  # and back, child to parent


def test_phi_and_psi_do_not_commute() -> None:
    a = CoordinateTransform(0, 0, 0, 40, 10, 80)
    b = CoordinateTransform(0, 0, 0, 40, 80, 10)  # the two angles exchanged
    assert a.apply((1.0, 2.0, 3.0)) != pytest.approx(b.apply((1.0, 2.0, 3.0)))


def test_the_chain_runs_from_the_part_through_the_rotary_table_to_the_machine() -> None:
    from pyippdme.types.csy import CSY_CHAIN, CsyContext

    assert CSY_CHAIN == (
        "MachineCsy",
        "MultipleArmCsy",
        "MoveableMachineCsy",
        "RotaryTableFixCsy",
        "RotaryTableVarCsy",
        "PartCsy",
    )
    transforms = {
        "MultipleArmCsy": CoordinateTransform(1, 0, 0, 0, 0, 0),
        "RotaryTableFixCsy": CoordinateTransform(0, 10, 0, 0, 0, 0),
        "PartCsy": CoordinateTransform(0, 0, 100, 0, 0, 0),
    }
    # A point of the part passes every system above it: +100 in z, +10 in y, +1 in x.
    assert CsyContext("PartCsy", transforms).to_machine((0, 0, 0)) == pytest.approx((1, 10, 100))
    # Entering the chain lower leaves out what is above: the rotary table system skips the part's.
    assert CsyContext("RotaryTableFixCsy", transforms).to_machine((0, 0, 0)) == pytest.approx(
        (1, 10, 0)
    )
    assert CsyContext("MachineCsy", transforms).to_machine((5, 6, 7)) == pytest.approx((5, 6, 7))


def test_context_converts_points_and_directions_both_ways() -> None:
    from pyippdme.types.csy import CsyContext

    context = CsyContext("PartCsy", {"PartCsy": CoordinateTransform(100, 0, 0, 0, 0, 90)})
    machine = context.to_machine((10.0, 0.0, 0.0))
    assert machine == pytest.approx((100.0, 10.0, 0.0))  # turned 90 degrees about z, then moved
    assert context.to_client(machine) == pytest.approx((10.0, 0.0, 0.0))
    # A direction is only turned, never moved.
    assert context.direction_to_machine((1.0, 0.0, 0.0)) == pytest.approx((0.0, 1.0, 0.0))
    assert context.direction_to_client((0.0, 1.0, 0.0)) == pytest.approx((1.0, 0.0, 0.0))


def test_rotary_table_system_follows_the_table_angle_when_it_is_given() -> None:
    from pyippdme.types.csy import CsyContext

    quarter_turn = np.eye(4)
    quarter_turn[:2, :2] = [[0.0, -1.0], [1.0, 0.0]]
    on = CsyContext("PartCsy", {}, quarter_turn)
    assert on.to_machine((1.0, 0.0, 0.0)) == pytest.approx((0.0, 1.0, 0.0))
    assert CsyContext("PartCsy", {}).to_machine((1.0, 0.0, 0.0)) == pytest.approx((1.0, 0.0, 0.0))
