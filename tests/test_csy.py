# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for coordinate-system transformation math, persistence, and commands."""

from __future__ import annotations

import pytest

from pyippdme import IppDmeClient
from pyippdme.exceptions import IppDmeServerError
from pyippdme.protocol.ast import BasicName, DataPayload, Items, Number, String
from pyippdme.protocol.commands import CommandName
from pyippdme.types.csy import CoordinateTransform, FileCsyStore, InMemoryCsyStore


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
