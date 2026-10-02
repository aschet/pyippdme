# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for VirtualCMM and IppDmeServer's machine_class override."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from pyippdme import IppDmeClient, IppDmeServer
from pyippdme.protocol.commands import CommandName
from pyippdme.server.registry import MachineState
from pyippdme.simulation import DEFAULT_COMMAND_CLASSES
from pyippdme.simulation.classes import (
    register_cartcmm_class,
    register_dme_class,
    register_scanning_class,
    register_server_class,
    register_tool_class,
)
from pyippdme.simulation.virtual_cmm import (
    VIRTUAL_CMM_MACHINE_CLASS,
    VirtualCMM,
    default_machine_class,
)
from pyippdme.types.csy import CoordinateTransform, FileCsyStore, InMemoryCsyStore


@pytest.fixture
async def virtual_server(tmp_path: object) -> AsyncIterator[VirtualCMM]:
    srv = VirtualCMM(csy_store=FileCsyStore(tmp_path))  # type: ignore[arg-type]
    await srv.start("127.0.0.1", 0)
    try:
        yield srv
    finally:
        await srv.close()


async def test_virtual_cmm_reports_the_virtualcmm_suffix(virtual_server: VirtualCMM) -> None:
    assert virtual_server.port is not None
    client = await IppDmeClient.connect("127.0.0.1", virtual_server.port)
    try:
        await client.start_session()
        (data,) = await client.call(CommandName.GET_MACHINE_CLASS)
        assert data.to_wire() == f'"{VIRTUAL_CMM_MACHINE_CLASS}"'
    finally:
        await client.close()


async def test_virtual_cmm_persists_named_csy_across_instances(tmp_path: object) -> None:
    store_a = FileCsyStore(tmp_path)  # type: ignore[arg-type]
    await store_a.save("Fixture", CoordinateTransform(1, 2, 3, 0, 0, 0))

    store_b = FileCsyStore(tmp_path)  # type: ignore[arg-type]
    loaded = await store_b.load("Fixture")
    assert loaded == CoordinateTransform(1, 2, 3, 0, 0, 0)


def test_virtual_cmm_defaults_to_file_csy_store_not_in_memory() -> None:
    server = VirtualCMM()
    assert isinstance(server.csy_store, FileCsyStore)


def test_plain_ipp_dme_server_still_defaults_to_in_memory_csy_store() -> None:
    server: IppDmeServer[MachineState] = IppDmeServer()
    assert isinstance(server.csy_store, InMemoryCsyStore)


def test_ipp_dme_server_accepts_custom_machine_class() -> None:
    server: IppDmeServer[MachineState] = IppDmeServer(machine_class="Custom_Class_String")
    assert server.machine_class == "Custom_Class_String"


def test_default_machine_class_reports_none_for_every_omitted_optional_component() -> None:
    components = (register_server_class, register_dme_class, register_cartcmm_class)
    assert default_machine_class(components) == (
        "CartCMM_None_Fixed_None_None_None_None_VirtualCMM"
    )


def test_default_machine_class_reports_touchtrigger_when_only_tool_is_present() -> None:
    components = (
        register_server_class,
        register_dme_class,
        register_cartcmm_class,
        register_tool_class,
    )
    assert default_machine_class(components) == (
        "CartCMM_TouchTrigger_Fixed_None_None_None_None_VirtualCMM"
    )


def test_default_machine_class_prefers_scanning_over_touchtrigger_when_both_are_present() -> None:
    components = (
        register_server_class,
        register_dme_class,
        register_cartcmm_class,
        register_tool_class,
        register_scanning_class,
    )
    assert default_machine_class(components) == (
        "CartCMM_Scanning_Fixed_None_None_None_None_VirtualCMM"
    )


def test_default_machine_class_reports_none_for_server_segment_without_cartcmm() -> None:
    components = (register_server_class, register_dme_class)
    assert default_machine_class(components) == "None_None_Fixed_None_None_None_None_VirtualCMM"


def test_default_machine_class_with_every_command_class_matches_the_full_default() -> None:
    assert default_machine_class(DEFAULT_COMMAND_CLASSES) == VIRTUAL_CMM_MACHINE_CLASS


def test_virtual_cmm_derives_machine_class_from_a_trimmed_command_classes() -> None:
    server = VirtualCMM(
        command_classes=(register_server_class, register_dme_class, register_cartcmm_class)
    )
    assert server.machine_class == "CartCMM_None_Fixed_None_None_None_None_VirtualCMM"


def test_virtual_cmm_still_accepts_an_explicit_machine_class_override() -> None:
    server = VirtualCMM(machine_class="Custom_Class_String")
    assert server.machine_class == "Custom_Class_String"
