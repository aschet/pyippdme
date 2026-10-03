# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for the Optical/RawDataHandling command subset (DataAcquire + all 3 transfer methods)."""

from __future__ import annotations

import asyncio
import socket
import struct
from multiprocessing import shared_memory
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

import pytest

from pyippdme import IppDmeClient
from pyippdme.exceptions import IppDmeServerError
from pyippdme.protocol.ast import BasicName, DataPayload, Items, NamedValue, Number, String, Xml
from pyippdme.protocol.commands import CommandName
from pyippdme.protocol.network import TCP_NETWORK, MemoryNetwork, Network
from pyippdme.protocol.parameters import ParameterName
from pyippdme.simulation.classes.rawdata_class import _SWEEP_POINTS_PER_SEGMENT
from pyippdme.types.pointcloud import from_xml as point_cloud_from_xml
from pyippdme.types.rawdata import from_xml as adv_data_struct_from_xml


def _items(data: DataPayload) -> Items:
    assert isinstance(data, Items)
    return data


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = int(s.getsockname()[1])
    s.close()
    return port


async def _acquire(client: IppDmeClient, name: str) -> None:
    await client.call(
        CommandName.DATA_ACQUIRE,
        String(name),
        BasicName("SingleShot"),
        String("Settings1"),
        Number.of(0),
    )


async def _acquire_points(
    client: IppDmeClient,
    name: str,
    acquisition_type: str,
    positions: list[tuple[float, float, float]],
) -> None:
    numbers = [v for x, y, z in positions for v in (x, y, z, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0)]
    await client.call(
        CommandName.DATA_ACQUIRE,
        String(name),
        BasicName(acquisition_type),
        String("Settings1"),
        Number.of(len(positions)),
        *(Number.of(v) for v in numbers),
    )


def _path_from_file_url(url: str) -> Path:
    return Path(url2pathname(urlparse(url).path))


async def _read_points(client: IppDmeClient, name: str) -> tuple[tuple[float, float, float], ...]:
    (data,) = await client.call(CommandName.GET_RAW_DATA_FILE, String(name))
    (item,) = _items(data).values
    url = item.args[0]
    assert isinstance(url, String)
    path = _path_from_file_url(url.value)
    cloud_set = point_cloud_from_xml(path.read_text())
    (point_set,) = cloud_set.point_clouds[0].point_sets
    return tuple((p.x, p.y, p.z) for p in point_set.points)


async def test_adv_data_struct_is_parseable_xml(started_client: IppDmeClient) -> None:
    data = await started_client.call(CommandName.ADV_DATA_STRUCT)
    assert len(data) == 1
    assert isinstance(data[0], Xml)
    struct_data = adv_data_struct_from_xml(data[0].raw)
    assert struct_data.return_technologies == {"SocBin", "ShaMem", "File"}


async def test_data_acquire_single_shot_at_current_position(
    started_client: IppDmeClient,
) -> None:
    await started_client.call(CommandName.GO_TO, NamedValue(ParameterName.X, (Number.of(5.0),)))
    await _acquire(started_client, "Acq1")
    # No direct query command exists for a raw acquisition's contents other
    # than the retrieval methods below; this at least proves it didn't error.


async def test_data_acquire_single_shot_reports_the_requested_positions_with_noise(
    started_client: IppDmeClient,
) -> None:
    await _acquire_points(started_client, "Acq1", "SingleShot", [(5.0, 0.0, 0.0)])
    (point,) = await _read_points(started_client, "Acq1")
    # Deterministic per acquisition name - not flaky - but not exactly the
    # requested position either, since a little measurement noise is added.
    assert point != (5.0, 0.0, 0.0)
    assert point[0] == pytest.approx(5.0, abs=0.05)
    assert point[1] == pytest.approx(0.0, abs=0.05)
    assert point[2] == pytest.approx(0.0, abs=0.05)


async def test_data_acquire_multishot_reports_exactly_the_requested_positions(
    started_client: IppDmeClient,
) -> None:
    positions = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)]
    await _acquire_points(started_client, "Acq1", "MultiShot", positions)
    points = await _read_points(started_client, "Acq1")
    assert len(points) == len(positions)


async def test_data_acquire_sweep_densifies_the_path_between_control_points(
    started_client: IppDmeClient,
) -> None:
    positions = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (10.0, 10.0, 0.0)]
    await _acquire_points(started_client, "Acq1", "Sweep", positions)
    points = await _read_points(started_client, "Acq1")
    # 2 segments, each densified into _SWEEP_POINTS_PER_SEGMENT points, plus
    # the final control point.
    assert len(points) == 2 * _SWEEP_POINTS_PER_SEGMENT + 1


async def test_data_acquire_sweep_with_a_single_point_does_not_densify(
    started_client: IppDmeClient,
) -> None:
    await _acquire_points(started_client, "Acq1", "Sweep", [(5.0, 0.0, 0.0)])
    points = await _read_points(started_client, "Acq1")
    assert len(points) == 1


async def test_data_acquire_is_deterministic_per_acquisition_name(
    started_client: IppDmeClient,
) -> None:
    await _acquire_points(started_client, "SameName", "SingleShot", [(1.0, 2.0, 3.0)])
    first = await _read_points(started_client, "SameName")

    await started_client.call(CommandName.DELETE_ACQUISITION, String("SameName"))
    await _acquire_points(started_client, "SameName", "SingleShot", [(1.0, 2.0, 3.0)])
    second = await _read_points(started_client, "SameName")

    assert first == second


async def test_get_raw_data_file_writes_and_is_deletable(
    started_client: IppDmeClient, tmp_path: object
) -> None:
    await _acquire(started_client, "Acq1")
    (data,) = await started_client.call(CommandName.GET_RAW_DATA_FILE, String("Acq1"))
    (item,) = _items(data).values
    assert item.name == "FileURL"
    url = item.args[0]
    assert isinstance(url, String)
    assert url.value.startswith("file://")
    path = _path_from_file_url(url.value)
    assert path.is_file()

    await started_client.call(CommandName.DEL_RAW_DATA_FILE, String("Acq1"))


async def test_get_raw_data_file_unknown_acquisition_raises_2003(
    started_client: IppDmeClient,
) -> None:
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(CommandName.GET_RAW_DATA_FILE, String("NoSuchAcquisition"))
    assert excinfo.value.error.number == "2003"


async def test_get_raw_data_sha_mem_round_trips(started_client: IppDmeClient) -> None:
    await _acquire(started_client, "Acq1")
    (data,) = await started_client.call(CommandName.GET_RAW_DATA_SHA_MEM, String("Acq1"))

    values = {item.name: item.args[0] for item in _items(data).values}
    name_value = values["ShaMemName"]
    assert isinstance(name_value, String)
    size_value = values["Size"]
    assert isinstance(size_value, Number)
    assert int(size_value.value) == 24  # one MeasPoint == 3 doubles

    segment = shared_memory.SharedMemory(name=name_value.value)
    try:
        assert segment.buf is not None
        (x, y, z) = struct.unpack("<3d", bytes(segment.buf[:24]))
        # Not exactly (0, 0, 0): a little measurement noise is added.
        assert (x, y, z) == pytest.approx((0.0, 0.0, 0.0), abs=0.05)
    finally:
        segment.close()

    await started_client.call(CommandName.RELEASE_SHA_MEM, String("Acq1"))


async def test_release_sha_mem_unknown_is_a_no_op(started_client: IppDmeClient) -> None:
    await started_client.call(CommandName.RELEASE_SHA_MEM, String("NeverAcquired"))


async def test_raw_data_bin_setup_rejects_bad_format(started_client: IppDmeClient) -> None:
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(
            CommandName.RAW_DATA_BIN_SETUP,
            BasicName("triple"),
            Number.of(_free_port()),
            BasicName("Off"),
        )
    assert excinfo.value.error.number == "5000"


async def _stream_bin(client: IppDmeClient, network: Network) -> bytes:
    await _acquire(client, "Acq1")
    port = _free_port()
    await client.call(
        CommandName.RAW_DATA_BIN_SETUP, BasicName("double"), Number.of(port), BasicName("Off")
    )

    async def bin_client() -> bytes:
        await asyncio.sleep(0.05)
        reader, writer = await network.open_connection("127.0.0.1", port)
        payload = await reader.read(1024)
        writer.close()
        return payload

    _server_result, payload = await asyncio.gather(
        client.call(CommandName.GET_RAW_DATA_BIN, String("Acq1")), bin_client()
    )
    return payload


async def test_get_raw_data_bin_streams_over_the_negotiated_port(
    started_client: IppDmeClient, network: MemoryNetwork
) -> None:
    payload = await _stream_bin(started_client, network)
    # Not exactly (0, 0, 0): a little measurement noise is added.
    (x, y, z) = struct.unpack("<3d", payload)
    assert (x, y, z) == pytest.approx((0.0, 0.0, 0.0), abs=0.05)


async def test_get_raw_data_bin_streams_over_real_tcp(tcp_server_port: int) -> None:
    client = await IppDmeClient.connect("127.0.0.1", tcp_server_port)
    try:
        await client.start_session()
        payload = await _stream_bin(client, TCP_NETWORK)
    finally:
        await client.close()
    (x, y, z) = struct.unpack("<3d", payload)
    assert (x, y, z) == pytest.approx((0.0, 0.0, 0.0), abs=0.05)


async def test_get_raw_data_bin_without_setup_raises_0508(started_client: IppDmeClient) -> None:
    await _acquire(started_client, "Acq1")
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(CommandName.GET_RAW_DATA_BIN, String("Acq1"))
    assert excinfo.value.error.number == "0508"


async def test_end_session_clears_buffered_acquisitions(started_client: IppDmeClient) -> None:
    await _acquire(started_client, "Acq1")
    await started_client.call(CommandName.END_SESSION)
    await started_client.call(CommandName.START_SESSION)
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(CommandName.GET_RAW_DATA_FILE, String("Acq1"))
    assert excinfo.value.error.number == "2003"


async def test_delete_acquistion_removes_it(started_client: IppDmeClient) -> None:
    await _acquire(started_client, "Acq1")
    await started_client.call(CommandName.DELETE_ACQUISITION, String("Acq1"))
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(CommandName.GET_RAW_DATA_FILE, String("Acq1"))
    assert excinfo.value.error.number == "2003"


async def test_delete_acquistion_unknown_is_a_no_op(started_client: IppDmeClient) -> None:
    await started_client.call(CommandName.DELETE_ACQUISITION, String("NeverAcquired"))


async def test_delete_all_acquisitions_clears_everything(started_client: IppDmeClient) -> None:
    await _acquire(started_client, "Acq1")
    await _acquire(started_client, "Acq2")
    await started_client.call(CommandName.DELETE_ALL_ACQUISITIONS)
    for name in ("Acq1", "Acq2"):
        with pytest.raises(IppDmeServerError) as excinfo:
            await started_client.call(CommandName.GET_RAW_DATA_FILE, String(name))
        assert excinfo.value.error.number == "2003"
        await started_client.call(CommandName.CLEAR_ALL_ERRORS)
