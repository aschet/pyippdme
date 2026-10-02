# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""The three same-machine/cross-machine raw-data transfer mechanics (6.17.2).

Each function here implements one of the transfer technologies
``AdvDataStruct.ReturnTechnology`` can advertise (:mod:`pyippdme.types.rawdata`):
a binary TCP socket (6.17.2.1), OS shared memory (6.17.2.2, via the
standard library's cross-platform ``multiprocessing.shared_memory``, so it
works unmodified on both Linux and Windows), and a plain file (6.17.2.3).
All three serialize a :class:`~pyippdme.types.pointcloud.PointCloudSet` as flat
``(x, y, z)`` float32/float64 samples, little-endian ("Intel format"),
matching 6.17.2.1's ``RawDataBinSetup`` wire format description.
"""

from __future__ import annotations

import asyncio
import tempfile
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np
import numpy.typing as npt

from pyippdme.types.pointcloud import PointCloudSet

#: How long a binary-socket listener waits for the client to connect before
#: giving up (this is a reference/simulated implementation, not a tuned
#: production default).
BIN_SOCKET_ACCEPT_TIMEOUT = 30.0

#: The little-endian dtype :func:`pack_samples`/:func:`unpack_samples` use for
#: each wire ``data_format`` (6.17.2.1: "float" is 32-bit, "double" 64-bit).
_SAMPLE_DTYPES: dict[str, npt.DTypeLike] = {"float": "<f4", "double": "<f8"}


def flatten_points(data: PointCloudSet) -> npt.NDArray[np.float64]:
    """Flatten every ``(x, y, z)`` sample into one array, in document order."""
    values = [
        (point.x, point.y, point.z)
        for cloud in data.point_clouds
        for point_set in cloud.point_sets
        for point in point_set.points
    ]
    if not values:
        return np.empty(0, dtype=np.float64)
    return np.asarray(values, dtype=np.float64).reshape(-1)


def pack_samples(values: npt.NDArray[np.float64], data_format: str) -> bytes:
    """Pack ``values`` as little-endian IEEE 754 ``float``/``double`` (6.17.2.1)."""
    if data_format not in _SAMPLE_DTYPES:
        raise ValueError(f"data_format must be 'float' or 'double', got {data_format!r}")
    return np.asarray(values, dtype=_SAMPLE_DTYPES[data_format]).tobytes()


async def port_is_available(host: str, port: int) -> bool:
    try:
        server = await asyncio.start_server(_discard_client, host, port)
    except OSError:
        return False
    server.close()
    await server.wait_closed()
    return True


async def send_once(host: str, port: int, payload: bytes) -> None:
    """Listen on ``(host, port)``, accept exactly one connection, write ``payload``, close.

    Bounded by :data:`BIN_SOCKET_ACCEPT_TIMEOUT` so a client that never
    connects cannot leak the listener forever. A fresh listener is opened
    per call (rather than kept alive from ``RawDataBinSetup``), since only
    one client connection is ever expected per acquisition retrieval.
    """
    loop = asyncio.get_running_loop()
    connected: asyncio.Future[asyncio.StreamWriter] = loop.create_future()

    async def _on_client(_reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        if not connected.done():
            connected.set_result(writer)

    server = await asyncio.start_server(_on_client, host, port)
    try:
        try:
            writer = await asyncio.wait_for(connected, timeout=BIN_SOCKET_ACCEPT_TIMEOUT)
        except TimeoutError:
            return
        # The writer must be closed *before* the server: Server.wait_closed()
        # can otherwise block on the still-open accepted connection.
        try:
            writer.write(payload)
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
    finally:
        server.close()
        await server.wait_closed()


async def _discard_client(_reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    writer.close()


def publish_shared_memory(payload: bytes) -> shared_memory.SharedMemory:
    """Create an OS shared-memory segment containing ``payload`` at offset 0."""
    segment = shared_memory.SharedMemory(create=True, size=max(len(payload), 1))
    assert segment.buf is not None  # noqa: S101 (just-created segment always has a live buffer)
    segment.buf[: len(payload)] = payload
    return segment


def write_raw_data_file(directory: Path, acq_name: str, payload: bytes, suffix: str) -> Path:
    """Write ``payload`` to a file under ``directory``, named after ``acq_name``."""
    directory.mkdir(parents=True, exist_ok=True)
    # AcqName is client-controlled ([string]); do not trust it as a path component.
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in acq_name)
    path = directory / f"{safe_name}{suffix}"
    path.write_bytes(payload)
    return path


def default_raw_data_directory() -> Path:
    return Path(tempfile.gettempdir()) / "pyippdme" / "rawdata"
