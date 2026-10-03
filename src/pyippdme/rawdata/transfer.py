# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""The three same-machine/cross-machine raw-data transfer mechanics (6.17.2).

Each function here implements one of the transfer technologies
``AdvDataStruct.ReturnTechnology`` can advertise (:mod:`pyippdme.types.rawdata`):
a binary TCP socket (6.17.2.1), OS shared memory (6.17.2.2, via the
standard library's cross-platform ``multiprocessing.shared_memory``, so it
works unmodified on both Linux and Windows), and a plain file (6.17.2.3).
All three carry a :class:`~pyippdme.types.pointcloud.PointCloudSet` as an ESBF
stream (Annex C.2, :mod:`pyippdme.rawdata.formats`) of ``float`` or ``double``
values, little-endian ("Intel format", 6.17.2.1).
"""

from __future__ import annotations

import asyncio
import tempfile
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np
import numpy.typing as npt

from pyippdme.protocol.network import TCP_NETWORK, Network, StreamReaderLike, StreamWriterLike
from pyippdme.rawdata.formats import PointSetData, ScanData, pack_esbf, unpack_esbf
from pyippdme.types.pointcloud import PointCloudSet

#: How long a binary-socket listener waits for the client to connect before
#: giving up (this is a reference/simulated implementation, not a tuned
#: production default).
BIN_SOCKET_ACCEPT_TIMEOUT = 30.0


def point_groups(data: PointCloudSet) -> tuple[tuple[PointSetData, ...], ...]:
    """Turn the point clouds into SBF/ESBF point groups: one group per cloud, one set per set."""
    return tuple(
        tuple(
            PointSetData(
                point_set.normal,
                np.asarray([(p.x, p.y, p.z) for p in point_set.points], dtype=np.float64).reshape(
                    -1, 3
                ),
            )
            for point_set in cloud.point_sets
        )
        for cloud in data.point_clouds
    )


def esbf_payload(data: PointCloudSet, data_format: str, *, counted: bool = True) -> bytes:
    """Serialize ``data`` as an ESBF stream (Annex C.2), the format all transfers here use."""
    return pack_esbf(point_groups(data), data_format, description="pyippdme", counted=counted)


async def read_samples(
    host: str,
    port: int,
    data_format: str,
    *,
    network: Network = TCP_NETWORK,
    timeout: float = 5.0,
) -> npt.NDArray[np.float64]:
    """Connect to a server's raw-data port, read until it closes, and return all points.

    The stream is ESBF (Annex C.2); ``data_format`` is what ``RawDataBinSetup``
    asked for and is checked against the stream's own header. The server only
    listens while a ``GetRawDataBin`` is running, so connection attempts are
    retried until ``timeout`` seconds have passed.
    """
    scan = await read_scan_data(host, port, network=network, timeout=timeout)
    if scan.data_format != data_format:
        raise ValueError(f"Expected {data_format!r} samples, the stream holds {scan.data_format!r}")
    return scan.points()


async def read_scan_data(
    host: str, port: int, *, network: Network = TCP_NETWORK, timeout: float = 5.0
) -> ScanData:
    """Like :func:`read_samples`, but return the point groups and sets of the ESBF stream."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        try:
            reader, writer = await network.open_connection(host, port)
            break
        except OSError:
            if loop.time() >= deadline:
                raise
            await asyncio.sleep(0.02)
    try:
        payload = await reader.read()
    finally:
        writer.close()
        await writer.wait_closed()
    return unpack_esbf(payload)


async def port_is_available(host: str, port: int, *, network: Network = TCP_NETWORK) -> bool:
    try:
        listener = await network.start_server(_discard_client, host, port)
    except OSError:
        return False
    await listener.close()
    return True


async def send_once(
    host: str, port: int, payload: bytes, *, network: Network = TCP_NETWORK
) -> None:
    """Listen on ``(host, port)``, accept exactly one connection, write ``payload``, close.

    Bounded by :data:`BIN_SOCKET_ACCEPT_TIMEOUT` so a client that never
    connects cannot leak the listener forever. A fresh listener is opened
    per call (rather than kept alive from ``RawDataBinSetup``), since only
    one client connection is ever expected per acquisition retrieval.
    """
    loop = asyncio.get_running_loop()
    connected: asyncio.Future[StreamWriterLike] = loop.create_future()

    async def _on_client(_reader: StreamReaderLike, writer: StreamWriterLike) -> None:
        if not connected.done():
            connected.set_result(writer)

    server = await network.start_server(_on_client, host, port)
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
        await server.close()


async def _discard_client(_reader: StreamReaderLike, writer: StreamWriterLike) -> None:
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
