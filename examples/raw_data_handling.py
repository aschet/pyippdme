#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Walk through all three RawDataHandling retrieval technologies (VDMA 8722 6.17.2).

Starts an in-process VirtualCMM, triggers one raw-data acquisition
(``DataAcquire``, 6.15.1), and retrieves it three different ways - the same
acquisition, in each case:

1. As a file (6.17.2.3): ``GetRawDataFile`` writes the samples to a file as
   an ESBF stream (Annex C.2) and returns its ``file://`` URL; a client
   reads it back with ``pyippdme.rawdata.formats.unpack_esbf``.
2. Via OS shared memory (6.17.2.2): ``GetRawDataShaMem`` publishes the same
   ESBF stream in a named ``multiprocessing.shared_memory.SharedMemory``
   segment - genuinely readable by another process on the same machine,
   not just simulated.
3. Via a binary TCP socket (6.17.2.1): ``RawDataBinSetup`` negotiates a
   port and format, then ``GetRawDataBin`` streams the samples to whoever
   connects to that port - here, a second connection made by this same
   script, standing in for a real second process.

Run directly: ``python examples/raw_data_handling.py``. No server needs to
be running first - this script starts and stops its own.
"""

from __future__ import annotations

import asyncio
import socket
from multiprocessing import shared_memory
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

from pyippdme import IppDmeMachine, VirtualCMM
from pyippdme.rawdata.formats import unpack_esbf


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = int(s.getsockname()[1])
    s.close()
    return port


async def _receive_binary(host: str, port: int) -> bytes:
    await asyncio.sleep(0.05)  # give GetRawDataBin's listener a moment to start
    reader, writer = await asyncio.open_connection(host, port)
    payload = await reader.read()
    writer.close()
    return payload


async def main() -> None:
    server = VirtualCMM()
    host = "127.0.0.1"
    port = await server.start(host, 0)

    try:
        machine = await IppDmeMachine.connect(host, port)
        await machine.start_session()

        struct_data = await machine.raw_data.get_adv_data_struct()
        print(f"AdvDataStruct advertises: {sorted(struct_data.return_technologies)}")

        await machine.raw_data.data_acquire("Acq1", "SingleShot", "DefaultSettings")
        print("DataAcquire('Acq1') done\n")

        # 1. File
        file_url = await machine.raw_data.get_raw_data_file("Acq1")
        scan = unpack_esbf(Path(url2pathname(urlparse(file_url).path)).read_bytes())
        print(f"[File]         GetRawDataFile -> {file_url}")
        print(f"               read via pyippdme.rawdata.formats.unpack_esbf: {scan.points()}")
        await machine.raw_data.del_raw_data_file("Acq1")

        # 2. Shared memory
        shmem_name, _offset, size = await machine.raw_data.get_raw_data_sha_mem("Acq1")
        print(f"[SharedMemory] GetRawDataShaMem -> name={shmem_name!r} size={size}")
        segment = shared_memory.SharedMemory(name=shmem_name)
        assert segment.buf is not None  # noqa: S101 (just-opened segment always has a live buffer)
        samples = unpack_esbf(bytes(segment.buf[:size])).points()
        print(f"               read back directly via multiprocessing.shared_memory: {samples}")
        segment.close()
        await machine.raw_data.release_sha_mem("Acq1")

        # 3. Binary socket
        bin_port = _free_port()
        await machine.raw_data.raw_data_bin_setup("double", bin_port)
        _done, payload = await asyncio.gather(
            machine.raw_data.get_raw_data_bin("Acq1"), _receive_binary(host, bin_port)
        )
        print(
            f"[BinarySocket] GetRawDataBin over port {bin_port} -> {unpack_esbf(payload).points()}"
        )

        await machine.end_session()
        await machine.close()
    finally:
        await server.close()

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
