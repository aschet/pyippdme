# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from pyippdme import IppDmeClient, VirtualCMM
from pyippdme.protocol.network import MemoryNetwork
from pyippdme.types.csy import InMemoryCsyStore


@pytest.fixture
def network() -> MemoryNetwork:
    return MemoryNetwork()


@pytest.fixture
async def server(network: MemoryNetwork) -> AsyncIterator[VirtualCMM]:
    # In-memory, not VirtualCMM's own persistent-by-default FileCsyStore
    # (~/.pyippdme/csy): tests must not leak named coordinate systems into
    # the real user's home directory or across otherwise-unrelated test runs.
    srv = VirtualCMM(csy_store=InMemoryCsyStore(), network=network)
    await srv.start("127.0.0.1", 0)
    try:
        yield srv
    finally:
        await srv.close()


@pytest.fixture
def server_port(server: VirtualCMM) -> int:
    assert server.port is not None
    return server.port


@pytest.fixture
async def client(network: MemoryNetwork, server_port: int) -> AsyncIterator[IppDmeClient]:
    c = await IppDmeClient.connect("127.0.0.1", server_port, network=network)
    try:
        yield c
    finally:
        await c.close()


@pytest.fixture
async def started_client(client: IppDmeClient) -> IppDmeClient:
    await client.start_session()
    return client


@pytest.fixture
async def tcp_server() -> AsyncIterator[VirtualCMM]:
    srv = VirtualCMM(csy_store=InMemoryCsyStore())
    await srv.start("127.0.0.1", 0)
    try:
        yield srv
    finally:
        await srv.close()


@pytest.fixture
def tcp_server_port(tcp_server: VirtualCMM) -> int:
    assert tcp_server.port is not None
    return tcp_server.port
