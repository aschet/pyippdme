# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from pyippdme import IppDmeClient, VirtualCMM
from pyippdme.types.csy import InMemoryCsyStore


@pytest.fixture
async def server() -> AsyncIterator[VirtualCMM]:
    # In-memory, not VirtualCMM's own persistent-by-default FileCsyStore
    # (~/.pyippdme/csy): tests must not leak named coordinate systems into
    # the real user's home directory or across otherwise-unrelated test runs.
    srv = VirtualCMM(csy_store=InMemoryCsyStore())
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
async def client(server_port: int) -> AsyncIterator[IppDmeClient]:
    c = await IppDmeClient.connect("127.0.0.1", server_port)
    try:
        yield c
    finally:
        await c.close()


@pytest.fixture
async def started_client(client: IppDmeClient) -> IppDmeClient:
    await client.start_session()
    return client
