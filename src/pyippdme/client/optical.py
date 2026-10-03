# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Reading an optical or scanning sensor through the protocol: what it is, and what it sees.

:func:`read_sensor_info` asks the server about the active tool and how it hands over raw data
(``Tool.Id``, ``Tool.AdvDataStruct``). :func:`acquire` runs ``DataAcquire`` and fetches the points
over the binary transfer (6.15.1, 6.17.2.1) into a :class:`~pyippdme.rawdata.formats.ScanData`.
Neither needs a GUI.
"""

from __future__ import annotations

import socket
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pyippdme.client.builders import AcquisitionPoint
from pyippdme.client.model import IppDmeMachine
from pyippdme.exceptions import IppDmeError
from pyippdme.protocol.network import TCP_NETWORK, MemoryNetwork, Network
from pyippdme.rawdata.formats import ScanData
from pyippdme.rawdata.transfer import read_scan_data
from pyippdme.types.vec3 import Vec3

__all__ = [
    "ACQUISITION_TYPES",
    "SensorInfo",
    "acquire",
    "bounding_box",
    "path_points",
    "read_sensor_info",
]

ACQUISITION_TYPES = ("SingleShot", "MultiShot", "Sweep")


@dataclass(frozen=True, slots=True)
class SensorInfo:
    """What the server says about the active tool and its raw data."""

    tool_name: str
    tool_kind: str
    technologies: tuple[str, ...]
    data_format: str


async def read_sensor_info(machine: IppDmeMachine) -> SensorInfo:
    """Ask for the active tool and how its raw data can be fetched."""
    name = str(await machine.tool.get_name())
    tool_id = await machine.tool.get_id()
    kind = type(tool_id).__name__.removeprefix("ToolId")
    try:
        adv = await machine.raw_data.get_adv_data_struct()
    except IppDmeError:
        return SensorInfo(name, kind, (), "")
    return SensorInfo(name, kind, tuple(sorted(adv.return_technologies)), str(adv.format))


def _free_port(network: Network) -> int:
    if isinstance(network, MemoryNetwork):
        return 40000 + int(np.random.default_rng().integers(0, 20000))
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def path_points(
    start: Vec3,
    end: Vec3,
    count: int,
    *,
    primary: Vec3 = (0.0, 0.0, 1.0),
    secondary: Vec3 = (1.0, 0.0, 0.0),
) -> list[AcquisitionPoint]:
    """``count`` positions on the line from ``start`` to ``end``, all with the same orientation."""
    if count < 1:
        raise ValueError("count must be at least 1")
    fractions = [0.0] if count == 1 else [i / (count - 1) for i in range(count)]
    return [
        AcquisitionPoint(
            (
                start[0] + (end[0] - start[0]) * t,
                start[1] + (end[1] - start[1]) * t,
                start[2] + (end[2] - start[2]) * t,
            ),
            primary,
            secondary,
        )
        for t in fractions
    ]


async def acquire(
    machine: IppDmeMachine,
    host: str,
    *,
    name: str = "Acq1",
    acquisition_type: str = "SingleShot",
    settings_name: str = "Settings",
    points: Sequence[AcquisitionPoint] = (),
    step_width: float | None = None,
    data_format: str = "double",
    port: int | None = None,
    network: Network = TCP_NETWORK,
    keep: bool = False,
) -> ScanData:
    """Acquire with the active tool and return the points, in the client's coordinate system.

    ``host`` and ``network`` are where the server's raw-data port can be reached. Unless ``keep``
    is set the acquisition is deleted on the server afterwards.
    """
    if acquisition_type not in ACQUISITION_TYPES:
        raise ValueError(f"acquisition_type must be one of {ACQUISITION_TYPES}")
    chosen = port or _free_port(network)
    await machine.raw_data.data_acquire(
        name, acquisition_type, settings_name, points, step_width=step_width
    )
    try:
        await machine.raw_data.raw_data_bin_setup(data_format, chosen)
        transfer = machine.raw_data.get_raw_data_bin(name)
        data = await read_scan_data(host, chosen, network=network)
        await transfer
    finally:
        if not keep:
            await machine.raw_data.delete_acquisition(name)
    return data


def bounding_box(points: npt.NDArray[np.float64]) -> tuple[Vec3, Vec3] | None:
    """Return the smallest and largest coordinates of a point array, or ``None`` if it is empty."""
    if len(points) == 0:
        return None
    lo, hi = points.min(axis=0), points.max(axis=0)
    return (float(lo[0]), float(lo[1]), float(lo[2])), (float(hi[0]), float(hi[1]), float(hi[2]))
