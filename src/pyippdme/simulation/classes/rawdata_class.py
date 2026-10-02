# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""A subset of the ``Optical``/``RawDataHandling`` command classes (6.15, 6.17).

Implements ``DataAcquire``/``DeleteAcquistion``/``DeleteAllAcquisitions``
(6.15.1, ``DataAcquire`` simplified - see below) and the full
``RawDataHandling`` retrieval surface (6.17.2): ``AdvDataStruct`` plus all
three transfer technologies it can advertise - binary socket
(``RawDataBinSetup``/``GetRawDataBin``), OS shared memory
(``GetRawDataShaMem``/``ReleaseShaMem``), and file
(``GetRawDataFile``/``DelRawDataFile``); see :mod:`pyippdme.rawdata.transfer`
for the underlying mechanics and :mod:`pyippdme.types.pointcloud` for the XML
payload shape (Annex E) these acquisitions are made of.

Scope note on ``DataAcquire``: there is no simulated optical sensor or real
surface, so "acquiring" synthesizes a plausible point cloud from whatever
control points the client sent (or the current TCP position, for a
0-position single shot) rather than measuring anything real - enough to
exercise ``AdvDataStruct``/``GetRawDataBin``/``GetRawDataShaMem``/
``GetRawDataFile`` end to end, and to give a point-cloud consumer (a test, a
viewer) something resembling real scan data rather than its own input
echoed back verbatim. ``"SingleShot"``/``"MultiShot"`` report exactly the
requested positions (each with a little measurement noise, see
:func:`_synthesize_measurement_points`); ``"Sweep"`` (a continuous scanning
motion, 6.15.1) instead densifies the path between consecutive
control points into a line of points, standing in for a real line/area
optical scan. ``AcquiseSettingsName`` is accepted but unused (there is no
settings-container store), and ``S``/``RT`` are parsed but not applied to
the (nonexistent) sensor motion.
"""

from __future__ import annotations

import itertools
import random

from pyippdme.protocol.ast import Argument, BasicName, Number, String, Xml
from pyippdme.protocol.commands import CommandName
from pyippdme.protocol.errors import ErrorCode, ErrorSeverity, ServerError
from pyippdme.protocol.signature import DataType, Parameter
from pyippdme.rawdata.transfer import (
    default_raw_data_directory,
    flatten_points,
    pack_samples,
    port_is_available,
    publish_shared_memory,
    send_once,
    write_raw_data_file,
)
from pyippdme.server import builders
from pyippdme.server._util import bad_argument
from pyippdme.server.registry import CommandRegistry, HandlerResult
from pyippdme.simulation.context import Ctx
from pyippdme.types.pointcloud import MeasPoint, PointCloud, PointCloudSet, PointSet
from pyippdme.types.pointcloud import to_xml as point_cloud_to_xml
from pyippdme.types.rawdata import (
    AdvDataStruct,
    GenFormat,
    PixelStructure,
    PosInformation,
    SampleStructure,
    Vector,
)
from pyippdme.types.rawdata import to_xml as adv_data_struct_to_xml

_DATA_ACQUIRE_PARAMS = (
    Parameter("AcqName", DataType.STRING, positional=True),
    Parameter("Type", DataType.NAME, positional=True),
    Parameter("AcquiseSettingsName", DataType.STRING, positional=True),
    Parameter("n", DataType.INT, positional=True),
)
_ACQ_NAME_PARAMS = (Parameter("AcqName", DataType.STRING, positional=True),)
_RAW_DATA_BIN_SETUP_PARAMS = (
    Parameter("RawDataFormat", DataType.NAME, positional=True),
    Parameter("Port", DataType.INT, positional=True),
    Parameter("LiveMode", DataType.NAME, positional=True),
)
_ACQUISITION_TYPES = ("SingleShot", "MultiShot", "Sweep")
#: A plausible order-of-magnitude figure for an optical/laser point sensor's
#: measurement noise (single-digit micrometers is typical across vendors -
#: this is not any specific sensor's measured spec), so repeated
#: acquisitions of the same nominal path don't look suspiciously identical.
_OPTICAL_NOISE_STD_MM = 0.005
#: How many points a "Sweep" densifies each control-point segment into,
#: standing in for a real continuous line/area optical scan.
_SWEEP_POINTS_PER_SEGMENT = 20


def _synthesize_measurement_points(
    acq_name: str,
    acquisition_type: str,
    positions: list[tuple[float, float, float]],
) -> tuple[MeasPoint, ...]:
    """Turn ``DataAcquire``'s requested control points into a synthetic point cloud.

    ``"Sweep"`` linearly interpolates :data:`_SWEEP_POINTS_PER_SEGMENT`
    points between each consecutive pair of control points - a demo line
    scan - rather than reporting only the (few) points the client actually
    sent; ``"SingleShot"``/``"MultiShot"`` report exactly the requested
    positions. Every point gets independent Gaussian noise
    (:data:`_OPTICAL_NOISE_STD_MM`) on each axis. Deterministic per
    ``acq_name`` (:class:`random.Random` seeded from it, not affected by
    ``PYTHONHASHSEED``), so acquiring the same name twice reproduces the
    same cloud - useful for a reproducible demo/test fixture.
    """
    if acquisition_type == "Sweep" and len(positions) >= 2:
        path: list[tuple[float, float, float]] = []
        for start, end in itertools.pairwise(positions):
            for step in range(_SWEEP_POINTS_PER_SEGMENT):
                t = step / _SWEEP_POINTS_PER_SEGMENT
                path.append(
                    (
                        start[0] + (end[0] - start[0]) * t,
                        start[1] + (end[1] - start[1]) * t,
                        start[2] + (end[2] - start[2]) * t,
                    )
                )
        path.append(positions[-1])
    else:
        path = list(positions)
    rng = random.Random(acq_name)  # noqa: S311 # nosec B311 (demo data, not cryptographic)
    return tuple(
        MeasPoint(
            x + rng.gauss(0.0, _OPTICAL_NOISE_STD_MM),
            y + rng.gauss(0.0, _OPTICAL_NOISE_STD_MM),
            z + rng.gauss(0.0, _OPTICAL_NOISE_STD_MM),
            0,
        )
        for x, y, z in path
    )


#: What this server advertises via ``GetProp(Tool.AdvDataStruct())`` - all
#: three transfer technologies, a generic (TCP-only) sample format.
_ADV_DATA_STRUCT = AdvDataStruct(
    return_technologies=frozenset({"SocBin", "ShaMem", "File"}),
    format=GenFormat(
        pos_information=PosInformation(tcp=Vector(0.0, 0.0, 0.0)),
        pixel_structure=PixelStructure(bits_per_color_channel=0, num_color_channels=0),
        sample_structure=SampleStructure(()),
    ),
)


async def _adv_data_struct(_ctx: Ctx, _args: tuple[Argument, ...]) -> HandlerResult:
    return Xml(adv_data_struct_to_xml(_ADV_DATA_STRUCT))


async def _data_acquire(ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
    if (
        len(args) < 4
        or not isinstance(args[0], String)
        or not isinstance(args[1], BasicName)
        or not isinstance(args[2], String)
        or not isinstance(args[3], Number)
    ):
        raise bad_argument(CommandName.DATA_ACQUIRE)
    acq_name = args[0].value
    acquisition_type = args[1].value
    if acquisition_type not in _ACQUISITION_TYPES:
        raise bad_argument(CommandName.DATA_ACQUIRE, "Type must be SingleShot, MultiShot or Sweep")
    n = int(args[3].value)
    if n < 0:
        raise bad_argument(CommandName.DATA_ACQUIRE, "n must not be negative")

    remaining = args[4:]
    vector_count = 6 * n
    if len(remaining) not in (vector_count, vector_count + 1, vector_count + 2):
        raise bad_argument(
            CommandName.DATA_ACQUIRE, "Wrong number of position/orientation arguments"
        )
    if not all(isinstance(a, Number) for a in remaining):
        raise bad_argument(CommandName.DATA_ACQUIRE, "Positions/orientations must be plain numbers")
    numbers = [a.value for a in remaining if isinstance(a, Number)]

    if n == 0:
        positions = [ctx.state.cart_cmm.position]
        directions = [(0.0, 0.0, 0.0)]
    else:
        positions = []
        directions = []
        for i in range(n):
            base = i * 6
            positions.append((numbers[base], numbers[base + 1], numbers[base + 2]))
            directions.append((numbers[base + 3], numbers[base + 4], numbers[base + 5]))

    point_set = PointSet(
        normal=(0.0, 0.0, 1.0),
        direction=directions[0],
        points=_synthesize_measurement_points(acq_name, acquisition_type, positions),
    )
    ctx.state.raw_data.acquisitions[acq_name] = PointCloudSet((PointCloud((point_set,)),))
    return None


def _require_acquisition(
    ctx: Ctx, args: tuple[Argument, ...], cause: str
) -> tuple[str, PointCloudSet]:
    if len(args) != 1 or not isinstance(args[0], String):
        raise bad_argument(cause)
    acq_name = args[0].value
    data = ctx.state.raw_data.acquisitions.get(acq_name)
    if data is None:
        raise ServerError(
            ErrorSeverity.CRITICAL,
            ErrorCode.RAW_DATA_NOT_AVAILABLE,
            cause,
            "Raw data of Acquisition not available",
        )
    return acq_name, data


async def _raw_data_bin_setup(ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
    if (
        len(args) != 3
        or not isinstance(args[0], BasicName)
        or not isinstance(args[1], Number)
        or not isinstance(args[2], BasicName)
    ):
        raise bad_argument(CommandName.RAW_DATA_BIN_SETUP)
    data_format, port, live_mode = args[0].value, int(args[1].value), args[2].value
    if data_format not in ("float", "double"):
        raise ServerError(
            ErrorSeverity.ERROR,
            ErrorCode.WRONG_DATA_FORMAT,
            CommandName.RAW_DATA_BIN_SETUP,
            "Wrong data format",
        )
    if live_mode not in ("On", "Off"):
        raise bad_argument(CommandName.RAW_DATA_BIN_SETUP, "LiveMode must be On or Off")
    if not await port_is_available("0.0.0.0", port):  # noqa: S104 # nosec B104
        raise ServerError(
            ErrorSeverity.ERROR,
            ErrorCode.PORT_NOT_AVAILABLE,
            CommandName.RAW_DATA_BIN_SETUP,
            "Port not available",
        )
    ctx.state.raw_data.bin_format = data_format
    ctx.state.raw_data.bin_port = port
    return None


async def _get_raw_data_bin(ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
    _acq_name, data = _require_acquisition(ctx, args, CommandName.GET_RAW_DATA_BIN)
    if ctx.state.raw_data.bin_format is None or ctx.state.raw_data.bin_port is None:
        raise ServerError(
            ErrorSeverity.CRITICAL,
            ErrorCode.BAD_CONTEXT,
            CommandName.GET_RAW_DATA_BIN,
            "RawDataBinSetup was not called",
        )
    payload = pack_samples(flatten_points(data), ctx.state.raw_data.bin_format)
    # "0.0.0.0": accept the client's connection to the negotiated port on any
    # interface, matching the "different computers" cross-machine use case
    # RawDataBinSetup's Port parameter exists for.
    await send_once("0.0.0.0", ctx.state.raw_data.bin_port, payload)  # noqa: S104 # nosec B104
    return None


async def _get_raw_data_sha_mem(ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
    acq_name, data = _require_acquisition(ctx, args, CommandName.GET_RAW_DATA_SHA_MEM)
    payload = pack_samples(flatten_points(data), "double")
    try:
        segment = publish_shared_memory(payload)
    except OSError as exc:
        raise ServerError(
            ErrorSeverity.CRITICAL,
            ErrorCode.NO_SHARED_MEMORY_AVAILABLE,
            CommandName.GET_RAW_DATA_SHA_MEM,
            "No shared memory space available",
        ) from exc
    ctx.state.raw_data.shared_memory_segments[acq_name] = segment
    # "Hex coded offset"/"Hex coded size" (Table 101) describes how a C/C++
    # client typically reads these values, not a distinct wire number type -
    # The standard's [int] grammar has no separate hex literal, so these are
    # sent as ordinary decimal Number values.
    return builders.get_raw_data_sha_mem(segment.name, 0, len(payload))


async def _release_sha_mem(ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
    if len(args) != 1 or not isinstance(args[0], String):
        raise bad_argument(CommandName.RELEASE_SHA_MEM)
    segment = ctx.state.raw_data.shared_memory_segments.pop(args[0].value, None)
    if segment is not None:
        segment.close()
        segment.unlink()
    return None


async def _get_raw_data_file(ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
    acq_name, data = _require_acquisition(ctx, args, CommandName.GET_RAW_DATA_FILE)
    directory = default_raw_data_directory()
    payload = point_cloud_to_xml(data).encode("utf-8")
    path = write_raw_data_file(directory, acq_name, payload, ".xml")
    ctx.state.raw_data.files[acq_name] = path
    # FileURL is documented as kind [name] (Table 103), but a file:// URL
    # contains characters (':', '/') the [name] grammar (a bare identifier)
    # cannot represent; sent as a quoted [string] instead.
    return builders.get_raw_data_file(path.as_uri())


async def _del_raw_data_file(ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
    if len(args) != 1 or not isinstance(args[0], String):
        raise bad_argument(CommandName.DEL_RAW_DATA_FILE)
    path = ctx.state.raw_data.files.pop(args[0].value, None)
    if path is not None and path.is_file():
        path.unlink()
    return None


async def _delete_acquisition(ctx: Ctx, args: tuple[Argument, ...]) -> HandlerResult:
    if len(args) != 1 or not isinstance(args[0], String):
        raise bad_argument(CommandName.DELETE_ACQUISITION)
    # No error is defined for an unknown Name (Table 96); a no-op, like the
    # other by-name acquisition/file/shared-memory cleanup commands above.
    ctx.state.raw_data.acquisitions.pop(args[0].value, None)
    return None


async def _delete_all_acquisitions(ctx: Ctx, _args: tuple[Argument, ...]) -> HandlerResult:
    ctx.state.raw_data.acquisitions.clear()
    return None


def _release_all_acquisitions_on_end_session(ctx: Ctx) -> None:
    # 6.15.1: "EndSession() implicitly deletes all buffered acquisitions" -
    # including whatever they were retrieved as (shared memory, temp files).
    ctx.state.raw_data.acquisitions.clear()
    for segment in ctx.state.raw_data.shared_memory_segments.values():
        segment.close()
        segment.unlink()
    ctx.state.raw_data.shared_memory_segments.clear()
    for path in ctx.state.raw_data.files.values():
        if path.is_file():
            path.unlink()
    ctx.state.raw_data.files.clear()


def register(registry: CommandRegistry) -> None:
    registry.register(CommandName.DATA_ACQUIRE, _data_acquire, arguments=_DATA_ACQUIRE_PARAMS)
    registry.register(CommandName.ADV_DATA_STRUCT, _adv_data_struct, arguments=())
    registry.register(
        CommandName.RAW_DATA_BIN_SETUP, _raw_data_bin_setup, arguments=_RAW_DATA_BIN_SETUP_PARAMS
    )
    registry.register(CommandName.GET_RAW_DATA_BIN, _get_raw_data_bin, arguments=_ACQ_NAME_PARAMS)
    registry.register(
        CommandName.GET_RAW_DATA_SHA_MEM, _get_raw_data_sha_mem, arguments=_ACQ_NAME_PARAMS
    )
    registry.register(CommandName.RELEASE_SHA_MEM, _release_sha_mem, arguments=_ACQ_NAME_PARAMS)
    registry.register(CommandName.GET_RAW_DATA_FILE, _get_raw_data_file, arguments=_ACQ_NAME_PARAMS)
    registry.register(CommandName.DEL_RAW_DATA_FILE, _del_raw_data_file, arguments=_ACQ_NAME_PARAMS)
    registry.register(
        CommandName.DELETE_ACQUISITION, _delete_acquisition, arguments=_ACQ_NAME_PARAMS
    )
    registry.register(CommandName.DELETE_ALL_ACQUISITIONS, _delete_all_acquisitions, arguments=())
    registry.register_session_end_hook(_release_all_acquisitions_on_end_session)
