# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""The Scanpoints Binary Format (SBF) and its extension ESBF (Annex C).

Both formats hold point groups, each made of point sets, each made of points,
little-endian, after a 512-byte header. SBF stores ``float`` (4 byte) values only.
ESBF adds the choice between ``float`` and ``double`` (8 byte), and lets a group
of point sets be sent without knowing their number in advance (C.2): a set with
zero points ends the group.

Invalid points are marked with the largest value of the data type in all three
coordinates; they are NaN once read here, and written as that value.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pyippdme.exceptions import IppDmeProtocolError
from pyippdme.types.vec3 import Vec3

#: The identifier every file starts with.
IDENTIFIER = b"IDME"
#: The only version defined (Annex C).
VERSION = 0
#: Length of the descriptive text in the header.
DESCRIPTION_LENGTH = 128

_FORMAT_CODES = {"float": 0, "double": 1}
_DTYPES: dict[str, np.dtype[np.floating]] = {
    "float": np.dtype("<f4"),
    "double": np.dtype("<f8"),
}
_RESERVED_GROUP_INTS = 15
_RESERVED_SET_INTS = 12


@dataclass(frozen=True, slots=True)
class PointSetData:
    """The points of one set and the scan orientation vector that applies to all of them."""

    orientation: Vec3
    points: npt.NDArray[np.float64]  # shape (n, 3)

    def __post_init__(self) -> None:
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError("points must have the shape (n, 3)")


@dataclass(frozen=True, slots=True)
class ScanData:
    """The content of an SBF or ESBF stream: point groups made of point sets."""

    groups: tuple[tuple[PointSetData, ...], ...]
    description: str = ""
    normal_direction: int = 0
    #: ``"float"`` or ``"double"``: what the stream stored the values as.
    data_format: str = "float"

    def points(self) -> npt.NDArray[np.float64]:
        """All points of all sets, one row per point."""
        parts = [ps.points for group in self.groups for ps in group]
        if not parts:
            return np.empty((0, 3), dtype=np.float64)
        return np.concatenate(parts)


def _encode_description(description: str) -> bytes:
    data = description.encode("ascii", errors="replace")[:DESCRIPTION_LENGTH]
    return data.ljust(DESCRIPTION_LENGTH, b"\0")


def _to_wire(values: npt.ArrayLike, data_format: str) -> bytes:
    dtype = _DTYPES[data_format]
    array = np.array(values, dtype=np.float64)
    array[np.isnan(array)] = np.finfo(dtype).max
    return array.astype(dtype).tobytes()


def _from_wire(payload: bytes, data_format: str) -> npt.NDArray[np.float64]:
    dtype = _DTYPES[data_format]
    array = np.frombuffer(payload, dtype=dtype).astype(np.float64)
    array[array >= float(np.finfo(dtype).max)] = np.nan
    return array


def pack_sbf(
    groups: tuple[tuple[PointSetData, ...], ...],
    *,
    description: str = "",
    normal_direction: int = 0,
) -> bytes:
    """Write ``groups`` as an SBF stream (C.1)."""
    out = bytearray()
    out += IDENTIFIER + struct.pack("<i", VERSION) + _encode_description(description)
    out += struct.pack("<ii", len(groups), normal_direction) + bytes(92 * 4)
    for group in groups:
        out += struct.pack("<i", len(group)) + bytes(_RESERVED_GROUP_INTS * 4)
        for point_set in group:
            out += _pack_set(point_set, "float")
    return bytes(out)


def pack_esbf(
    groups: tuple[tuple[PointSetData, ...], ...],
    data_format: str = "float",
    *,
    description: str = "",
    normal_direction: int = 0,
    counted: bool = True,
) -> bytes:
    """Write ``groups`` as an ESBF stream (C.2).

    With ``counted=False`` the number of point sets is not given and each group
    ends with an empty point set instead, as when the data is sent while it is
    acquired (``LiveMode(On)``, 6.17.2.1).
    """
    if data_format not in _FORMAT_CODES:
        raise ValueError(f"data_format must be 'float' or 'double', got {data_format!r}")
    out = bytearray()
    out += IDENTIFIER + struct.pack("<i", VERSION) + _encode_description(description)
    out += struct.pack("<iii", len(groups), normal_direction, _FORMAT_CODES[data_format])
    out += bytes(91 * 4)
    for group in groups:
        out += struct.pack("<ii", len(group) if counted else 0, 1 if counted else 0)
        out += bytes(_RESERVED_GROUP_INTS * 4)
        for point_set in group:
            out += _pack_set(point_set, data_format)
        if not counted:
            out += _pack_set(PointSetData((0.0, 0.0, 0.0), np.empty((0, 3))), data_format)
    return bytes(out)


def _pack_set(point_set: PointSetData, data_format: str) -> bytes:
    return (
        struct.pack("<i", len(point_set.points))
        + _to_wire(point_set.orientation, data_format)
        + bytes(_RESERVED_SET_INTS * 4)
        + _to_wire(point_set.points.reshape(-1), data_format)
    )


class _Reader:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self._offset = 0

    def take(self, size: int) -> bytes:
        end = self._offset + size
        if end > len(self._payload):
            raise IppDmeProtocolError("The scan data ends before its header says it does")
        chunk = self._payload[self._offset : end]
        self._offset = end
        return chunk

    def integers(self, count: int) -> tuple[int, ...]:
        return struct.unpack(f"<{count}i", self.take(4 * count))


def _unpack_set(reader: _Reader, data_format: str) -> PointSetData:
    size = _DTYPES[data_format].itemsize
    (count,) = reader.integers(1)
    orientation = _from_wire(reader.take(3 * size), data_format)
    reader.take(_RESERVED_SET_INTS * 4)
    points = _from_wire(reader.take(count * 3 * size), data_format).reshape(-1, 3)
    return PointSetData((orientation[0], orientation[1], orientation[2]), points)


def _read_prelude(reader: _Reader) -> str:
    """Check the identifier and version; return the descriptive text of the header."""
    if reader.take(4) != IDENTIFIER:
        raise IppDmeProtocolError("Not a scanpoints binary stream: identifier IDME is missing")
    (version,) = reader.integers(1)
    if version != VERSION:
        raise IppDmeProtocolError(f"Unsupported scanpoints binary version {version}")
    return reader.take(DESCRIPTION_LENGTH).split(b"\0", 1)[0].decode("ascii", "replace")


def unpack_sbf(payload: bytes) -> ScanData:
    """Read an SBF stream (C.1)."""
    reader = _Reader(payload)
    description = _read_prelude(reader)
    group_count, normal_direction = reader.integers(2)
    reader.take(92 * 4)
    groups = []
    for _ in range(group_count):
        (set_count,) = reader.integers(1)
        reader.take(_RESERVED_GROUP_INTS * 4)
        groups.append(tuple(_unpack_set(reader, "float") for _ in range(set_count)))
    return ScanData(tuple(groups), description, normal_direction, "float")


def unpack_esbf(payload: bytes) -> ScanData:
    """Read an ESBF stream (C.2), including one sent without counting its point sets."""
    reader = _Reader(payload)
    description = _read_prelude(reader)
    group_count, normal_direction, format_code = reader.integers(3)
    reader.take(91 * 4)
    data_format = {code: name for name, code in _FORMAT_CODES.items()}.get(format_code)
    if data_format is None:
        raise IppDmeProtocolError(f"Unknown ESBF data format {format_code}")
    groups = []
    for _ in range(group_count):
        set_count, counted = reader.integers(2)
        reader.take(_RESERVED_GROUP_INTS * 4)
        point_sets: list[PointSetData] = []
        if counted:
            point_sets = [_unpack_set(reader, data_format) for _ in range(set_count)]
        else:
            while True:
                point_set = _unpack_set(reader, data_format)
                if len(point_set.points) == 0:
                    break
                point_sets.append(point_set)
        groups.append(tuple(point_sets))
    return ScanData(tuple(groups), description, normal_direction, data_format)
