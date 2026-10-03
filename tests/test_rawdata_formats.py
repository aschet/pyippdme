# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""The scanpoints binary formats of Annex C."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from pyippdme.exceptions import IppDmeProtocolError
from pyippdme.protocol.conformance import CONFORMANCE_CLASSES
from pyippdme.rawdata.formats import (
    IDENTIFIER,
    PointSetData,
    ScanData,
    pack_esbf,
    pack_sbf,
    unpack_esbf,
    unpack_sbf,
)

GROUPS = (
    (
        PointSetData((0.0, 0.0, 1.0), np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])),
        PointSetData((0.0, 1.0, 0.0), np.array([[7.0, 8.0, 9.0]])),
    ),
    (PointSetData((1.0, 0.0, 0.0), np.array([[10.0, 11.0, 12.0]])),),
)


def test_sbf_has_the_layout_of_table_126() -> None:
    payload = pack_sbf(GROUPS, description="hello", normal_direction=3)
    assert payload[:4] == IDENTIFIER
    assert struct.unpack_from("<i", payload, 4) == (0,)  # version
    assert payload[8:13] == b"hello"
    assert struct.unpack_from("<ii", payload, 136) == (2, 3)  # groups, normal direction
    # 512 byte header + 2 group headers of 64 bytes + 3 set headers of 64 bytes + 4 points
    assert len(payload) == 512 + 2 * 64 + 3 * 64 + 4 * 12


def test_sbf_round_trips() -> None:
    scan = unpack_sbf(pack_sbf(GROUPS, description="d", normal_direction=1))
    assert scan.description == "d"
    assert scan.normal_direction == 1
    assert scan.data_format == "float"
    assert len(scan.groups) == 2
    assert len(scan.groups[0]) == 2
    assert scan.groups[0][1].orientation == (0.0, 1.0, 0.0)
    assert scan.points().tolist() == [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0],
    ]


@pytest.mark.parametrize("data_format", ["float", "double"])
@pytest.mark.parametrize("counted", [True, False])
def test_esbf_round_trips_in_both_data_formats_and_counting_modes(
    data_format: str, counted: bool
) -> None:
    scan = unpack_esbf(pack_esbf(GROUPS, data_format, counted=counted))
    assert scan.data_format == data_format
    assert [len(group) for group in scan.groups] == [2, 1]
    assert scan.points().shape == (4, 3)
    assert scan.points()[3].tolist() == [10.0, 11.0, 12.0]


def test_esbf_header_records_the_format_and_group_headers_the_counting() -> None:
    floats = pack_esbf(GROUPS, "float")
    doubles = pack_esbf(GROUPS, "double")
    assert struct.unpack_from("<i", floats, 144) == (0,)  # 0 = float (4 bytes)
    assert struct.unpack_from("<i", doubles, 144) == (1,)  # 1 = double (8 bytes)
    assert struct.unpack_from("<ii", floats, 512) == (2, 1)  # two sets, count is valid
    assert struct.unpack_from("<ii", pack_esbf(GROUPS, counted=False), 512) == (0, 0)


def test_a_live_esbf_group_ends_with_an_empty_point_set() -> None:
    """C.2: "The transfer ends when zero length pointset is received"."""
    payload = pack_esbf(((GROUPS[0][1],),), "float", counted=False)
    assert payload.endswith(struct.pack("<i", 0) + bytes(12) + bytes(48))


def test_invalid_points_are_the_largest_value_of_the_type_and_come_back_as_nan() -> None:
    nan_set = PointSetData((0.0, 0.0, 1.0), np.array([[1.0, 2.0, 3.0], [np.nan] * 3]))
    payload = pack_esbf(((nan_set,),), "double")
    assert struct.unpack_from("<3d", payload, len(payload) - 24) == (np.finfo("<f8").max,) * 3
    points = unpack_esbf(payload).points()
    assert np.isnan(points[1]).all()
    assert points[0].tolist() == [1.0, 2.0, 3.0]
    assert np.isnan(unpack_sbf(pack_sbf(((nan_set,),))).points()[1]).all()


def test_the_description_is_cut_to_128_characters() -> None:
    scan = unpack_esbf(pack_esbf(GROUPS, description="x" * 200))
    assert scan.description == "x" * 128


def test_bad_input_is_rejected() -> None:
    with pytest.raises(ValueError, match="data_format"):
        pack_esbf(GROUPS, "triple")
    with pytest.raises(IppDmeProtocolError, match="IDME"):
        unpack_esbf(b"JUNK" + bytes(600))
    with pytest.raises(IppDmeProtocolError, match="ends before"):
        unpack_esbf(pack_esbf(GROUPS)[:-1])
    with pytest.raises(IppDmeProtocolError, match="version"):
        unpack_sbf(IDENTIFIER + struct.pack("<i", 7) + bytes(600))
    bad_format = bytearray(pack_esbf(GROUPS))
    bad_format[144:148] = struct.pack("<i", 9)
    with pytest.raises(IppDmeProtocolError, match="data format"):
        unpack_esbf(bytes(bad_format))
    with pytest.raises(ValueError, match="shape"):
        PointSetData((0.0, 0.0, 1.0), np.zeros((2, 2)))


def test_scan_data_without_points_is_empty() -> None:
    assert ScanData(()).points().shape == (0, 3)


def test_the_conformance_classes_of_annex_j1() -> None:
    assert [c.machine_class for c in CONFORMANCE_CLASSES] == [
        "CartCMM_TouchTrigger_Fixed_None_ToolChanger_None_None",
        "CartCMM_TouchTrigger_Fixed_Cartesian_ToolChanger_None_None",
        "CartCMM_TouchTrigger_Alignable_AB_Cartesian_ToolChanger_None_None",
        "CartCMM_Scanning_Fixed_Cartesian_ToolChanger_None_None",
        "CartCMM_Scanning_Fixed_Cartesian_ToolChanger_None_RotaryTable",
    ]
    assert CONFORMANCE_CLASSES[2].description == "CNC machine with tactile probe and CAA Head"
