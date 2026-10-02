# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Round-trip tests for the three bundled XML schemas (Annexes E, F, G)."""

from __future__ import annotations

import pytest

from pyippdme.types import pointcloud, rawdata, tool_id


def test_point_cloud_round_trip() -> None:
    data = pointcloud.PointCloudSet(
        (
            pointcloud.PointCloud(
                (
                    pointcloud.PointSet(
                        normal=(0.0, 0.0, 1.0),
                        direction=(0.0, 0.0, -1.0),
                        points=(
                            pointcloud.MeasPoint(1.0, 2.0, 3.0, 0),
                            pointcloud.MeasPoint(4.0, 5.0, 6.0, 12),
                        ),
                    ),
                ),
            ),
        )
    )
    xml = pointcloud.to_xml(data)
    assert xml.startswith("<PointCloudSet>")
    assert pointcloud.from_xml(xml) == data


def test_point_cloud_multiple_clouds_and_sets_round_trip() -> None:
    point_set = pointcloud.PointSet(
        normal=(1.0, 0.0, 0.0),
        direction=(0.0, 1.0, 0.0),
        points=(pointcloud.MeasPoint(0.0, 0.0, 0.0, 0),),
    )
    data = pointcloud.PointCloudSet(
        (
            pointcloud.PointCloud((point_set, point_set)),
            pointcloud.PointCloud((point_set,)),
        )
    )
    assert pointcloud.from_xml(pointcloud.to_xml(data)) == data


def test_adv_data_struct_simple_format_round_trip() -> None:
    data = rawdata.AdvDataStruct(return_technologies=frozenset({"File"}), format="SBF")
    xml = rawdata.to_xml(data)
    parsed = rawdata.from_xml(xml)
    assert parsed.return_technologies == {"File"}
    assert parsed.format == "SBF"


def test_adv_data_struct_gen_format_round_trip() -> None:
    gen = rawdata.GenFormat(
        pos_information=rawdata.PosInformation(
            tcp=rawdata.Vector(1.0, 2.0, 3.0),
            ijk=rawdata.Vector(0.0, 0.0, 1.0),
        ),
        pixel_structure=rawdata.PixelStructure(bits_per_color_channel=8, num_color_channels=3),
        sample_structure=rawdata.SampleStructure(
            (rawdata.DimensionSample(10.0, 640), rawdata.DimensionSample(8.0, 480))
        ),
    )
    data = rawdata.AdvDataStruct(return_technologies=frozenset({"SocBin", "ShaMem"}), format=gen)
    parsed = rawdata.from_xml(rawdata.to_xml(data))
    assert parsed.return_technologies == {"SocBin", "ShaMem"}
    assert isinstance(parsed.format, rawdata.GenFormat)
    assert parsed.format.sample_structure.no_of_dims == 2
    assert parsed.format == gen


def test_adv_data_struct_rejects_empty_technologies() -> None:
    with pytest.raises(ValueError, match="non-empty subset"):
        rawdata.AdvDataStruct(return_technologies=frozenset(), format="XML")


def test_tool_id_tactile_round_trip() -> None:
    tool = tool_id.ToolIdTactileMeasuring(
        id="Scanning1",
        basic_functions=("ScanOnLine", "ScanOnCircle"),
        cnc_axes=("X", "Y", "Z"),
        supports_optimization_mode=True,
        align_mode=tool_id.ContinuousAlignMode(aligncaa=True),
        move_on_acquisition=False,
    )
    assert tool_id.from_xml(tool_id.to_xml(tool)) == tool


def test_tool_id_optical_with_roi_round_trip() -> None:
    tool = tool_id.ToolIdOptical2DRt(
        id="Laser1",
        basic_functions=("DataAcquire",),
        cnc_axes=("X", "Y", "Z", "A", "B"),
        supports_optimization_mode=False,
        align_mode=tool_id.IndexedAlignMode(aligncaa=False, step=5.0),
        move_on_acquisition=True,
        supports_livestream=True,
        supports_binary_socket=True,
        daq_functions=tool_id.DataAcquisitionFunctions(
            single_shot=tool_id.DataAcquisitionFunction(10.0, 10.0, None),
            sweep=tool_id.DataAcquisitionFunction(50.0, 50.0, 20.0),
        ),
        roi=tool_id.Roi(kind="circle", name="center", roi_type="inclusion", include=True),
    )
    assert tool_id.from_xml(tool_id.to_xml(tool)) == tool


def test_tool_id_rejects_unknown_roi_kind() -> None:
    with pytest.raises(ValueError, match="Unknown ROI kind"):
        tool_id.Roi(kind="hexagon", name="n", roi_type="t", include=True)
