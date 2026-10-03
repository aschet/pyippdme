# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for pyippdme.server.builders: typed HandlerResult construction for command responses."""

from __future__ import annotations

from pyippdme.protocol.ast import (
    BasicName,
    Items,
    NamedValue,
    NameValue,
    Number,
    NumericData,
    PropertyData,
    String,
    StringValue,
)
from pyippdme.protocol.commands import CommandName
from pyippdme.protocol.parameters import ParameterName
from pyippdme.protocol.signature import DataType, Parameter
from pyippdme.server import builders
from pyippdme.types.csy import CoordinateTransform


def test_bare_names_builds_one_empty_named_value_per_name() -> None:
    assert builders.bare_names("X", "Y") == Items((NamedValue("X", ()), NamedValue("Y", ())))


def test_bare_names_with_nothing_is_empty() -> None:
    assert builders.bare_names() == Items(())


def test_named_numbers_builds_one_named_value_per_keyword() -> None:
    assert builders.named_numbers(X=10, Y=20) == Items(
        (NamedValue("X", (Number.of(10),)), NamedValue("Y", (Number.of(20),)))
    )


def test_number_and_boolean_build_bare_numeric_data() -> None:
    assert builders.number(3.5) == NumericData((Number.of(3.5),))
    assert builders.boolean(True) == NumericData((Number.of(1),))
    assert builders.boolean(False) == NumericData((Number.of(0),))


def test_named_boolean_builds_a_single_named_field() -> None:
    assert builders.named_boolean(CommandName.IS_HOMED, True) == Items(
        (NamedValue(CommandName.IS_HOMED, (Number.of(1),)),)
    )


def test_name_value_and_string_value_wrap_a_bare_scalar() -> None:
    assert builders.name_value("Machine") == NameValue("Machine")
    assert builders.string_value("A CMM") == StringValue(String("A CMM"))


def test_string_list_and_name_list_build_one_message_per_value() -> None:
    assert builders.string_list(["a", "b"]) == [StringValue(String("a")), StringValue(String("b"))]
    assert builders.name_list(["Foo", "Bar"]) == [NameValue("Foo"), NameValue("Bar")]


def test_property_entry_builds_a_name_kind_pair() -> None:
    assert builders.property_entry("Speed", DataType.FLOAT) == PropertyData(
        String("Speed"), String(DataType.FLOAT)
    )


def test_supported_arguments_builds_one_pair_per_parameter() -> None:
    parameters = (Parameter("X", DataType.FLOAT), Parameter("Y", DataType.FLOAT))
    assert builders.supported_arguments(parameters) == [
        PropertyData(String("X"), String(DataType.FLOAT)),
        PropertyData(String("Y"), String(DataType.FLOAT)),
    ]


def test_get_dme_version_builds_a_single_named_string() -> None:
    assert builders.get_dme_version("6.0") == Items(
        (NamedValue(CommandName.GET_DME_VERSION, (String("6.0"),)),)
    )


def test_get_xtd_err_status_with_no_error() -> None:
    assert builders.get_xtd_err_status(homed=True) == [
        Items((NamedValue(CommandName.IS_HOMED, (Number.of(1),)),))
    ]


def test_get_xtd_err_status_with_an_active_error() -> None:
    lines = builders.get_xtd_err_status(homed=False, error=(504, 2))
    assert lines == [
        Items((NamedValue(CommandName.IS_HOMED, (Number.of(0),)),)),
        Items(
            (
                NamedValue(ParameterName.ACTIVE_ERROR, (Number.of(504),)),
                NamedValue(ParameterName.SEVERITY, (Number.of(2),)),
            )
        ),
    ]


def test_csy_transformation_builds_the_six_fixed_fields() -> None:
    transform = CoordinateTransform(1, 2, 3, 45, 90, 180)
    assert builders.csy_transformation(transform) == Items(
        (
            NamedValue(ParameterName.X0, (Number.of(1),)),
            NamedValue(ParameterName.Y0, (Number.of(2),)),
            NamedValue(ParameterName.Z0, (Number.of(3),)),
            NamedValue(ParameterName.THETA, (Number.of(45),)),
            NamedValue(ParameterName.PSI, (Number.of(90),)),
            NamedValue(ParameterName.PHI, (Number.of(180),)),
        )
    )


def test_temperature_sensor_omits_scale_axis_when_unset() -> None:
    assert builders.temperature_sensor("T1", "Air", cmm_temp_correction=True) == Items(
        (
            NamedValue(ParameterName.NAME, (String("T1"),)),
            NamedValue(ParameterName.KIND, (BasicName("Air"),)),
            NamedValue(ParameterName.CMM_TEMP_CORRECTION, (Number.of(1),)),
        )
    )


def test_temperature_sensor_includes_scale_axis_when_set() -> None:
    result = builders.temperature_sensor("T2", "Scale", cmm_temp_correction=False, scale_axis="X")
    assert result == Items(
        (
            NamedValue(ParameterName.NAME, (String("T2"),)),
            NamedValue(ParameterName.KIND, (BasicName("Scale"),)),
            NamedValue(ParameterName.CMM_TEMP_CORRECTION, (Number.of(0),)),
            NamedValue(ParameterName.SCALE, (String("X"),)),
        )
    )


def test_temperature_reading_builds_a_name_temperature_pair() -> None:
    assert builders.temperature_reading("T1", 21.5) == Items(
        (
            NamedValue(ParameterName.NAME, (String("T1"),)),
            NamedValue(ParameterName.TEMPERATURE, (Number.of(21.5),)),
        )
    )


def test_align_part_reports_both_vectors_with_a_zero_z() -> None:
    result = builders.align_part((1.0, 0.0), (0.0, 1.0))
    assert result == NumericData(tuple(Number.of(v) for v in (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)))


def test_get_change_tool_action_defaults_to_switch() -> None:
    assert builders.get_change_tool_action(1.0, 2.0, 3.0) == Items(
        (
            NamedValue(ParameterName.ARGUMENT, (BasicName("Switch"),)),
            NamedValue(ParameterName.X, (Number.of(1.0),)),
            NamedValue(ParameterName.Y, (Number.of(2.0),)),
            NamedValue(ParameterName.Z, (Number.of(3.0),)),
        )
    )


def test_get_raw_data_sha_mem_builds_the_three_fixed_fields() -> None:
    assert builders.get_raw_data_sha_mem("Seg", 0, 1024) == Items(
        (
            NamedValue(ParameterName.SHA_MEM_NAME, (String("Seg"),)),
            NamedValue(ParameterName.DATA_SEGMENT, (Number.of(0),)),
            NamedValue(ParameterName.SIZE, (Number.of(1024),)),
        )
    )


def test_get_raw_data_file_builds_a_single_named_url() -> None:
    assert builders.get_raw_data_file("file:///tmp/x.bin") == Items(
        (NamedValue(ParameterName.FILE_URL, (String("file:///tmp/x.bin"),)),)
    )
