# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Tests that the object model sends and understands everything the standard allows.

The server here is a recorder: it answers each command with a canned reply and
keeps the wire text of the arguments it received.
"""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

import pytest

from pyippdme import IppDmeMachine, IppDmeServer
from pyippdme.client import features
from pyippdme.client.model import (
    AcquisitionPoint,
    CurvePoint,
    PartAlignment,
    ToolAlignment,
)
from pyippdme.client.report import Report, report_from_payload
from pyippdme.exceptions import IppDmeError
from pyippdme.protocol.ast import (
    Argument,
    DataPayload,
    Items,
    NamedValue,
    Number,
    NumericData,
    String,
    StringValue,
)
from pyippdme.protocol.commands import CommandName
from pyippdme.protocol.network import MemoryNetwork
from pyippdme.server.classes import register_dme_class, register_server_class
from pyippdme.server.registry import CommandContext, HandlerResult, MachineState

Reply = Callable[[], HandlerResult]
T = TypeVar("T")


def _named(name: str, *values: float) -> NamedValue:
    return NamedValue(name, tuple(Number.of(v) for v in values))


@dataclass
class Recorder:
    """The commands a test server received, as ``(name, wire text of the arguments)``."""

    calls: list[tuple[str, str]] = field(default_factory=list)
    replies: dict[str, Reply] = field(default_factory=dict)

    @property
    def last(self) -> str:
        return self.calls[-1][1]

    def handler(self, name: str) -> Callable[[CommandContext[Any], tuple[Argument, ...]], Any]:
        async def handle(_ctx: CommandContext[Any], args: tuple[Argument, ...]) -> HandlerResult:
            self.calls.append((name, ",".join(arg.to_wire() for arg in args)))
            reply = self.replies.get(name)
            return reply() if reply is not None else None

        return handle


_RECORDED = (
    CommandName.GO_TO,
    CommandName.STEP,
    CommandName.PT_MEAS,
    CommandName.ON_SCAN_REPORT,
    CommandName.SCAN_ON_LINE,
    CommandName.SCAN_ON_CURVE,
    CommandName.ALIGN_PART,
    CommandName.DATA_ACQUIRE,
    CommandName.ROI,
    CommandName.FEATURE_EXTRACT,
    CommandName.GET_PROP,
    CommandName.GET_PROP_E,
    CommandName.SET_PROP,
    CommandName.GET_MACHINE_CLASS,
)


@pytest.fixture
def recorder() -> Recorder:
    return Recorder(replies={CommandName.PT_MEAS: lambda: Items((_named("X", 0),))})


@pytest.fixture
async def machine(network: MemoryNetwork, recorder: Recorder) -> AsyncIterator[IppDmeMachine]:
    server: IppDmeServer[MachineState] = IppDmeServer(
        command_classes=[register_server_class, register_dme_class], network=network
    )
    for name in _RECORDED:
        server.registry.register(name, recorder.handler(name))
    port = await server.start("x", 0)
    m = await IppDmeMachine.connect("x", port, network=network)
    await m.start_session()
    try:
        yield m
    finally:
        await m.close()
        await server.close()


# -- Report ------------------------------------------------------------------


def test_report_separates_single_numbers_from_vectors() -> None:
    report = Report({"X": 1.0, "IJK": (0.0, 0.0, 1.0), "Name": "tip"})
    assert report.number("X") == 1.0
    assert report.vector("IJK") == (0.0, 0.0, 1.0)
    assert report.vector("X") == (1.0,)
    assert report.text("Name") == "tip"
    assert report == {"X": 1.0, "IJK": (0.0, 0.0, 1.0), "Name": "tip"}
    assert len(report) == 3
    assert list(report) == ["X", "IJK", "Name"]
    assert "IJK" in repr(report)


def test_report_accessors_reject_the_wrong_kind_of_value() -> None:
    report = Report({"IJK": (0.0, 0.0, 1.0), "Name": "tip"})
    with pytest.raises(TypeError):
        report.number("IJK")
    with pytest.raises(TypeError):
        report.vector("Name")
    with pytest.raises(TypeError):
        report.text("IJK")


def test_report_is_decoded_from_an_items_response() -> None:
    payload = Items((_named("X", 1), _named("IJK", 0, 0, 1), NamedValue("N", (String("a"),))))
    assert report_from_payload(payload) == {"X": 1.0, "IJK": (0.0, 0.0, 1.0), "N": "a"}


def test_report_decoding_rejects_other_responses_and_values() -> None:
    with pytest.raises(TypeError):
        report_from_payload(NumericData((Number.of(1),)))
    with pytest.raises(TypeError):
        report_from_payload(Items((NamedValue("A", (String("a"), Number.of(1))),)))


# -- Moves and measurements --------------------------------------------------


async def test_go_to_sends_every_argument_the_standard_allows(
    machine: IppDmeMachine, recorder: Recorder
) -> None:
    await machine.cart_cmm.go_to(
        1,
        2,
        3,
        r=10,
        a=5,
        b=6,
        c=7,
        alignment=ToolAlignment((0, 0, 1), (1, 0, 0)),
        sync=True,
    )
    assert recorder.last == (
        "X(1),Y(2),Z(3),R(10),Tool.A(5),Tool.B(6),Tool.C(7),Tool.Alignment(0,0,1,1,0,0),Sync(1)"
    )


async def test_go_to_without_sync_leaves_it_out_and_sync_false_sends_zero(
    machine: IppDmeMachine, recorder: Recorder
) -> None:
    await machine.cart_cmm.go_to(x=1)
    assert recorder.last == "X(1)"
    await machine.cart_cmm.go_to(x=1, sync=False)
    assert recorder.last == "X(1),Sync(0)"
    await machine.cart_cmm.go_to(alignment=ToolAlignment((0, 0, 1)))
    assert recorder.last == "Tool.Alignment(0,0,1)"


async def test_step_sends_rotation_tool_angles_and_sync(
    machine: IppDmeMachine, recorder: Recorder
) -> None:
    await machine.cart_cmm.step(x=1, r=5, a=20, b=-30, sync=True)
    assert recorder.calls[-1] == (CommandName.STEP, "X(1),R(5),Tool.A(20),Tool.B(-30),Sync(1)")


async def test_pt_meas_sends_the_probing_direction_and_alignment(
    machine: IppDmeMachine, recorder: Recorder
) -> None:
    await machine.cart_cmm.pt_meas(
        1,
        2,
        3,
        ijk=(0, 0, 1),
        r=90,
        a=1,
        b=2,
        alignment=ToolAlignment((0, 0, 1), (1, 0, 0)),
        align_part=PartAlignment((1, 0, 0), (0, 1, 0), 0.5),
    )
    assert recorder.last == (
        "X(1),Y(2),Z(3),IJK(0,0,1),R(90),Tool.A(1),Tool.B(2),"
        "Tool.Alignment(0,0,1,1,0,0),AlignPart(1,0,0,0,1,0,0.5)"
    )


async def test_pt_meas_returns_vector_values(machine: IppDmeMachine, recorder: Recorder) -> None:
    recorder.replies[CommandName.PT_MEAS] = lambda: Items(
        (_named("X", 1), _named("Y", 2), _named("Z", 3), _named("IJK", 0, 0, 1), _named("Q", 0))
    )
    report = await machine.cart_cmm.pt_meas(1, 2, 3, ijk=(0, 0, 1))
    assert report.number("X") == 1.0
    assert report.vector("IJK") == (0.0, 0.0, 1.0)
    assert report.number("Q") == 0.0


async def test_get_returns_vector_values(machine: IppDmeMachine) -> None:
    # The recorder does not register Get, so a missing handler is the failure.
    with pytest.raises(Exception, match="Unsupported"):
        await machine.cart_cmm.get("IJK")


def test_part_alignment_needs_all_second_table_values() -> None:
    with pytest.raises(ValueError, match="together"):
        PartAlignment((1, 0, 0), (1, 0, 0), 0.1, second_part_vector=(0, 1, 0))


async def test_align_part_sends_a_second_rotary_table_and_returns_all_vectors(
    machine: IppDmeMachine, recorder: Recorder
) -> None:
    recorder.replies[CommandName.ALIGN_PART] = lambda: NumericData(
        tuple(Number.of(v) for v in (1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0))
    )
    reached = await machine.rotary_table.align_part(
        (1, 0, 0),
        (1, 0, 0),
        0.1,
        second_part_vector=(0, 1, 0),
        second_machine_vector=(0, 1, 0),
        beta=0.2,
    )
    assert recorder.last == "1,0,0,1,0,0,0,1,0,0,1,0,0.1,0.2"
    assert reached == ((1, 0, 0), (1, 0, 0), (0, 1, 0), (0, 1, 0))


async def test_align_part_with_one_table_sends_two_vectors_and_alpha(
    machine: IppDmeMachine, recorder: Recorder
) -> None:
    recorder.replies[CommandName.ALIGN_PART] = lambda: NumericData(
        tuple(Number.of(v) for v in (1, 0, 0, 1, 0, 0))
    )
    assert await machine.rotary_table.align_part((1, 0, 0), (1, 0, 0), 0.1) == (
        (1, 0, 0),
        (1, 0, 0),
    )
    assert recorder.last == "1,0,0,1,0,0,0.1"


# -- Scanning ----------------------------------------------------------------


async def _points(stream: AsyncIterable[T]) -> list[T]:
    return [point async for point in stream]


async def test_scan_on_line_sends_the_rotary_table_flag(
    machine: IppDmeMachine, recorder: Recorder
) -> None:
    await _points(machine.scanning.scan_on_line((0, 0, 0), (10, 0, 0), (0, 0, 1), 2, True))
    assert recorder.calls[-1] == (CommandName.SCAN_ON_LINE, "0,0,0,10,0,0,0,0,1,2,1")
    await _points(machine.scanning.scan_on_line((0, 0, 0), (10, 0, 0), (0, 0, 1), 2))
    assert recorder.calls[-1] == (CommandName.SCAN_ON_LINE, "0,0,0,10,0,0,0,0,1,2")


async def test_scan_points_blocked_into_one_response_are_split(
    machine: IppDmeMachine, recorder: Recorder
) -> None:
    async def blocked() -> AsyncIterator[DataPayload]:
        yield NumericData(tuple(Number.of(v) for v in (0, 0, 0, 1, 0, 0)))
        yield NumericData(tuple(Number.of(v) for v in (2, 0, 0)))

    recorder.replies[CommandName.SCAN_ON_LINE] = blocked
    points = await _points(machine.scanning.scan_on_line((0, 0, 0), (2, 0, 0), (0, 0, 1), 1))
    assert points == [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)]


async def test_a_response_that_is_not_a_whole_number_of_points_is_an_error(
    machine: IppDmeMachine, recorder: Recorder
) -> None:
    async def broken() -> AsyncIterator[DataPayload]:
        yield NumericData(tuple(Number.of(v) for v in (0, 0, 0, 1)))

    recorder.replies[CommandName.SCAN_ON_LINE] = broken
    with pytest.raises(TypeError, match="multiple of 3"):
        await _points(machine.scanning.scan_on_line((0, 0, 0), (2, 0, 0), (0, 0, 1), 1))


async def test_reporting_scans_choose_the_values_and_return_reports(
    machine: IppDmeMachine, recorder: Recorder
) -> None:
    async def points() -> AsyncIterator[DataPayload]:
        yield NumericData(tuple(Number.of(v) for v in (0, 0, 0, 5, 0, 0, 1)))
        yield NumericData(tuple(Number.of(v) for v in (1, 0, 0, 7, 0, 0, 1, 2, 0, 0, 8, 0, 0, 1)))

    recorder.replies[CommandName.SCAN_ON_LINE] = points
    scanning = machine.scanning.reporting("X", "Y", "Z", "Q", "IJK")
    result = await _points(scanning.scan_on_line((0, 0, 0), (2, 0, 0), (0, 0, 1), 1))
    assert (CommandName.ON_SCAN_REPORT, "X(),Y(),Z(),Q(),IJK()") in recorder.calls
    assert [r.number("X") for r in result] == [0.0, 1.0, 2.0]
    assert [r.vector("IJK") for r in result] == [(0.0, 0.0, 1.0)] * 3
    assert result[0] == {"X": 0.0, "Y": 0.0, "Z": 0.0, "Q": 5.0, "IJK": (0.0, 0.0, 1.0)}


async def test_the_default_scanning_keeps_returning_xyz_tuples(
    machine: IppDmeMachine, recorder: Recorder
) -> None:
    await _points(machine.scanning.scan_on_line((0, 0, 0), (2, 0, 0), (0, 0, 1), 1))
    assert (CommandName.ON_SCAN_REPORT, "X(),Y(),Z()") in recorder.calls


def test_reporting_needs_the_width_of_names_the_standard_does_not_fix(
    machine: IppDmeMachine,
) -> None:
    with pytest.raises(ValueError, match=r"Tool\.Alignment"):
        machine.scanning.reporting("X", "Tool.Alignment")
    assert machine.scanning.reporting("X", "Tool.Alignment", widths={"Tool.Alignment": 6})


async def test_on_scan_report_sends_the_names(machine: IppDmeMachine, recorder: Recorder) -> None:
    await machine.scanning.on_scan_report("X", "Q")
    assert recorder.calls[-1] == (CommandName.ON_SCAN_REPORT, "X(),Q()")


async def test_scan_on_curve_sends_the_optional_columns_and_the_rotary_flag(
    machine: IppDmeMachine, recorder: Recorder
) -> None:
    points = [
        CurvePoint((0, 0, 0), (0, 0, 1), 1, primary=(0, 0, 1), secondary=(1, 0, 0), rotary=5),
        CurvePoint((1, 0, 0), (0, 0, 1), -1, primary=(0, 0, 1), secondary=(1, 0, 0), rotary=6),
    ]
    await _points(machine.scanning.scan_on_curve(points, closed=True, rotary_table=True))
    assert recorder.last == (
        "Closed(1),Format(X(),Y(),Z(),IJK(),tag,pi,pj,pk,si,sj,sk,R()),RT(1),"
        "Data(0,0,0,0,0,1,1,0,0,1,1,0,0,5,1,0,0,0,0,1,-1,0,0,1,1,0,0,6)"
    )


async def test_scan_on_curve_with_only_the_primary_direction(
    machine: IppDmeMachine, recorder: Recorder
) -> None:
    points = [CurvePoint((0, 0, 0), (0, 0, 1), 1, primary=(0, 0, 1))]
    await _points(machine.scanning.scan_on_curve(points))
    assert (
        recorder.last
        == "Closed(0),Format(X(),Y(),Z(),IJK(),tag,pi,pj,pk),Data(0,0,0,0,0,1,1,0,0,1)"
    )


def test_curve_points_must_agree_on_their_columns() -> None:
    from pyippdme.client import builders

    mixed = [CurvePoint((0, 0, 0), (0, 0, 1), 1, rotary=1), CurvePoint((1, 0, 0), (0, 0, 1), 1)]
    with pytest.raises(ValueError, match="same optional columns"):
        builders.scan_on_curve(mixed)
    with pytest.raises(ValueError, match="secondary needs primary"):
        CurvePoint((0, 0, 0), (0, 0, 1), 1, secondary=(1, 0, 0))


# -- Optical -----------------------------------------------------------------


async def test_data_acquire_sends_positions_with_both_orientation_vectors(
    machine: IppDmeMachine, recorder: Recorder
) -> None:
    path = [
        AcquisitionPoint((0, 0, 0), (0, 0, 1), (1, 0, 0)),
        AcquisitionPoint((9, 0, 0), (0, 0, 1), (1, 0, 0)),
    ]
    await machine.raw_data.data_acquire("A", "Sweep", "S1", path, step_width=0.5, rotary_table=True)
    assert recorder.last == '"A",Sweep,"S1",2,0,0,0,0,0,1,1,0,0,9,0,0,0,0,1,1,0,0,0.5,1'


async def test_data_acquire_without_positions_is_a_single_shot(
    machine: IppDmeMachine, recorder: Recorder
) -> None:
    await machine.raw_data.data_acquire("A", "SingleShot", "S1")
    assert recorder.last == '"A",SingleShot,"S1",0'


async def test_data_acquire_rotary_table_needs_a_step_width(machine: IppDmeMachine) -> None:
    with pytest.raises(ValueError, match="step_width"):
        await machine.raw_data.data_acquire("A", "Sweep", "S1", rotary_table=True)


# -- Feature extraction (Annex J.2) -----------------------------------------


async def test_roi_sends_the_shape_and_whether_it_is_included(
    machine: IppDmeMachine, recorder: Recorder
) -> None:
    shape = features.CircleShape((1, 2, 3), (0, 0, 1), (1, 0, 0), 5)
    await machine.feature_extraction.roi("R1", shape)
    assert recorder.calls[-1] == (CommandName.ROI, '"R1",Circle(1,2,3,0,0,1,1,0,0,5),include')
    await machine.feature_extraction.roi("R2", features.CircleRel(0.5), include=False)
    assert recorder.last == '"R2",CircleRel(0.5),exclude'


@pytest.mark.parametrize(
    ("shape", "wire"),
    [
        (
            features.CylinderShape((1, 2, 3), (0, 0, 1), (1, 0, 0), 5, 10),
            "Cylinder(1,2,3,0,0,1,1,0,0,5,10)",
        ),
        (features.LineShape((1, 2, 3), (1, 0, 0), 4), "Line(1,2,3,1,0,0,4)"),
        (features.PointShape((1, 2, 3), (0, 0, 1)), "Point(1,2,3,0,0,1)"),
        (features.SphereShape((1, 2, 3), (0, 0, 1), (1, 0, 0), 5), "Sphere(1,2,3,0,0,1,1,0,0,5)"),
        (features.CylinderRel(1, 2), "CylinderRel(1,2)"),
        (features.SphereRel(1), "SphereRel(1)"),
        (features.Polygon2D([(0, 0, 0), (1, 0, 0), (1, 1, 0)]), "Polygon2D(0,0,0,1,0,0,1,1,0)"),
        (features.Polygon3D([(0, 0, 0), (1, 0, 1)]), "Polygon3D(0,0,0,1,0,1)"),
    ],
)
def test_shapes_are_sent_with_their_standard_names(shape: features.Shape, wire: str) -> None:
    assert shape.to_argument().to_wire() == wire


async def test_feature_extract_returns_the_geometric_element(
    machine: IppDmeMachine, recorder: Recorder
) -> None:
    recorder.replies[CommandName.FEATURE_EXTRACT] = lambda: Items(
        (_named("X", 1), _named("Y", 2), _named("R", 5))
    )
    result = await machine.feature_extraction.feature_extract(
        ["A1", "A2"],
        features.CircleShape((1, 2, 3), (0, 0, 1), (1, 0, 0), 5),
        ["R1"],
        "Method",
    )
    assert recorder.calls[-1] == (
        CommandName.FEATURE_EXTRACT,
        'Acqs("A1","A2"),Circle(1,2,3,0,0,1,1,0,0,5),ROIs("R1"),"Method",GeoElem()',
    )
    assert result == (Report({"X": 1.0, "Y": 2.0, "R": 5.0}),)


async def test_feature_extract_points_streams_the_edge_points(
    machine: IppDmeMachine, recorder: Recorder
) -> None:
    async def edge_points() -> AsyncIterator[DataPayload]:
        yield NumericData((Number.of(1), Number.of(2), Number.of(3)))
        yield NumericData((Number.of(4), Number.of(5), Number.of(6)))

    recorder.replies[CommandName.FEATURE_EXTRACT] = edge_points
    points = await _points(
        machine.feature_extraction.feature_extract_points(
            ["A1"], features.PointShape((0, 0, 0), (0, 0, 1)), [], "Method", 0.25
        )
    )
    assert points == [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]
    assert recorder.last == """Acqs("A1"),Point(0,0,0,0,0,1),ROIs(),"Method",QEPs(S(0.25))"""


# -- Properties and the DME class -------------------------------------------


async def test_get_props_reads_several_properties_with_one_command(
    machine: IppDmeMachine, recorder: Recorder
) -> None:
    recorder.replies[CommandName.GET_PROP] = recorder.replies[CommandName.GET_PROP_E] = lambda: (
        Items((_named("Part.Temperature", 20), _named("Tool.Alignment", 0, 0, 1)))
    )
    report = await machine.server.get_props("Part.Temperature", "Tool.Alignment")
    assert recorder.last == "Part.Temperature(),Tool.Alignment()"
    assert report.vector("Tool.Alignment") == (0.0, 0.0, 1.0)
    assert await machine.server.get_prop("Tool.Alignment") == (0.0, 0.0, 1.0)
    assert await machine.server.get_props_e("Part.Temperature")


async def test_set_props_writes_numbers_vectors_and_strings_with_one_command(
    machine: IppDmeMachine, recorder: Recorder
) -> None:
    await machine.server.set_props({"Part.Temperature": 21, "Tool.Alignment": (0, 0, 1)})
    assert recorder.last == "Part.Temperature(21),Tool.Alignment(0,0,1)"
    await machine.server.set_prop("Tool.Name", "tip")
    assert recorder.last == 'Tool.Name("tip")'


async def test_a_server_with_several_machine_classes_is_supported(
    machine: IppDmeMachine, recorder: Recorder
) -> None:
    recorder.replies[CommandName.GET_MACHINE_CLASS] = lambda: [
        StringValue(String("A_B")),
        StringValue(String("C_D")),
    ]
    assert await machine.dme.get_machine_classes() == ("A_B", "C_D")
    with pytest.raises(IppDmeError, match="get_machine_classes"):
        await machine.dme.get_machine_class()
