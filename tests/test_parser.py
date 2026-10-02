# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Grammar tests using real wire-format examples.

Command examples are drawn from the VDMA 8722:2024-04 spec text and
real-world captured client/server session logs.
"""

from __future__ import annotations

import pytest

from pyippdme.exceptions import IppDmeProtocolError
from pyippdme.protocol.ast import (
    AckResponse,
    BasicName,
    DataResponse,
    DoneResponse,
    ErrorResponse,
    EventTag,
    Items,
    NamedValue,
    Number,
    NumericData,
    PropertyData,
    StringValue,
    Tag,
)
from pyippdme.protocol.codec import decode_command_line, decode_response_line

# -- commands ----------------------------------------------------------------

COMMAND_LINES = [
    "00001 StartSession()\r\n",
    "09999 EndSession()\r\n",
    "00010 GetDMEVersion()\r\n",
    "00030 SetProp(Tool.GoToPar.Speed(5))\r\n",
    "00070 OnPtMeasReport(X(), Y(), Z(), IJK())\r\n",
    "00080 PtMeas(X(1.33), Y(-2.6E01), Z(30.998), IJK(2,2,2))\r\n",
    "00190 PtMeas(X(-2.6E01))\r\n",
    "00010 GoTo(X(1.003E01), Y(-0.129), Z(+1.245E01))\r\n",
    "00090 GoTo(Tool.A(45.0))\r\n",
    "00120 GoTo(Tool.A(45), Z(-4.003E-01), Y(-0.129), X(+1.245E01), Tool.B(4.0) )\r\n",
    "00010 GetProp(Tool.Name())\r\n",
    "00020 GetProp(FoundTool.Name(), FoundTool.Id())\r\n",
    "00040 GetProp(Tool.PtMeasPar.Accel(), Tool.PtMeasPar.Accel.Max())\r\n",
    "00010 GoTo(X(10000000))\r\n",
    "E0553 OnMoveReportE(Time(1),Dis(20),X(), Y(), Z())\r\n",
    "00049 StopDaemon(E0553)\r\n",
    "00001 AlignPart(1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 2.0)\r\n",
    "00001 ScanOnLine(0,1,2,14,15,2,0,0,1,0.5)\r\n",
    'E0001 Test("Ref Tool",E0001,XT.R.A(),Roi(-5,.5,+5E7,5e-03))\r\n',
    'E0001 TestE(<x><y a="b">z</y></x>)\r\n',
    "00001 SetProp(Tool.GoToPar.Speed(0))\r\n",
    "00001 SetProp(Tool.GoToPar.Speed(+90))\r\n",
    "00001 SetProp(Tool.GoToPar.Speed(-90))\r\n",
    "00001 SetProp(Tool.GoToPar.Speed(1234567890123456))\r\n",
    "00001 SetProp(Tool.GoToPar.Speed(-0000))\r\n",
    "00001 SetProp(Tool.GoToPar.Speed(+0000.5E7))\r\n",
    "00001 SetProp(Tool.GoToPar.Speed(3.9E2))\r\n",
]


@pytest.mark.parametrize("line", COMMAND_LINES)
def test_command_parses_and_round_trips(line: str) -> None:
    command = decode_command_line(line)
    # Our own encoder always emits canonical (single-space) formatting, so a
    # round trip only reproduces the original text when the input already
    # used canonical spacing; otherwise it must at least re-parse identically.
    reencoded = decode_command_line(command.to_wire() + "\r\n")
    assert reencoded == command


def test_command_tag_and_event_tag() -> None:
    cmd = decode_command_line("00001 StartSession()\r\n")
    assert cmd.tag == Tag("00001")
    evt = decode_command_line("E0553 OnMoveReportE()\r\n")
    assert evt.tag == EventTag("E0553")


def test_command_method_name_and_args() -> None:
    cmd = decode_command_line("00080 PtMeas(X(1.33), Y(-2.6E01))\r\n")
    assert cmd.method.name == "PtMeas"
    assert len(cmd.method.args) == 2
    x_arg = cmd.method.args[0]
    assert isinstance(x_arg, NamedValue)
    assert x_arg.name == "X"
    x_value = x_arg.args[0]
    assert isinstance(x_value, Number)
    assert x_value.value == pytest.approx(1.33)


def test_xml_payload_is_captured_raw() -> None:
    cmd = decode_command_line('E0001 TestE(<x><y a="b">z</y></x>)\r\n')
    assert cmd.method.xml is not None
    assert cmd.method.xml.raw == '<x><y a="b">z</y></x>'


def test_bare_name_and_event_tag_arguments_disambiguated() -> None:
    cmd = decode_command_line('00001 Test("Ref Tool",E0001,XT.R.A(),Roi(-5,.5,+5E7,5e-03))\r\n')
    args = cmd.method.args
    assert args[1] == EventTag("E0001")
    named = args[2]
    assert isinstance(named, NamedValue)
    assert named.name == "XT.R.A"
    assert named.args == ()


def test_bare_basic_name_argument() -> None:
    cmd = decode_command_line("00200 ChangeTool(Switch)\r\n")
    assert cmd.method.args == (BasicName("Switch"),)


# -- responses -----------------------------------------------------------------

RESPONSE_LINES = [
    "00001 &\r\n",
    "00001 %\r\n",
    "E0553 &\r\n",
    "E0553 # X(50), Y(433), Z(500)\r\n",
    '00010 # DMEVersion("1.4")\r\n',
    "00080 # IJK(0.00000,0.00000,1.00000), X(3.06205), Y(-24.26795), Z(32.73005)\r\n",
    "00190 # ER(3.00000), X(-23.00000), Tool.A(0.00000)\r\n",
    "00010 # 0.6, 0., 0.8\r\n",
    "00010 # 0.6,0.,0.8\r\n",
    "00200 #ChangeToolAction(Switch, X(0.00000), Y(0.00000), Z(-8.00000))\r\n",
    '00010 # "GoToPar.Speed.Def", "Number"\r\n',
    '00010 # "RefTool"\r\n',
    (
        '00070 ! Error(2,0006,"00070 GoTo(X(20.0),Y(40.0),Z(60.0))",'
        '"Transaction aborted (Use ClearAllErrors to Continue)")\r\n'
    ),
    (
        '00010 ! Error(3, 1008, "00010 GoTo(X(10000000)) : GoTo: TARGET POINT OUT OF WORK VOLUME", '
        '"Target position out of machine volume")\r\n'
    ),
]


@pytest.mark.parametrize("line", RESPONSE_LINES)
def test_response_parses_and_round_trips(line: str) -> None:
    response = decode_response_line(line)
    reencoded = decode_response_line(response.to_wire() + "\r\n")
    assert reencoded == response


def test_ack_and_done() -> None:
    assert decode_response_line("00001 &\r\n") == AckResponse(Tag("00001"))
    assert decode_response_line("00001 %\r\n") == DoneResponse(Tag("00001"))


def test_data_response_always_encodes_the_space_after_hash() -> None:
    # Cross-validated against the independent NIST I++ DME reference parser
    # (parserRes), which rejects a data response missing this space with
    # "SPACE MISSING AT EIGHTH CHARACTER OF RESPONSE" - see the docstring of
    # pyippdme.protocol.ast.DataResponse for why the formal VDMA 8722 grammar
    # doesn't show it but real implementations require it.
    response = DataResponse(Tag("00010"), NumericData((Number("1"),)))
    assert response.to_wire() == "00010 # 1"


def test_numeric_data_response() -> None:
    response = decode_response_line("00010 # 0.6, 0., 0.8\r\n")
    assert isinstance(response, DataResponse)
    assert isinstance(response.data, NumericData)
    assert [n.value for n in response.data.values] == pytest.approx([0.6, 0.0, 0.8])


def test_property_data_response_values() -> None:
    response = decode_response_line('00010 # "GoToPar.Speed.Def", "Number"\r\n')
    assert isinstance(response, DataResponse)
    payload = response.data
    assert isinstance(payload, PropertyData)
    assert payload.first.value == "GoToPar.Speed.Def"
    assert payload.second.value == "Number"


def test_bare_string_response() -> None:
    response = decode_response_line('00010 # "RefTool"\r\n')
    assert isinstance(response, DataResponse)
    assert isinstance(response.data, StringValue)
    assert response.data.value.value == "RefTool"


def test_items_response_with_nested_method_shaped_data() -> None:
    response = decode_response_line(
        "00200 #ChangeToolAction(Switch, X(0.00000), Y(0.00000), Z(-8.00000))\r\n"
    )
    assert isinstance(response, DataResponse)
    assert isinstance(response.data, Items)
    (change_tool_action,) = response.data.values
    assert change_tool_action.name == "ChangeToolAction"
    assert change_tool_action.args[0] == BasicName("Switch")


def test_error_response_fields() -> None:
    response = decode_response_line(
        '00010 ! Error(3, 1008, "cause text", "Target position out of machine volume")\r\n'
    )
    assert isinstance(response, ErrorResponse)
    assert response.severity == 3
    assert response.number == "1008"
    assert response.cause == "cause text"
    assert response.text == "Target position out of machine volume"


# -- malformed input -----------------------------------------------------------


@pytest.mark.parametrize(
    "line",
    [
        "00001 StartSession()",  # missing terminator
        "0001 StartSession()\r\n",  # tag too short
        "00000 StartSession()\r\n",  # reserved tag
        "00001 12BadName()\r\n",  # method name can't start with a digit
        "00001 StartSession(\r\n",  # unterminated argument list
    ],
)
def test_malformed_command_raises(line: str) -> None:
    with pytest.raises(IppDmeProtocolError):
        decode_command_line(line)


def test_missing_mandatory_space_is_accepted_leniently() -> None:
    # Whitespace between tokens is ignored uniformly during parsing (see
    # pyippdme.protocol.parser module docstring), which is more lenient than
    # the standard's exactly-one-mandatory-space rule at position 6. This
    # only affects what we accept on input; our own encoder always emits
    # the correct spacing.
    command = decode_command_line("00001StartSession()\r\n")
    assert command.method.name == "StartSession"
