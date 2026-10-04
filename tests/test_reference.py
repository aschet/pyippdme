# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""The command reference and the meta commands of the shell."""

from __future__ import annotations

from pyippdme.cli.client import CommandCompleter
from pyippdme.client import reference_text
from pyippdme.client.reference import CommandReference, MetaCommands


def test_every_command_has_a_description_and_every_argument_a_type() -> None:
    reference = CommandReference()
    for name in reference.names():
        doc = reference.find(name)
        assert doc is not None
        assert doc.summary != "No description yet.", name
        assert doc.group in reference.groups()
        assert all(a.datatype for a in doc.arguments)
    assert sorted(n for names in reference.groups().values() for n in names) == reference.names()


def test_the_texts_name_no_unknown_commands() -> None:
    names = set(CommandReference().names())
    for table in (
        reference_text.SUMMARIES,
        reference_text.NOTES,
        reference_text.RETURNS,
        reference_text.EXAMPLES,
    ):
        assert set(table) <= names, sorted(set(table) - names)
    assert {c for c, _ in reference_text.ARGUMENT_OVERRIDES} <= names


def test_arguments_say_what_is_optional_and_a_bare_value() -> None:
    reference = CommandReference()
    goto = reference.find("goto")  # case does not matter
    assert goto is not None
    assert goto.signature == "GoTo(Positions, Sync*)"
    assert [(a.name, a.datatype, a.mandatory) for a in goto.arguments] == [
        ("Positions", "enum", True),
        ("Sync", "int", False),
    ]
    scan = reference.find("ScanOnLine")
    assert scan is not None
    assert all(a.positional for a in scan.arguments)
    assert scan.returns
    text = reference.manual("GoTo")
    assert "Sync*" in text
    assert "* optional" in text
    assert "Returns:" in text
    assert "Example: GoTo(" in text


def test_a_command_of_your_own_can_be_described() -> None:
    from pyippdme.server.catalog import CommandInfo

    reference = CommandReference({"MyCmd": CommandInfo(None, None)})  # type: ignore[arg-type]
    reference.describe("MyCmd", "Do my thing.", returns="A number.", example="MyCmd()")
    doc = reference.find("mycmd")
    assert doc is not None
    assert (doc.summary, doc.returns, doc.example) == ("Do my thing.", "A number.", "MyCmd()")


def test_meta_commands_help_cmds_man_quit() -> None:
    meta = MetaCommands()
    assert meta.is_meta(" .help")
    assert not meta.is_meta("GoTo(X(1))")
    assert ".man" in meta.run(".help").text
    assert "Move:" in meta.run(".cmds").text
    only = meta.run(".cmds scan").text
    assert "Scan:" in only
    assert "Move:" not in only
    assert "GoTo -" in meta.run(".man GoTo").text
    assert "Did you mean" in meta.run(".man GoT").text
    assert meta.run(".man").text.startswith("Usage")
    assert meta.run(".nope").text.startswith("Unknown meta command")
    assert meta.run(".quit").quit
    assert meta.run(".exit").quit
    assert not meta.run(".help").quit


def test_tab_completes_meta_commands_and_the_name_after_man() -> None:
    names = ["GoTo", "GoToPar", "Home"]
    first = CommandCompleter(names, lambda: "")
    out: list[str] = []
    while (m := first(".m", len(out))) is not None:
        out.append(m)
    assert out == [".man"]
    second = CommandCompleter(names, lambda: ".man ")
    out = []
    while (m := second("goto", len(out))) is not None:
        out.append(m)
    assert out == ["GoTo", "GoToPar"]  # a name for .man, without a bracket
