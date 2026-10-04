# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Tab completion of command names in the simple shell."""

from __future__ import annotations

import os
import sys

import pytest

from pyippdme.cli.client import CommandCompleter

NAMES = ["PtMeas", "PtMeasPar", "GoTo", "GetProp", "Get"]


def _all(completer: CommandCompleter, text: str) -> list[str]:
    found: list[str] = []
    while (match := completer(text, len(found))) is not None:
        found.append(match)
    return found


def test_command_names_complete_ignoring_case_and_open_the_bracket() -> None:
    completer = CommandCompleter(NAMES)
    assert _all(completer, "ptm") == ["PtMeas(", "PtMeasPar("]
    assert _all(completer, "G") == ["Get(", "GetProp(", "GoTo("]
    assert _all(completer, "x") == []


def test_only_the_first_word_is_completed() -> None:
    inside = CommandCompleter(NAMES, lambda: "GoTo(")
    assert _all(inside, "Pt") == []
    start = CommandCompleter(NAMES, lambda: "  ")
    assert _all(start, "Pt") == ["PtMeas(", "PtMeasPar("]


@pytest.mark.skipif(sys.platform == "win32", reason="needs a terminal and readline")
def test_tab_cycles_through_the_matches_in_place() -> None:
    pty = pytest.importorskip("pty")
    code = (
        "import readline\n"
        "if 'libedit' in (readline.__doc__ or ''): raise SystemExit(77)\n"
        "from pyippdme.cli.client import install_completion\n"
        "install_completion(['PtMeas','PtMeasPar'])\n"
        "print('LINE:' + input('> '))\n"
    )
    pid, fd = pty.fork()
    if pid == 0:  # pragma: no cover - the child
        os.execv(sys.executable, [sys.executable, "-c", code])  # noqa: S606
    try:
        os.read(fd, 100)  # the prompt
        os.write(fd, b"Pt\t")  # first match
        os.write(fd, b"\t")  # next match, replacing the first
        os.write(fd, b"\r")
        output = b""
        while True:
            try:
                chunk = os.read(fd, 1000)
            except OSError:
                break
            if not chunk:
                break
            output += chunk
    finally:
        _, status = os.waitpid(pid, 0)
    if os.waitstatus_to_exitcode(status) == 77:
        pytest.skip("libedit has no menu completion")
    text = output.decode(errors="replace")
    assert "LINE:PtMeasPar(" in text  # the second Tab replaced the first match
    assert "PtMeas( " not in text  # and no list of matches was printed
