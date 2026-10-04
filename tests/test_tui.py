# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for the Textual-based TUI (pyippdme.cli.tui), driven headlessly."""

from __future__ import annotations

from textual.widgets import Button, Input, ListView, RichLog, Static
from textual_autocomplete import AutoComplete, DropdownItem

from pyippdme.cli.completion import completion_context
from pyippdme.cli.tui import (
    _HELP_PLACEHOLDER,
    _HELP_USAGE,
    IppDmeTui,
    _format_signature,
)


def _dropdown_values(autocomplete: AutoComplete) -> list[str]:
    option_list = autocomplete.option_list
    values = []
    for i in range(option_list.option_count):
        option = option_list.get_option_at_index(i)
        assert isinstance(option, DropdownItem)
        values.append(option.value)
    return values


async def test_tui_connects_and_round_trips_a_command(tcp_server_port: int) -> None:
    app = IppDmeTui("127.0.0.1", tcp_server_port)
    async with app.run_test() as pilot:
        await pilot.pause()
        log = app.query_one("#log", RichLog)
        assert any("Connected" in str(line) for line in log.lines)

        command_input = app.query_one("#command_input", Input)
        command_input.value = "StartSession()"
        await pilot.press("enter")
        await pilot.pause()

        command_input.value = "GetDMEVersion()"
        await pilot.press("enter")
        await pilot.pause()

        text = "\n".join(str(line) for line in log.lines)
        assert "GetDMEVersion" in text
        assert "2.5" in text


async def test_tui_shows_server_error(tcp_server_port: int) -> None:
    app = IppDmeTui("127.0.0.1", tcp_server_port)
    async with app.run_test() as pilot:
        await pilot.pause()
        command_input = app.query_one("#command_input", Input)
        # No StartSession() first: every command is rejected with 0008.
        command_input.value = "GetDMEVersion()"
        await pilot.press("enter")
        await pilot.pause()

        text = "\n".join(str(line) for line in app.query_one("#log", RichLog).lines)
        assert "0008" in text


async def test_tui_sidebar_selection_fills_command_input(tcp_server_port: int) -> None:
    app = IppDmeTui("127.0.0.1", tcp_server_port)
    async with app.run_test() as pilot:
        await pilot.pause()
        list_view = app.query_one("#sidebar", ListView)
        list_view.focus()
        list_view.index = 0
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        command_input = app.query_one("#command_input", Input)
        assert command_input.value.endswith("()")


async def test_tui_virtual_starts_and_connects_to_an_embedded_server() -> None:
    app = IppDmeTui(virtual=True)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app._embedded_server is not None
        assert app._embedded_server.port is not None
        assert app.client is not None

        log_text = "\n".join(str(line) for line in app.query_one("#log", RichLog).lines)
        assert "Started an in-process VirtualCMM" in log_text
        assert "Connected" in log_text

        command_input = app.query_one("#command_input", Input)
        command_input.value = "StartSession()"
        await pilot.press("enter")
        await pilot.pause()
        command_input.value = "GetMachineClass()"
        await pilot.press("enter")
        await pilot.pause()

        text = "\n".join(str(line) for line in app.query_one("#log", RichLog).lines)
        assert "_VirtualCMM" in text

    # on_unmount must have torn the embedded server down too, not just the client.
    assert app._embedded_server is None


async def test_tui_connection_button_toggles_label_and_variant(tcp_server_port: int) -> None:
    app = IppDmeTui(virtual=True)
    async with app.run_test() as pilot:
        await pilot.pause()
        button = app.query_one("#connection_button", Button)
        assert str(button.label) == "Disconnect"
        assert button.variant == "error"

        await pilot.click("#connection_button")
        await pilot.pause()

        assert str(button.label) == "Connect"
        assert button.variant == "success"
        assert app.client is None
    assert app.client is None


async def test_tui_command_input_shows_a_dropdown_with_multiple_matches(
    tcp_server_port: int,
) -> None:
    """The original bug this replaced SuggestFromList over: two commands share a prefix."""
    app = IppDmeTui("127.0.0.1", tcp_server_port)
    async with app.run_test() as pilot:
        await pilot.pause()
        command_input = app.query_one("#command_input", Input)
        autocomplete = app.query_one(AutoComplete)
        command_input.focus()

        for char in "Enable":
            await pilot.press(char)
        await pilot.pause()

        assert autocomplete.display
        values = _dropdown_values(autocomplete)
        assert "EnableUser" in values
        assert "EnableRotaryTableVarCsy" in values


async def test_tui_command_input_tab_accepts_the_highlighted_dropdown_item(
    tcp_server_port: int,
) -> None:
    app = IppDmeTui("127.0.0.1", tcp_server_port)
    async with app.run_test() as pilot:
        await pilot.pause()
        command_input = app.query_one("#command_input", Input)
        command_input.focus()

        await pilot.press("G", "o", "T")
        await pilot.pause()

        await pilot.press("tab")
        await pilot.pause()

        # Auto-opens the parens too (GoTo takes arguments) - see
        # apply_completion's docstring.
        assert command_input.value == "GoTo("
        assert app.focused is command_input  # Tab did not also switch focus


async def test_tui_command_input_tab_completes_an_argument_name_in_place(
    tcp_server_port: int,
) -> None:
    """Regression test: accepting an argument completion must not wipe out the command name."""
    app = IppDmeTui("127.0.0.1", tcp_server_port)
    async with app.run_test() as pilot:
        await pilot.pause()
        command_input = app.query_one("#command_input", Input)
        command_input.focus()

        for char in "GoT":
            await pilot.press(char)
        await pilot.pause()
        await pilot.press("tab")
        await pilot.pause()
        # Auto-opens the parens too (GoTo takes arguments) - see
        # apply_completion's docstring.
        assert command_input.value == "GoTo("

        for char in "Sy":
            await pilot.press(char)
        await pilot.pause()
        await pilot.press("tab")
        await pilot.pause()

        assert command_input.value == "GoTo(Sync"
        assert command_input.cursor_position == len("GoTo(Sync")


async def test_tui_command_input_tab_closes_a_zero_argument_command(
    tcp_server_port: int,
) -> None:
    app = IppDmeTui("127.0.0.1", tcp_server_port)
    async with app.run_test() as pilot:
        await pilot.pause()
        command_input = app.query_one("#command_input", Input)
        command_input.focus()

        for char in "StartSess":
            await pilot.press(char)
        await pilot.pause()
        await pilot.press("tab")
        await pilot.pause()

        assert command_input.value == "StartSession()"
        assert command_input.cursor_position == len("StartSession(")


async def test_tui_command_input_enter_does_not_submit_an_incomplete_command_name(
    tcp_server_port: int,
) -> None:
    """Accepting "GoTo" opens its parens for more typing - Enter must not also submit that."""
    app = IppDmeTui("127.0.0.1", tcp_server_port)
    async with app.run_test() as pilot:
        await pilot.pause()
        command_input = app.query_one("#command_input", Input)
        command_input.focus()

        for char in "GoTo":
            await pilot.press(char)
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        assert command_input.value == "GoTo("
        autocomplete = app.query_one(AutoComplete)
        assert autocomplete.display
        assert set(_dropdown_values(autocomplete)) >= {"Sync", "X", "Y", "Z", "R"}


async def test_tui_command_input_enter_still_submits_a_zero_argument_command(
    tcp_server_port: int,
) -> None:
    app = IppDmeTui("127.0.0.1", tcp_server_port)
    async with app.run_test() as pilot:
        await pilot.pause()
        command_input = app.query_one("#command_input", Input)
        command_input.focus()

        for char in "StartSession":
            await pilot.press(char)
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        assert command_input.value == ""
        log_text = "\n".join(line.text for line in app.query_one("#log", RichLog).lines)
        assert "&" in log_text.splitlines()


async def test_tui_command_input_enter_still_submits_a_help_meta_command(
    tcp_server_port: int,
) -> None:
    """Regression test: completing ".help GoTo"'s argument must not swallow Enter's submit.

    "GoTo" reuses the exact same bare-command-name completion path a real
    call would, but must not get the "open its parens" treatment here -
    ".help" only ever wants the plain name.
    """
    app = IppDmeTui("127.0.0.1", tcp_server_port)
    async with app.run_test() as pilot:
        await pilot.pause()
        command_input = app.query_one("#command_input", Input)
        panel = app.query_one("#signature_text", Static)
        command_input.focus()

        for char in ".help GoTo":
            await pilot.press(char)
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        assert command_input.value == ""
        assert r"Positions \[enum]" in str(panel.content)


async def test_tui_command_input_tab_works_past_the_first_argument(
    tcp_server_port: int,
) -> None:
    """Regression test: Tab must keep working once the partial word already fully matches.

    "GoTo(X(10), Y" has exactly one remaining candidate (Y) which already
    equals what's typed - the dropdown must not hide itself in that case,
    or Tab silently does nothing and falls through to its default
    focus-switch action instead of accepting the completion.
    """
    app = IppDmeTui("127.0.0.1", tcp_server_port)
    async with app.run_test() as pilot:
        await pilot.pause()
        command_input = app.query_one("#command_input", Input)
        autocomplete = app.query_one(AutoComplete)
        command_input.focus()

        for char in "GoTo(X(10), Y":
            await pilot.press(char)
        await pilot.pause()
        assert autocomplete.display

        await pilot.press("tab")
        await pilot.pause()

        assert command_input.value == "GoTo(X(10), Y"
        assert app.focused is command_input  # Tab did not fall through to focus-switch


async def test_tui_command_input_tab_switches_focus_with_nothing_to_complete(
    tcp_server_port: int,
) -> None:
    app = IppDmeTui("127.0.0.1", tcp_server_port)
    async with app.run_test() as pilot:
        await pilot.pause()
        command_input = app.query_one("#command_input", Input)
        autocomplete = app.query_one(AutoComplete)
        command_input.focus()
        assert command_input.value == ""
        assert not autocomplete.display

        await pilot.press("tab")
        await pilot.pause()

        assert app.focused is not command_input


async def test_tui_command_input_suggests_argument_names_inside_a_command(
    tcp_server_port: int,
) -> None:
    app = IppDmeTui("127.0.0.1", tcp_server_port)
    async with app.run_test() as pilot:
        await pilot.pause()
        command_input = app.query_one("#command_input", Input)
        autocomplete = app.query_one(AutoComplete)
        command_input.focus()

        for char in "GoTo(Sy":
            await pilot.press(char)
        await pilot.pause()

        assert autocomplete.display
        assert _dropdown_values(autocomplete) == ["Sync"]


async def test_tui_command_input_suggests_axis_names_for_a_purely_enum_command(
    tcp_server_port: int,
) -> None:
    """Get(...)'s only argument (Axes) is DataType.ENUM - the axis-name fallback applies."""
    app = IppDmeTui("127.0.0.1", tcp_server_port)
    async with app.run_test() as pilot:
        await pilot.pause()
        command_input = app.query_one("#command_input", Input)
        autocomplete = app.query_one(AutoComplete)
        command_input.focus()

        for char in "Get(":
            await pilot.press(char)
        await pilot.pause()

        assert autocomplete.display
        assert set(_dropdown_values(autocomplete)) == {"X", "Y", "Z", "R"}


async def test_tui_command_input_combines_named_and_enum_fallback_arguments(
    tcp_server_port: int,
) -> None:
    """GoTo(...) has both a real named argument (Sync) and an ENUM one (Positions)."""
    app = IppDmeTui("127.0.0.1", tcp_server_port)
    async with app.run_test() as pilot:
        await pilot.pause()
        command_input = app.query_one("#command_input", Input)
        autocomplete = app.query_one(AutoComplete)
        command_input.focus()

        for char in "GoTo(":
            await pilot.press(char)
        await pilot.pause()

        assert autocomplete.display
        assert set(_dropdown_values(autocomplete)) == {"Sync", "X", "Y", "Z", "R"}


async def test_tui_command_input_suggests_nothing_inside_a_positional_command(
    tcp_server_port: int,
) -> None:
    """ScanOnLine(1, 2, 3, ...) takes bare values in order, not Name(value) pairs.

    Suggesting its parameter names (Sx, Sy, ...) would be actively wrong.
    """
    app = IppDmeTui("127.0.0.1", tcp_server_port)
    async with app.run_test() as pilot:
        await pilot.pause()
        command_input = app.query_one("#command_input", Input)
        autocomplete = app.query_one(AutoComplete)
        command_input.focus()

        for char in "ScanOnLine(":
            await pilot.press(char)
        await pilot.pause()

        assert not autocomplete.display
        assert autocomplete.option_list.option_count == 0


async def test_tui_typing_a_positional_command_shows_a_live_argument_hint(
    tcp_server_port: int,
) -> None:
    """The panel must point at which bare value comes next as the user types.

    With no dropdown to lean on (see the test above), this is the only
    in-context hint for a positional command like
    CenterPart(Px, Py, Pz, Limit).
    """
    app = IppDmeTui("127.0.0.1", tcp_server_port)
    async with app.run_test() as pilot:
        await pilot.pause()
        command_input = app.query_one("#command_input", Input)
        panel = app.query_one("#signature_text", Static)
        command_input.focus()

        for char in "CenterPart(":
            await pilot.press(char)
        await pilot.pause()
        assert r"[bold]- Px \[float][/bold]" in str(panel.content)

        for char in "10, 20, ":
            await pilot.press(char)
        await pilot.pause()
        text = str(panel.content)
        assert r"[bold]- Pz \[float][/bold]" in text
        assert r"[bold]- Px" not in text


async def test_tui_signature_panel_autoscrolls_to_keep_the_highlighted_argument_visible(
    tcp_server_port: int,
) -> None:
    """ScanOnHelix has 14 arguments - more than the panel's fixed height shows at once."""
    app = IppDmeTui("127.0.0.1", tcp_server_port)
    async with app.run_test() as pilot:
        await pilot.pause()
        command_input = app.query_one("#command_input", Input)
        panel = app.query_one("#signature_panel")
        command_input.focus()

        for char in "ScanOnHelix(" + "1," * 12:
            await pilot.press(char)
        await pilot.pause()

        viewport_height = panel.size.height
        # The 13th argument (index 12, "pitch") is well past a panel that
        # only shows a handful of lines at once - it must have scrolled
        # into view, not stayed clipped below the fold.
        assert panel.scroll_y > 0
        assert panel.scroll_y <= 12 <= panel.scroll_y + viewport_height - 1


async def test_tui_typing_a_named_command_shows_its_signature_without_a_marker(
    tcp_server_port: int,
) -> None:
    """Order doesn't matter for a named (Name(value)) command, so no argument is singled out."""
    app = IppDmeTui("127.0.0.1", tcp_server_port)
    async with app.run_test() as pilot:
        await pilot.pause()
        command_input = app.query_one("#command_input", Input)
        panel = app.query_one("#signature_text", Static)
        command_input.focus()

        for char in "GoTo(":
            await pilot.press(char)
        await pilot.pause()

        text = str(panel.content)
        assert "GoTo" not in text  # no command-name header - just the parameters
        assert r"Positions \[enum]" in text
        assert "[bold]" not in text


async def test_tui_typing_a_command_name_does_not_touch_the_signature_panel(
    tcp_server_port: int,
) -> None:
    """No hint to show yet before the cursor is inside a command's parens."""
    app = IppDmeTui("127.0.0.1", tcp_server_port)
    async with app.run_test() as pilot:
        await pilot.pause()
        command_input = app.query_one("#command_input", Input)
        panel = app.query_one("#signature_text", Static)
        command_input.focus()

        for char in "GoTo":
            await pilot.press(char)
        await pilot.pause()

        assert str(panel.content) == _HELP_PLACEHOLDER


async def test_tui_help_for_a_positional_command_lists_its_parameters(
    tcp_server_port: int,
) -> None:
    """A positional command's help is just its parameter list too - no command name header."""
    app = IppDmeTui("127.0.0.1", tcp_server_port)
    async with app.run_test() as pilot:
        await pilot.pause()
        command_input = app.query_one("#command_input", Input)
        panel = app.query_one("#signature_text", Static)
        command_input.focus()

        command_input.value = ".help ScanOnLine"
        await pilot.press("enter")
        await pilot.pause()

        text = str(panel.content)
        assert "ScanOnLine" not in text
        assert r"Sx \[float]" in text
        assert r"RT* \[bool]" in text


def test_completion_context_parses_bare_command_name() -> None:
    assert completion_context("GoT") == (None, "GoT")


def test_completion_context_parses_inside_a_commands_arguments() -> None:
    assert completion_context("GoTo(Sy") == ("GoTo", "Sy")


def test_completion_context_after_closing_the_command_reverts_to_bare() -> None:
    assert completion_context("GoTo(X(10)) ") == (None, "")


def test_completion_context_nested_groups_reuse_the_outer_command() -> None:
    assert completion_context("ScanOnCurve(Format(") == ("ScanOnCurve", "")


def test_format_signature_renders_a_dash_bullet_list_without_indentation() -> None:
    assert _format_signature("GoTo") == "- Positions \\[enum]\n- Sync* \\[int]"


def test_format_signature_marks_the_highlighted_argument_in_bold() -> None:
    """Bold, not a distinct leading marker - see the docstring on why."""
    text = _format_signature("GoTo", highlight_index=1)
    assert text == "- Positions \\[enum]\n[bold]- Sync* \\[int][/bold]"


def test_format_signature_is_empty_for_a_zero_argument_command() -> None:
    assert _format_signature("StartSession") == ""


async def test_tui_command_input_history_cycles_with_up_down(tcp_server_port: int) -> None:
    app = IppDmeTui("127.0.0.1", tcp_server_port)
    async with app.run_test() as pilot:
        await pilot.pause()
        command_input = app.query_one("#command_input", Input)
        command_input.focus()

        for command in ("StartSession()", "GetDMEVersion()", "GetMachineClass()"):
            command_input.value = command
            await pilot.press("enter")
            await pilot.pause()

        # Something typed but not yet submitted, preserved across the round trip.
        command_input.value = "NotSubmittedYet"

        await pilot.press("up")
        await pilot.pause()
        assert command_input.value == "GetMachineClass()"

        await pilot.press("up")
        await pilot.pause()
        assert command_input.value == "GetDMEVersion()"

        await pilot.press("up")
        await pilot.pause()
        assert command_input.value == "StartSession()"

        await pilot.press("up")  # already at the oldest entry: stays put
        await pilot.pause()
        assert command_input.value == "StartSession()"

        await pilot.press("down")
        await pilot.pause()
        assert command_input.value == "GetDMEVersion()"

        await pilot.press("down")
        await pilot.pause()
        assert command_input.value == "GetMachineClass()"

        await pilot.press("down")  # past the newest entry: restores what was typed
        await pilot.pause()
        assert command_input.value == "NotSubmittedYet"


async def test_tui_command_input_history_skips_consecutive_duplicates(
    tcp_server_port: int,
) -> None:
    app = IppDmeTui("127.0.0.1", tcp_server_port)
    async with app.run_test() as pilot:
        await pilot.pause()
        command_input = app.query_one("#command_input", Input)
        command_input.focus()

        for _ in range(3):
            command_input.value = "StartSession()"
            await pilot.press("enter")
            await pilot.pause()

        assert app._history == ["StartSession()"]


async def test_tui_sidebar_has_a_caption(tcp_server_port: int) -> None:
    app = IppDmeTui("127.0.0.1", tcp_server_port)
    async with app.run_test() as pilot:
        await pilot.pause()
        caption = app.query_one("#sidebar_caption", Static)
        assert str(caption.content) == "Commands"


async def test_tui_signature_panel_has_a_caption(tcp_server_port: int) -> None:
    app = IppDmeTui("127.0.0.1", tcp_server_port)
    async with app.run_test() as pilot:
        await pilot.pause()
        caption = app.query_one("#signature_caption", Static)
        assert str(caption.content) == "Arguments"


async def test_tui_signature_panel_shows_placeholder_initially(tcp_server_port: int) -> None:
    app = IppDmeTui("127.0.0.1", tcp_server_port)
    async with app.run_test() as pilot:
        await pilot.pause()
        panel = app.query_one("#signature_text", Static)
        assert str(panel.content) == _HELP_PLACEHOLDER


async def test_tui_sidebar_highlight_updates_the_signature_panel(tcp_server_port: int) -> None:
    app = IppDmeTui("127.0.0.1", tcp_server_port)
    async with app.run_test() as pilot:
        await pilot.pause()
        list_view = app.query_one("#sidebar", ListView)
        panel = app.query_one("#signature_text", Static)
        list_view.focus()

        list_view.index = 0
        await pilot.pause()
        assert str(panel.content) != _HELP_PLACEHOLDER


async def test_tui_help_meta_command_shows_a_signature(tcp_server_port: int) -> None:
    app = IppDmeTui("127.0.0.1", tcp_server_port)
    async with app.run_test() as pilot:
        await pilot.pause()
        command_input = app.query_one("#command_input", Input)
        panel = app.query_one("#signature_text", Static)
        command_input.focus()

        for char in ".help GoTo":
            await pilot.press(char)
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        assert command_input.value == ""  # the line was consumed, not sent to the server
        text = str(panel.content)
        assert "GoTo" not in text  # no command-name header - just the parameters
        assert r"Positions \[enum]" in text  # mandatory is the default, left unmarked
        assert r"Positions* \[enum]" not in text
        assert r"Sync* \[int]" in text


async def test_tui_help_meta_command_is_never_sent_to_the_server(tcp_server_port: int) -> None:
    app = IppDmeTui("127.0.0.1", tcp_server_port)
    async with app.run_test() as pilot:
        await pilot.pause()
        command_input = app.query_one("#command_input", Input)
        command_input.focus()

        for char in ".help GoTo":
            await pilot.press(char)
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        log_lines = [line.text for line in app.query_one("#log", RichLog).lines]
        # A real GoTo(...) sent to the server would produce an Ack ("&")
        # line via _send() - .help never reaches that code path at all.
        assert "&" not in log_lines
        assert not any("Parse error" in line for line in log_lines)


async def test_tui_help_meta_command_output_is_echoed_to_the_log(tcp_server_port: int) -> None:
    """Regression test: .help's response must be visible in the main log, not only the panel.

    The signature panel alone isn't where anyone watching the log for a
    response to what they just typed would think to look.
    """
    app = IppDmeTui("127.0.0.1", tcp_server_port)
    async with app.run_test() as pilot:
        await pilot.pause()
        command_input = app.query_one("#command_input", Input)
        command_input.focus()

        for char in ".help GoTo":
            await pilot.press(char)
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        log_text = "\n".join(line.text for line in app.query_one("#log", RichLog).lines)
        assert ".help GoTo" in log_text
        assert "Positions [enum]" in log_text
        assert "Sync* [int]" in log_text


async def test_tui_help_meta_command_without_argument_shows_usage(tcp_server_port: int) -> None:
    app = IppDmeTui("127.0.0.1", tcp_server_port)
    async with app.run_test() as pilot:
        await pilot.pause()
        command_input = app.query_one("#command_input", Input)
        panel = app.query_one("#signature_text", Static)
        command_input.focus()

        command_input.value = ".help"
        await pilot.press("enter")
        await pilot.pause()

        assert str(panel.content) == _HELP_USAGE


async def test_tui_help_meta_command_for_an_unknown_command(tcp_server_port: int) -> None:
    app = IppDmeTui("127.0.0.1", tcp_server_port)
    async with app.run_test() as pilot:
        await pilot.pause()
        command_input = app.query_one("#command_input", Input)
        panel = app.query_one("#signature_text", Static)
        command_input.focus()

        command_input.value = ".help Bogus"
        await pilot.press("enter")
        await pilot.pause()

        assert str(panel.content) == "Unknown command: Bogus"


async def test_tui_help_meta_command_is_case_insensitive(tcp_server_port: int) -> None:
    app = IppDmeTui("127.0.0.1", tcp_server_port)
    async with app.run_test() as pilot:
        await pilot.pause()
        command_input = app.query_one("#command_input", Input)
        panel = app.query_one("#signature_text", Static)
        command_input.focus()

        command_input.value = ".help goto"
        await pilot.press("enter")
        await pilot.pause()

        assert r"Positions \[enum]" in str(panel.content)


async def test_tui_unknown_meta_command_is_reported_in_the_log(tcp_server_port: int) -> None:
    app = IppDmeTui("127.0.0.1", tcp_server_port)
    async with app.run_test() as pilot:
        await pilot.pause()
        command_input = app.query_one("#command_input", Input)
        command_input.focus()

        command_input.value = ".bogus"
        await pilot.press("enter")
        await pilot.pause()

        log_text = "\n".join(str(line) for line in app.query_one("#log", RichLog).lines)
        assert "Unknown meta-command" in log_text


async def test_tui_help_zero_argument_command(tcp_server_port: int) -> None:
    """A command with nothing to list renders as an empty panel, not a placeholder message."""
    app = IppDmeTui("127.0.0.1", tcp_server_port)
    async with app.run_test() as pilot:
        await pilot.pause()
        command_input = app.query_one("#command_input", Input)
        panel = app.query_one("#signature_text", Static)
        command_input.focus()

        command_input.value = ".help StartSession"
        await pilot.press("enter")
        await pilot.pause()

        assert str(panel.content) == ""


async def test_tui_enter_always_submits_even_with_the_dropdown_open(
    tcp_server_port: int,
) -> None:
    """Regression test: Enter must submit the line, not just silently accept a suggestion."""
    app = IppDmeTui("127.0.0.1", tcp_server_port)
    async with app.run_test() as pilot:
        await pilot.pause()
        command_input = app.query_one("#command_input", Input)
        autocomplete = app.query_one(AutoComplete)
        command_input.focus()

        for char in ".help GoTo":
            await pilot.press(char)
        await pilot.pause()
        assert autocomplete.display

        await pilot.press("enter")
        await pilot.pause()

        assert command_input.value == ""


async def test_tui_command_input_suggests_meta_commands_after_a_dot(
    tcp_server_port: int,
) -> None:
    app = IppDmeTui("127.0.0.1", tcp_server_port)
    async with app.run_test() as pilot:
        await pilot.pause()
        command_input = app.query_one("#command_input", Input)
        autocomplete = app.query_one(AutoComplete)
        command_input.focus()

        for char in ".he":
            await pilot.press(char)
        await pilot.pause()

        assert autocomplete.display
        assert _dropdown_values(autocomplete) == ["help"]


async def test_tui_command_input_tab_completes_a_meta_command_name(
    tcp_server_port: int,
) -> None:
    app = IppDmeTui("127.0.0.1", tcp_server_port)
    async with app.run_test() as pilot:
        await pilot.pause()
        command_input = app.query_one("#command_input", Input)
        command_input.focus()

        for char in ".he":
            await pilot.press(char)
        await pilot.pause()
        await pilot.press("tab")
        await pilot.pause()

        assert command_input.value == ".help"


async def test_tui_command_input_suggests_command_names_after_help_completes(
    tcp_server_port: int,
) -> None:
    """Regression test: completing ".help" must not swallow the command-name argument slot."""
    app = IppDmeTui("127.0.0.1", tcp_server_port)
    async with app.run_test() as pilot:
        await pilot.pause()
        command_input = app.query_one("#command_input", Input)
        autocomplete = app.query_one(AutoComplete)
        command_input.focus()

        for char in ".help Go":
            await pilot.press(char)
        await pilot.pause()

        assert autocomplete.display
        assert set(_dropdown_values(autocomplete)) >= {"GoTo"}


async def test_tui_sidebar_selection_of_a_zero_argument_command_inserts_the_closed_call(
    tcp_server_port: int,
) -> None:
    app = IppDmeTui("127.0.0.1", tcp_server_port)
    async with app.run_test() as pilot:
        await pilot.pause()
        list_view = app.query_one("#sidebar", ListView)
        names = [item.name for item in list_view.children]
        list_view.focus()
        list_view.index = names.index("StartSession")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        command_input = app.query_one("#command_input", Input)
        assert command_input.value == "StartSession()"
        assert command_input.cursor_position == len("StartSession(")


async def test_tui_sidebar_selection_of_a_command_with_arguments_opens_completion(
    tcp_server_port: int,
) -> None:
    """Only the opening paren is inserted - the rest is the same completion flow as typing it."""
    app = IppDmeTui("127.0.0.1", tcp_server_port)
    async with app.run_test() as pilot:
        await pilot.pause()
        list_view = app.query_one("#sidebar", ListView)
        names = [item.name for item in list_view.children]
        list_view.focus()
        list_view.index = names.index("GoTo")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        command_input = app.query_one("#command_input", Input)
        assert command_input.value == "GoTo("
        assert command_input.cursor_position == len("GoTo(")

        autocomplete = app.query_one(AutoComplete)
        assert autocomplete.display
        assert set(_dropdown_values(autocomplete)) >= {"Sync", "X", "Y", "Z", "R"}


async def test_tui_sidebar_selection_of_a_positional_command_shows_the_live_hint_instead(
    tcp_server_port: int,
) -> None:
    """No dropdown for a positional command (nothing to suggest) - the bold hint takes over."""
    app = IppDmeTui("127.0.0.1", tcp_server_port)
    async with app.run_test() as pilot:
        await pilot.pause()
        list_view = app.query_one("#sidebar", ListView)
        names = [item.name for item in list_view.children]
        list_view.focus()
        list_view.index = names.index("CenterPart")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        command_input = app.query_one("#command_input", Input)
        assert command_input.value == "CenterPart("

        autocomplete = app.query_one(AutoComplete)
        assert not autocomplete.display
        panel = app.query_one("#signature_text", Static)
        assert r"[bold]- Px \[float][/bold]" in str(panel.content)
