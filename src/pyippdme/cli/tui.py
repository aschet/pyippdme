# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""A terminal UI for driving an I++ DME server interactively (built on Textual).

Same underlying mechanics as :mod:`pyippdme.cli.client` (a generic client
that parses and sends whatever method call it is given, via
:func:`pyippdme.cli._interaction.run_command_line`) with a fuller-screen
interface: a scrollback log of every command/response, a connection bar,
a sidebar of known command names, a real autocompletion dropdown, and
command history - none of which the barebones client has. Known command
names (seeded from :data:`~pyippdme.simulation.catalog.BUILTIN_COMMANDS`,
purely for reference - the TUI itself does not require the peer to
implement any of them) drive the sidebar and, via
:class:`~textual_autocomplete.AutoComplete` (a
third-party dropdown widget - Textual's own built-in
:class:`~textual.suggester.Suggester` can only ever show one ghost-text
suggestion, which isn't enough when several commands share a prefix,
e.g. ``EnableUser``/``EnableRotaryTableVarCsy``), a real multi-candidate
dropdown. :func:`_completion_context` makes that dropdown context-aware:
before any ``(``, it suggests command names; once inside one command's
top-level parens, it suggests *that* command's argument names instead
(from the same catalog's :class:`~pyippdme.protocol.signature.Parameter`
data) - see its docstring for the ``DataType.ENUM`` caveat this
implies. A "Kind U" positional-style command (e.g. ``ScanOnLine(1, 2,
3, ...)`` - bare values in a fixed order, not ``Name(value)`` pairs;
see :attr:`~pyippdme.protocol.signature.Parameter.positional`) instead
gets no argument suggestions at all, since its parameter names aren't
something you ever type - :func:`_positional_argument_index` covers
that gap by pointing at the current one (rendered bold) in the
signature panel instead, live, as the cursor moves past each comma,
autoscrolling to keep it visible if it's past the panel's fixed height.
Up/Down in the command input cycle through previously submitted
commands, like shell history, whenever the dropdown isn't open to claim
those keys for its own navigation instead.

A signature panel below the sidebar (:func:`_format_signature`, again
from the same catalog data - just the parameter list itself, one line
per argument with its wire datatype and, marked with a trailing ``*``,
whether it's optional; not the command's own name, which the sidebar
and command input already show, and not a prose description of what a
command *does*, since that text doesn't exist anywhere in this library
yet) updates as you arrow through the sidebar, as you type inside a
command's parens (see above), or on
demand for any command via the one supported ``.``-prefixed
meta-command, ``.help <command name>`` (typed into the command input
like any other line, echoed to the main log like any real command's
response - never sent to the server itself, though). Meta-commands get
the same dropdown treatment while their own name is still being typed
(``.he`` -> ``help``); once one is finished, whatever follows completes
exactly like a bare command name would (e.g. ``.help Go`` -> ``.help
GoTo``), since :func:`_completion_context` never looks at the leading
``.`` at all.

Selecting a sidebar entry (click or :kbd:`Enter`) fills the command
input directly: ``Name()`` for a zero-argument command, cursor before
the closing paren, or just ``Name(`` for one that takes arguments,
cursor right after it - which, since a programmatic ``Input.Changed``
still reaches :class:`_CommandAutoComplete` like any other, immediately
opens the same context-aware completion dropdown typing the ``(`` by
hand would have. Nothing more elaborate than that: no separate dialog,
just a head start on the normal typing flow. Completing a bare command
name from the dropdown itself (:meth:`_CommandAutoComplete.apply_completion`)
does the identical thing, whether accepted with :kbd:`Tab`, a click, or
:kbd:`Enter` - the last of which also has to *not* let that same
keystroke fall through to submitting the now-reopened, still-incomplete
call, unlike every other completion; see
:attr:`_CommandAutoComplete.prevent_default_enter`'s docstring.

Requires the optional ``tui`` dependency group (``pip install
pyippdme[tui]``); imported lazily by :mod:`pyippdme.cli.main` so the base
install does not need ``textual``/``textual-autocomplete``.
"""

from __future__ import annotations

import re
from typing import ClassVar

from rich.markup import escape as escape_markup
from textual import events
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    RichLog,
    Static,
)
from textual_autocomplete import AutoComplete, DropdownItem, TargetState

from pyippdme.cli._interaction import (
    Acked,
    Completed,
    ConnectionLost,
    Failed,
    ParseFailed,
    Received,
    format_error,
    run_command_line,
)
from pyippdme.cli.script import VIRTUAL_HOST, describe_address
from pyippdme.client import IppDmeClient
from pyippdme.exceptions import IppDmeConnectionError
from pyippdme.protocol.network import TCP_NETWORK, MemoryNetwork
from pyippdme.protocol.parameters import ParameterName
from pyippdme.protocol.signature import DataType
from pyippdme.protocol.transport import DEFAULT_PORT
from pyippdme.server import IppDmeServer
from pyippdme.simulation.catalog import BUILTIN_COMMANDS
from pyippdme.simulation.state import SimulationState

#: Fallback offered whenever the active command has at least one
#: ``DataType.ENUM`` top-level argument (``GoTo``'s ``Positions``,
#: ``Get``'s ``Axes``, ...). The standard's own ``GetSupportedArguments``
#: deliberately stops at that one schema label - "what belongs inside an
#: enum" isn't described anywhere machine-readable (see
#: :mod:`pyippdme.protocol.signature`'s module docstring) - so suggesting
#: the label itself (e.g. literally ``Positions(``) would be actively
#: wrong, nobody types that. In practice, across every built-in command,
#: what actually gets typed inside one of these is always drawn from this
#: same small set of axis-style names, so offering them is a pragmatic,
#: deliberately-scoped stand-in rather than a generic solution.
_ENUM_ARGUMENT_FALLBACK = (
    ParameterName.X,
    ParameterName.Y,
    ParameterName.Z,
    ParameterName.R,
)

_IDENTIFIER_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*$")

#: The only meta-commands :meth:`IppDmeTui._meta_command` recognizes; kept
#: in sync with it by hand since there's only the one.
_META_COMMANDS = ("help",)
_META_COMMAND_PREFIX_RE = re.compile(r"^\.([A-Za-z]*)$")


def _known_command_names() -> list[str]:
    return list(BUILTIN_COMMANDS)


def _completion_context(text_before_cursor: str) -> tuple[str | None, str]:
    """Parse text up to the cursor into ``(command whose args we're inside, partial word)``.

    ``command`` is ``None`` when the cursor isn't inside any command's
    parentheses at all (suggest command names); otherwise it's the name
    of the *outermost* open command, even if the cursor is actually
    nested deeper still (e.g. inside ``ScanOnCurve(Format(``) - nested
    groups are rare enough in this protocol (only ``ScanOnCurve`` has
    one) that reusing the outer command's argument names there, rather
    than tracking nesting precisely, is an acceptable simplification.
    """
    depth = 0
    command_name: str | None = None
    for index, char in enumerate(text_before_cursor):
        if char == "(":
            if depth == 0:
                match = _IDENTIFIER_RE.search(text_before_cursor[:index])
                command_name = match.group(0) if match else None
            depth += 1
        elif char == ")" and depth > 0:
            depth -= 1
            if depth == 0:
                command_name = None
    partial_match = _IDENTIFIER_RE.search(text_before_cursor)
    partial = partial_match.group(0) if partial_match else ""
    return (command_name if depth >= 1 else None), partial


def _completion_candidates(state: TargetState) -> list[DropdownItem]:
    prefix = state.text[: state.cursor_position]
    meta_prefix = _META_COMMAND_PREFIX_RE.match(prefix)
    if meta_prefix is not None:
        # Still typing the "."-word itself (e.g. ".", ".h", ".help") - offer
        # meta-command names, not built-in protocol commands (those only
        # become relevant once a meta-command like ".help " is finished and
        # its own argument starts, which falls through to the branches
        # below exactly like a bare command name would).
        needle = meta_prefix.group(1).casefold()
        return [DropdownItem(name) for name in _META_COMMANDS if name.casefold().startswith(needle)]
    command_name, partial = _completion_context(prefix)
    if command_name is None and not partial:
        # Nothing typed and not inside any command's parens: offering all
        # ~90 built-in commands here wouldn't be a useful suggestion, and
        # crucially, an empty candidate list is also what frees Down back
        # up for command history (AutoComplete claims Down outright to
        # open its dropdown whenever it has *any* loaded options, even
        # hidden ones - see IppDmeTui.on_key).
        return []
    if command_name is None:
        names: tuple[str, ...] = tuple(BUILTIN_COMMANDS)
    else:
        info = BUILTIN_COMMANDS.get(command_name)
        arguments = info.arguments if info is not None else None
        if not arguments or all(p.positional for p in arguments):
            # Either nothing on record, or a "Kind U" positional-style command
            # (e.g. ScanOnLine(1, 2, 3, ...)) - its parameter *names* aren't
            # something you ever type, only their values, in order, so there's
            # nothing to suggest here (see Parameter.positional's docstring).
            names = ()
        else:
            named = tuple(p.name for p in arguments if p.datatype != DataType.ENUM)
            has_enum_argument = any(p.datatype == DataType.ENUM for p in arguments)
            names = (*named, *_ENUM_ARGUMENT_FALLBACK) if has_enum_argument else named
    needle = partial.casefold()
    return [DropdownItem(name) for name in names if name.casefold().startswith(needle)]


_HELP_PLACEHOLDER = "Highlight a command in the list, or type .help <name>, to see its arguments."
_HELP_USAGE = "Usage: .help <command name>"


def _find_command_name(query: str) -> str | None:
    """Resolve ``query`` to its real, correctly-cased key in ``BUILTIN_COMMANDS``, if any."""
    if query in BUILTIN_COMMANDS:
        return query
    needle = query.casefold()
    return next((name for name in BUILTIN_COMMANDS if name.casefold() == needle), None)


def _positional_argument_index(text_before_cursor: str) -> tuple[str | None, int]:
    """Parse text up to the cursor into ``(command we're inside, 0-based argument index)``.

    The index is how many top-level commas have been seen since that
    command's own opening paren - i.e. which positional slot the cursor is
    currently in. Same depth-tracking as :func:`_completion_context` (see
    its docstring for the nested-group caveat), but counting commas
    instead of extracting the partial word, since the two are needed in
    different places: this one only to point at the right line in the
    signature panel while typing a "Kind U" positional command (see
    :attr:`~pyippdme.protocol.signature.Parameter.positional`), where a
    dropdown suggestion would be wrong (see :func:`_completion_candidates`).
    """
    depth = 0
    command_name: str | None = None
    arg_index = 0
    for index, char in enumerate(text_before_cursor):
        if char == "(":
            if depth == 0:
                match = _IDENTIFIER_RE.search(text_before_cursor[:index])
                command_name = match.group(0) if match else None
                arg_index = 0
            depth += 1
        elif char == ")" and depth > 0:
            depth -= 1
            if depth == 0:
                command_name = None
        elif char == "," and depth == 1:
            arg_index += 1
    return (command_name if depth >= 1 else None), arg_index


def _format_signature(query: str, *, highlight_index: int | None = None) -> str:
    """Render a command's known parameters as a plain, unindented bullet list.

    Only what the static catalog actually has - names, wire datatypes, and
    which are optional (marked with a trailing ``*``, mandocs-style;
    mandatory is the default and left unmarked, since most parameters
    are - flagging every single one would just be noise) - not the
    command's own name (the sidebar/command input already show that) or
    a prose description of what it *does*: that text doesn't exist
    anywhere in this library yet (see :mod:`pyippdme.simulation.catalog`'s
    ``CommandInfo``). A command with nothing to list (no arguments at
    all, or no schema on record) renders as an empty string, not a
    placeholder message - there's nothing to show, so the panel shows
    nothing.

    ``highlight_index``, if given, renders that positional slot in bold
    instead of the usual plain "``- name [type]``" - only meaningful (and
    only ever passed) for a "Kind U" positional command, where
    :func:`_positional_argument_index` can tell which bare value the
    cursor is currently on; irrelevant for a named command, where
    argument order doesn't matter. Bold rather than a distinct leading
    marker (e.g. ``->``) deliberately keeps every line's bullet the same
    single character - the signature panel (a ``Static``, ``markup=True``
    by default) renders this as Rich console markup, and a marker whose
    width changed line to line made the whole list visibly shift as the
    cursor moved between arguments.
    """
    name = _find_command_name(query)
    if name is None:
        return f"Unknown command: {escape_markup(query)}"
    info = BUILTIN_COMMANDS[name]
    if not info.arguments:
        return ""
    lines = []
    for index, param in enumerate(info.arguments):
        star = "" if param.mandatory else "*"
        # Escaped as a whole, not just the interpolated name/datatype:
        # markup=True parses a literal "[float]" as an attempted (and
        # here, unrecognized - silently dropped) style tag otherwise,
        # same as the RichLog bug fixed for the main log.
        line = escape_markup(f"- {param.name}{star} [{param.datatype}]")
        lines.append(f"[bold]{line}[/bold]" if index == highlight_index else line)
    return "\n".join(lines)


class _CommandAutoComplete(AutoComplete):
    """Matches :class:`AutoComplete`'s own fuzzy-filtering to what we already filtered by.

    By default it fuzzy-matches candidates against the *entire* text
    before the cursor (``get_search_string``'s default), which would
    wrongly discard :func:`_completion_candidates`'s results - those are
    already filtered against only the current partial word (``"Sy"`` in
    ``GoTo(Sy``, not ``"GoTo(Sy"``). Returning that same partial word here
    keeps the two in sync, so the built-in fuzzy step (used for sorting
    and match highlighting) sees what it expects instead of re-filtering
    against unrelated text.
    """

    #: Set by apply_completion() when it just auto-opened a command's
    #: parens; read (and cleared) by post_completion() right after - see
    #: both for why.
    _reopen_dropdown_after_completion: bool = False

    @property
    def prevent_default_enter(self) -> bool:
        """Whether Enter should *only* accept the highlighted completion, not also submit.

        The base class treats this as a fixed True/False set once at
        construction; overridden as a property instead so it can be
        recomputed on every keypress from what accepting the *currently*
        highlighted completion would actually do - _listen_to_messages's
        Enter handler reads ``self.prevent_default_enter`` immediately
        before deciding whether to prevent the default (submit) action,
        so this is the only point early enough to influence that decision
        (apply_completion/post_completion run afterward, too late).

        True (block the submit, accept-only, like Tab) exactly when the
        highlighted option is a bare, known command name that
        apply_completion would auto-open the parens of - accepting
        "GoTo" as ".help GoTo(" and immediately submitting *that* would
        send garbage and clear the input, the same "text vanishes"
        failure mode a plain Tab press never had. False (accept and let
        the same keystroke also submit) for everything else, including
        the exact-match case this whole mechanism was first built for
        (".help GoTo" already typed in full - nothing to expand, Enter
        should just submit) and finishing an already-open call.
        """
        if not self.display or not self.option_list.option_count:
            return False
        if self.target.value.startswith("."):
            return False  # a meta-command's own argument - see apply_completion
        command_name, _partial = _completion_context(
            self.target.value[: self.target.cursor_position]
        )
        if command_name is not None:
            return False
        option = self.option_list.get_option_at_index(self.option_list.highlighted or 0)
        assert isinstance(option, DropdownItem)  # noqa: S101 (narrows get_option_at_index)
        info = BUILTIN_COMMANDS.get(option.value)
        return info is not None and bool(info.arguments)

    @prevent_default_enter.setter
    def prevent_default_enter(self, value: bool) -> None:
        del value  # AutoComplete.__init__ assigns a starting value; always computed live instead

    def get_search_string(self, state: TargetState) -> str:
        return _completion_context(state.text[: state.cursor_position])[1]

    def should_show_dropdown(self, search_string: str) -> bool:
        del search_string  # required by the overridden signature; see below
        # The base implementation also hides the dropdown whenever
        # search_string (here, the current partial word) is empty - right
        # for a genuinely empty input (where _completion_candidates already
        # returns no candidates at all, so option_count == 0 below already
        # covers it), wrong the moment a "(" opens a known command's
        # argument list: "GoTo(" has an empty partial word too, but there
        # are real candidates (Sync, X, Y, Z, R) worth showing immediately.
        #
        # Deliberately not reusing the base class's other special case
        # (hide when the sole remaining candidate already equals what's
        # typed, e.g. having fully typed "Y" for GoTo(X(10), Y): its own
        # Tab handler only calls prevent_default()/stop() while the
        # dropdown is displayed, and _complete() itself is a no-op while
        # hidden - so hiding here would make Tab silently do nothing
        # *and* fall through to its default focus-switch action instead
        # of accepting the (admittedly redundant) completion.
        return self.option_list.option_count > 0

    def apply_completion(self, value: str, state: TargetState) -> None:
        # The base implementation replaces the *entire* input with just the
        # picked value - fine for completing a bare command name (the whole
        # input was the partial name anyway), but wrong once there's more
        # context around it: accepting "Sync" while completing "GoTo(Sy"
        # must produce "GoTo(Sync", not "Sync" on its own. Replace only the
        # current partial word (the same span get_search_string reports)
        # with the chosen value, leaving the rest of the text untouched.
        text = state.text
        cursor = state.cursor_position
        command_name, partial = _completion_context(text[:cursor])
        start = cursor - len(partial)
        target = self.target
        suffix = ""
        cursor_offset = len(value)
        if command_name is None and not text.startswith(".") and value in BUILTIN_COMMANDS:
            # Completing a bare, known command name at the top level of a
            # real call - not an argument name inside one already-open
            # (command_name would be set), and not ".help GoTo"'s own
            # argument either (text.startswith(".")): that reuses this
            # exact same bare-command-name completion path (see the
            # module docstring) but must stay a plain name - ".help"
            # only ever wants that, never "GoTo(...)". For everything
            # this guard does let through, keep going exactly like
            # selecting it in the sidebar would: a zero-argument command
            # gets its whole closed "()", cursor before the paren; one
            # that takes arguments gets just "(", cursor right after it,
            # immediately reopening completion for its own argument names
            # instead of leaving the user to type the "(" by hand.
            if BUILTIN_COMMANDS[value].arguments:
                suffix = "("
                cursor_offset = len(value) + 1
                self._reopen_dropdown_after_completion = True
            else:
                suffix = "()"
                cursor_offset = len(value) + 1  # before the closing paren
        target.value = text[:start] + value + suffix + text[cursor:]
        target.cursor_position = start + cursor_offset
        # Input.Changed is suppressed for this assignment (see
        # update_signature_panel_for_command_input's docstring), so the
        # signature panel needs a direct nudge here instead.
        if isinstance(self.app, IppDmeTui):
            self.app.update_signature_panel_for_command_input(target.value, target.cursor_position)

    def post_completion(self) -> None:
        # The base implementation unconditionally hides the dropdown after
        # any accepted completion - right after completing an argument
        # name, wrong right after apply_completion() just auto-opened a
        # command's parens: that's exactly the moment a new dropdown (that
        # command's own argument names) needs to appear, matching what
        # selecting it in the sidebar already does. _handle_target_update
        # is AutoComplete's own "recompute everything for the current
        # target state" method (same one a real keystroke would trigger
        # via Input.Changed, which this path doesn't get - see
        # apply_completion) - no public equivalent exists.
        if self._reopen_dropdown_after_completion:
            self._reopen_dropdown_after_completion = False
            self._handle_target_update()
        else:
            super().post_completion()


class IppDmeTui(App[None]):
    """A Textual application for interactively driving an I++ DME server."""

    CSS = """
    #sidebar_panel {
        width: 24;
    }
    #sidebar_caption {
        text-style: bold;
    }
    #sidebar {
        height: 1fr;
        border: solid $accent;
    }
    #signature_caption {
        text-style: bold;
        margin-top: 1;
    }
    #signature_panel {
        height: 12;
        border: solid $accent;
        padding: 0 1;
    }
    #main {
        width: 1fr;
    }
    #connection_bar {
        height: 3;
        dock: top;
    }
    #host_input, #port_input {
        width: 20;
    }
    #log {
        border: solid $accent;
    }
    """

    BINDINGS: ClassVar = [("ctrl+c", "quit", "Quit")]

    def __init__(
        self, host: str | None = None, port: int = DEFAULT_PORT, *, virtual: bool = False
    ) -> None:
        super().__init__()
        self._initial_host = host
        self._initial_port = port
        self._virtual = virtual
        self._embedded_server: IppDmeServer[SimulationState] | None = None
        self.client: IppDmeClient | None = None
        #: Previously submitted commands, oldest first (see :meth:`on_key`'s
        #: Up/Down cycling, mirroring shell command history).
        self._history: list[str] = []
        #: Index into ``_history`` while cycling, or ``None`` when not.
        self._history_index: int | None = None
        #: What the user had typed before they started cycling, restored on
        #: Down past the newest history entry.
        self._pending_input: str = ""

    def compose(self) -> ComposeResult:
        command_names = _known_command_names()
        yield Header()
        with Horizontal():
            with Vertical(id="main"):
                with Horizontal(id="connection_bar"):
                    yield Input(
                        value=self._initial_host or "127.0.0.1", placeholder="host", id="host_input"
                    )
                    yield Input(value=str(self._initial_port), placeholder="port", id="port_input")
                    yield Button("Connect", id="connection_button", variant="success")
                yield RichLog(id="log", wrap=True, highlight=True, markup=True)
                yield Input(
                    placeholder="StartSession(), GoTo(X(10), Y(20)), ...",
                    id="command_input",
                    # The default (True) selects the whole value on focus,
                    # which would clobber the cursor position
                    # _fill_command_input() deliberately places right before
                    # a zero-argument command's closing paren for the sidebar/
                    # dialog flows (Input.focus()'s own Focus handler runs
                    # after that assignment, since it's dispatched through
                    # the message queue rather than synchronously).
                    select_on_focus=False,
                )
                yield _CommandAutoComplete(
                    target="#command_input",
                    candidates=_completion_candidates,
                    # prevent_default_enter is a computed property on
                    # _CommandAutoComplete, not a fixed setting - see its
                    # docstring for what Enter actually does in each case.
                )
            with Vertical(id="sidebar_panel"):
                yield Static("Commands", id="sidebar_caption")
                yield ListView(
                    *(ListItem(Label(name), name=name) for name in command_names),
                    id="sidebar",
                    initial_index=None,  # match _HELP_PLACEHOLDER: nothing highlighted yet
                )
                yield Static("Arguments", id="signature_caption")
                with VerticalScroll(id="signature_panel"):
                    yield Static(_HELP_PLACEHOLDER, id="signature_text")
        yield Footer()

    async def on_mount(self) -> None:
        if self._virtual:
            from pyippdme.simulation.virtual_cmm import VirtualCMM

            self._embedded_server = VirtualCMM(network=MemoryNetwork())
            port = await self._embedded_server.start()
            self._append_log("[green]Started an in-process VirtualCMM[/green]")
            self.query_one("#host_input", Input).value = VIRTUAL_HOST
            self.query_one("#port_input", Input).value = str(port)
            await self._connect(VIRTUAL_HOST, port)
        elif self._initial_host:
            await self._connect(self._initial_host, self._initial_port)
        self.query_one("#command_input", Input).focus()

    async def on_unmount(self) -> None:
        # Without this, quitting the app (Ctrl+C, closing the terminal, ...)
        # without first clicking "Disconnect" leaks the connection: IppDmeClient
        # runs a background reader task for as long as it is open, which
        # otherwise keeps running (and the socket keeps the peer's connection
        # alive) after the UI itself is gone. Closes the client directly
        # rather than going through _disconnect(), which logs to a widget
        # tree that no longer exists by the time on_unmount runs.
        if self.client is not None:
            await self.client.close()
            self.client = None
        if self._embedded_server is not None:
            await self._embedded_server.close()
            self._embedded_server = None

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "connection_button":
            return
        if self.client is not None:
            await self._disconnect()
            return
        host = self.query_one("#host_input", Input).value.strip()
        port_text = self.query_one("#port_input", Input).value.strip()
        if not host:
            self._append_log("[red]Enter a host first.[/red]")
            return
        try:
            port = int(port_text)
        except ValueError:
            self._append_log(f"[red]Invalid port: {escape_markup(port_text)!r}[/red]")
            return
        await self._connect(host, port)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        # A command with arguments gets its opening paren only, with the
        # cursor right after it - Input.Changed still fires for a
        # programmatic .value assignment, so this immediately opens the
        # completion dropdown with that command's own argument names (or,
        # for a "Kind U" positional command with no names to suggest, the
        # live -> hint in the signature panel takes over instead - see
        # _positional_argument_index). A zero-argument command still just
        # gets the whole closed call, cursor before the closing paren.
        name = event.item.name
        if not name:
            return
        info = BUILTIN_COMMANDS.get(name)
        call_text = f"{name}(" if info is not None and info.arguments else f"{name}()"
        command_input = self.query_one("#command_input", Input)
        command_input.value = call_text
        command_input.focus()
        command_input.cursor_position = (
            len(call_text) - 1 if call_text.endswith("()") else len(call_text)
        )

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        if event.list_view.id != "sidebar":
            return
        name = event.item.name if event.item is not None else None
        text = _format_signature(name) if name else _HELP_PLACEHOLDER
        self.query_one("#signature_text", Static).update(text)
        self.query_one("#signature_panel", VerticalScroll).scroll_home(animate=False)

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "command_input":
            return
        self.update_signature_panel_for_command_input(event.value, event.input.cursor_position)

    def update_signature_panel_for_command_input(self, text: str, cursor_position: int) -> None:
        """Point at the current argument in the signature panel while typing.

        Only does anything once the cursor is inside some command's top-level
        parens - most useful for a "Kind U" positional command (e.g.
        CenterPart(10, 20, ...)), which gets no dropdown suggestions at all
        (see _completion_candidates) since there's no name to suggest, only
        a value; the bold-highlighted line is the only in-context hint of
        which argument comes next.

        Called both from on_input_changed (typing) and from
        _CommandAutoComplete.apply_completion (accepting a Tab/Enter/click
        completion) - AutoComplete._complete() wraps its own call to
        apply_completion() in ``with self.prevent(Input.Changed)`` (to
        avoid re-triggering its own dropdown-rebuild listener), which also
        silently swallows *this* widget's on_input_changed, so the panel
        would otherwise stay stuck on whichever argument was current before
        the completion was accepted.
        """
        command_name, arg_index = _positional_argument_index(text[:cursor_position])
        if command_name is None:
            return
        info = BUILTIN_COMMANDS.get(command_name)
        if info is None or not info.arguments:
            return
        highlight_index = arg_index if all(p.positional for p in info.arguments) else None
        self.query_one("#signature_text", Static).update(
            _format_signature(command_name, highlight_index=highlight_index)
        )
        if highlight_index is not None:
            self._reveal_signature_panel_line(highlight_index)

    def _reveal_signature_panel_line(self, line_index: int) -> None:
        """Scroll the signature panel just enough to keep ``line_index`` visible.

        _format_signature()'s output is one line per argument with nothing
        else mixed in (see its docstring), so the highlighted argument's
        index *is* its line number here - a command with more arguments
        than the panel's fixed height (e.g. ScanOnHelix, 14) would
        otherwise scroll the bold marker straight out of view as the
        cursor moves past each comma, with nothing telling you it's still
        there below the fold.
        """
        container = self.query_one("#signature_panel", VerticalScroll)
        viewport_height = container.size.height
        if viewport_height <= 0:
            return
        if line_index < container.scroll_y:
            container.scroll_to(y=line_index, animate=False)
        elif line_index >= container.scroll_y + viewport_height:
            container.scroll_to(y=line_index - viewport_height + 1, animate=False)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "command_input":
            return
        text = event.value.strip()
        event.input.value = ""
        self._history_index = None
        self._pending_input = ""
        if not text:
            return
        if not self._history or self._history[-1] != text:
            self._history.append(text)
        if text.startswith("."):
            self._meta_command(text[1:].strip())
        else:
            await self._send(text)

    def _meta_command(self, command: str) -> None:
        """Handle a ``.``-prefixed input - currently just ``.help [command name]``.

        Echoed to the main log like any other typed input, in addition to
        updating the signature panel - the panel alone isn't where anyone
        watching the log for a response to what they just typed would
        think to look.
        """
        self._append_log(f"[bold]> .{escape_markup(command)}[/bold]")
        parts = command.split()
        name = parts[0].lower() if parts else ""
        if name == "help":
            # _format_signature() already returns safe Rich markup (its
            # own literal "[float]"-style brackets pre-escaped, its bold
            # highlight a real [bold]/[/bold] pair) - escaping it again
            # here would double-escape the former and neuter the latter.
            # _HELP_USAGE has no brackets of its own, so it's unaffected
            # either way.
            text = _format_signature(parts[1]) if len(parts) > 1 else _HELP_USAGE
            self.query_one("#signature_text", Static).update(text)
            self.query_one("#signature_panel", VerticalScroll).scroll_home(animate=False)
            self._append_log(text)
        else:
            self._append_log(f"[red]Unknown meta-command: .{escape_markup(command)}[/red]")

    def on_key(self, event: events.Key) -> None:
        command_input = self.query_one("#command_input", Input)
        if self.focused is not command_input or event.key not in ("up", "down"):
            return
        # AutoComplete claims Up/Down for its own dropdown navigation while
        # it's open (and opens itself on Down when there's anything to show -
        # see textual_autocomplete's own key handling); only step through
        # history here when there's no dropdown up to fight with it over
        # those keys.
        if self.query_one(AutoComplete).display:
            return
        event.stop()
        event.prevent_default()
        if event.key == "up":
            self._history_prev(command_input)
        else:
            self._history_next(command_input)

    def _history_prev(self, command_input: Input) -> None:
        if not self._history:
            return
        if self._history_index is None:
            self._pending_input = command_input.value
            self._history_index = len(self._history)
        if self._history_index > 0:
            self._history_index -= 1
            command_input.value = self._history[self._history_index]
            command_input.cursor_position = len(command_input.value)

    def _history_next(self, command_input: Input) -> None:
        if self._history_index is None:
            return
        self._history_index += 1
        if self._history_index >= len(self._history):
            self._history_index = None
            command_input.value = self._pending_input
        else:
            command_input.value = self._history[self._history_index]
        command_input.cursor_position = len(command_input.value)

    async def _connect(self, host: str, port: int) -> None:
        await self._disconnect()
        try:
            network = (
                self._embedded_server.network
                if host == VIRTUAL_HOST and self._embedded_server is not None
                else TCP_NETWORK
            )
            self.client = await IppDmeClient.connect(host, port, network=network)
        except IppDmeConnectionError as exc:
            self._append_log(f"[red]Connection failed: {escape_markup(str(exc))}[/red]")
            return
        address = describe_address(host, port)
        self.sub_title = address
        self._append_log(f"[green]Connected to {escape_markup(address)}[/green]")
        self._update_connection_button()

    async def _disconnect(self) -> None:
        if self.client is not None:
            await self.client.close()
            self.client = None
            self.sub_title = ""
            self._append_log("[yellow]Disconnected[/yellow]")
        self._update_connection_button()

    def _update_connection_button(self) -> None:
        button = self.query_one("#connection_button", Button)
        if self.client is None:
            button.label = "Connect"
            button.variant = "success"
        else:
            button.label = "Disconnect"
            button.variant = "error"

    async def _send(self, text: str) -> None:
        if self.client is None:
            self._append_log("[red]Not connected. Fill in host/port and press Connect.[/red]")
            return
        self._append_log(f"[bold]> {escape_markup(text)}[/bold]")
        async for event in run_command_line(self.client, text):
            match event:
                case ParseFailed(error):
                    self._append_log(f"[red]Parse error: {escape_markup(str(error))}[/red]")
                case Acked():
                    self._append_log("&")
                case Received(payload):
                    self._append_log(f"# {escape_markup(payload.to_wire())}")
                case Completed():
                    self._append_log("%")
                case Failed(error):
                    self._append_log(f"[red]{escape_markup(format_error(error))}[/red]")
                case ConnectionLost(error):
                    self._append_log(f"[red]Connection lost: {escape_markup(str(error))}[/red]")

    def _append_log(self, text: str) -> None:
        """Write a line (or block) to the main log.

        ``text`` is Rich console markup (``[bold]``/``[red]``/... are real
        style tags) - any dynamic content spliced into it (user input,
        server responses) must go through :func:`escape_markup` first, or
        a literal ``[`` in that content (legal in a wire STRING - see
        :mod:`pyippdme.protocol.parser`'s grammar) gets silently parsed as
        an unrecognized markup tag and dropped instead of displayed.
        :func:`_format_signature`'s output is the one exception - already
        safe markup in its own right (its literal ``[datatype]`` brackets
        pre-escaped, its bold highlight real tags), escaping it *again*
        here would corrupt both.
        """
        self.query_one("#log", RichLog).write(text)


def run(host: str | None, port: int, *, virtual: bool = False) -> None:
    IppDmeTui(host, port, virtual=virtual).run()
