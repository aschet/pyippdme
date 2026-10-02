# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Lark-based parser for I++ DME command and response lines.

Implements the grammar in section 5.2. See the module
docstring of :mod:`pyippdme.protocol.ast` for the points where this grammar
deliberately deviates from (or generalizes beyond) a literal transcription
of the standard's EBNF, and why - notably, a ``name(...)`` group nested
inside a command's own argument list (``argument: ... | named_group | ...``)
uses the exact same fully recursive production as a response's top-level
named group (``items: named_group ...``); the two are otherwise identical,
so this grammar does not give them separate rules.

The grammar below uses Lark's Earley parser with a dynamic lexer rather than
a fixed tokenization pass, which matters for one real ambiguity in the
protocol text: a bare ``E0001``-shaped token is only an ``EventTag`` if it
is *not* followed by ``(``, resolved through grammar position (``argument``
lists ``EVENT_TAG`` and ``named_group`` as separate alternatives) rather
than manual backtracking.

Whitespace between tokens is uniformly ignored during parsing, which is more
lenient than the standard's "spaces are optional only around commas and
parentheses" rule. This only affects what we *accept*; everything this
library *sends* is rendered by :meth:`~pyippdme.protocol.ast` ``to_wire``
methods, which always emit exactly the spacing the standard requires.
"""

from __future__ import annotations

from lark import Lark, Token, Transformer
from lark.exceptions import LarkError, VisitError

from pyippdme.exceptions import IppDmeProtocolError
from pyippdme.protocol.ast import (
    AckResponse,
    Argument,
    BasicName,
    Command,
    DataResponse,
    DoneResponse,
    ErrorResponse,
    EventTag,
    Items,
    Method,
    NamedValue,
    NameValue,
    Number,
    NumericData,
    PropertyData,
    Response,
    String,
    StringValue,
    Xml,
    parse_tag,
)

_GRAMMAR = r"""
command: tag method
response: tag (ack | done | data_resp | error_resp)

tag: TAG | EVENT_TAG

ack: "&"
done: "%"
data_resp: "#" data
error_resp: "!" "Error" "(" DIGIT "," FOURDIGIT "," STRING "," STRING ")"

data: numeric_data | property_data | string_value | name_value | items | xml_value
numeric_data: NUMBER ("," NUMBER)*
property_data: STRING "," STRING
string_value: STRING
name_value: BASICNAME
xml_value: XML
items: named_group ("," named_group)*
named_group: NAME "(" (arglist | XML)? ")"

method: BASICNAME "(" method_body? ")"
method_body: arglist | XML

arglist: argument ("," argument)*
argument: STRING | NUMBER | named_group | EVENT_TAG | BASICNAME

NAME: BASICNAME ("." BASICNAME)*
BASICNAME: /[A-Za-z][A-Za-z0-9]*/
EVENT_TAG.2: /E\d{4}/
TAG: /\d{5}/
DIGIT: /\d/
FOURDIGIT: /\d{4}/
NUMBER: /[+-]?(\d+\.?\d*|\.\d+)([Ee][+-]?\d{1,3})?/
STRING: /"[ -!#-~]*"/
XML: /<[^)]*/

%ignore /[ \t]+/
"""


class _AstTransformer(Transformer[Token, object]):
    """Builds :mod:`pyippdme.protocol.ast` nodes from a Lark parse tree.

    ``BASICNAME`` and ``EVENT_TAG`` are deliberately *not* transformed at the
    terminal level: the same token text means different things depending on
    where it appears (a bare method name vs. an enum-like argument vs. a tag),
    so each rule method below decides how to wrap it.
    """

    def NUMBER(self, token: Token) -> Number:  # noqa: N802 (Lark terminal callback naming)
        return Number(str(token))

    def STRING(self, token: Token) -> String:  # noqa: N802
        return String(str(token)[1:-1])

    def XML(self, token: Token) -> Xml:  # noqa: N802
        return Xml(str(token))

    def tag(self, children: list[object]) -> object:
        return parse_tag(str(children[0]))

    def method(self, children: list[object]) -> Method:
        name = str(children[0])
        if len(children) == 1:
            return Method(name, ())
        body = children[1]
        if isinstance(body, Xml):
            return Method(name, (), body)
        assert isinstance(body, tuple)
        return Method(name, body)

    def method_body(self, children: list[object]) -> object:
        return children[0]

    def command(self, children: list[object]) -> Command:
        tag, method = children
        assert isinstance(method, Method)
        return Command(tag, method)  # type: ignore[arg-type]

    def ack(self, _children: list[object]) -> object:
        return lambda tag: AckResponse(tag)

    def done(self, _children: list[object]) -> object:
        return lambda tag: DoneResponse(tag)

    def data_resp(self, children: list[object]) -> object:
        data = children[0]
        return lambda tag: DataResponse(tag, data)  # type: ignore[arg-type]

    def error_resp(self, children: list[object]) -> object:
        digit_tok, four_tok, cause, text = children
        assert isinstance(cause, String)
        assert isinstance(text, String)
        return lambda tag: ErrorResponse(
            tag,
            severity=int(str(digit_tok)),
            number=str(four_tok),
            cause=cause.value,
            text=text.value,
        )

    def response(self, children: list[object]) -> Response:
        tag, maker = children
        assert callable(maker)
        return maker(tag)  # type: ignore[no-any-return]

    def numeric_data(self, children: list[object]) -> NumericData:
        return NumericData(tuple(children))  # type: ignore[arg-type]

    def property_data(self, children: list[object]) -> PropertyData:
        first, second = children
        assert isinstance(first, String)
        assert isinstance(second, String)
        return PropertyData(first, second)

    def string_value(self, children: list[object]) -> StringValue:
        value = children[0]
        assert isinstance(value, String)
        return StringValue(value)

    def xml_value(self, children: list[object]) -> Xml:
        value = children[0]
        assert isinstance(value, Xml)
        return value

    def name_value(self, children: list[object]) -> NameValue:
        return NameValue(str(children[0]))

    def items(self, children: list[object]) -> Items:
        return Items(tuple(children))  # type: ignore[arg-type]

    def named_group(self, children: list[object]) -> NamedValue:
        name = str(children[0])
        if len(children) == 1:
            return NamedValue(name)
        body = children[1]
        if isinstance(body, Xml):
            return NamedValue(name, (), body)
        assert isinstance(body, tuple)
        return NamedValue(name, body)

    def data(self, children: list[object]) -> object:
        return children[0]

    def arglist(self, children: list[object]) -> tuple[Argument, ...]:
        return tuple(children)  # type: ignore[arg-type]

    def argument(self, children: list[object]) -> Argument:
        child = children[0]
        if isinstance(child, Token):
            if child.type == "EVENT_TAG":
                return EventTag(str(child))
            if child.type == "BASICNAME":
                return BasicName(str(child))
            raise AssertionError(f"Unexpected token type in argument: {child.type}")
        return child  # type: ignore[return-value]


_LARK = Lark(
    _GRAMMAR,
    start=["command", "response", "method"],
    parser="earley",
    lexer="dynamic",
    maybe_placeholders=False,
)
_TRANSFORMER = _AstTransformer()


def _parse(text: str, start: str) -> object:
    """Parse and transform ``text``, surfacing any failure as :class:`IppDmeProtocolError`.

    Errors raised by the AST dataclasses themselves (e.g. a reserved tag)
    happen inside the transformer, where Lark wraps them in
    :class:`~lark.exceptions.VisitError`; that wrapper is unwrapped here so
    callers only ever see :class:`IppDmeProtocolError`.
    """
    try:
        tree = _LARK.parse(text, start=start)
        return _TRANSFORMER.transform(tree)
    except VisitError as exc:
        if isinstance(exc.orig_exc, IppDmeProtocolError):
            raise exc.orig_exc from exc
        raise IppDmeProtocolError(f"Malformed {start} line {text!r}: {exc}") from exc
    except LarkError as exc:
        raise IppDmeProtocolError(f"Malformed {start} line {text!r}: {exc}") from exc


def parse_command(text: str) -> Command:
    """Parse a full command line (no ``<CR><LF>``): ``(tag|eventTag) SP method``."""
    result = _parse(text, "command")
    assert isinstance(result, Command)
    return result


def parse_response(text: str) -> Response:
    """Parse a full response line (no ``<CR><LF>``): ``(tag|eventTag) SP marker ...``."""
    result = _parse(text, "response")
    assert isinstance(result, AckResponse | DoneResponse | DataResponse | ErrorResponse)
    return result


def parse_method(text: str) -> Method:
    """Parse a bare method call, e.g. ``GoTo(X(10),Y(20))``, without a tag.

    Used by the interactive CLI to accept user-typed command text.
    """
    result = _parse(text, "method")
    assert isinstance(result, Method)
    return result
