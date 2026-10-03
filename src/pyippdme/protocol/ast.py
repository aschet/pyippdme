# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Abstract syntax tree for the I++ DME wire grammar.

The grammar is formally defined in section 5.2 using
Extended Backus-Naur Form. Every node below can render itself back to wire
text via ``to_wire()``, so the AST is the single source of truth for both
parsing (see :mod:`pyippdme.protocol.parser`) and encoding.

Four deliberate deviations from a literal transcription of the grammar are
noted where they occur:

* The XML payload referenced by ``method = basicName '(' (methodArgumentList
  | xml) ')'`` is left undefined by the standard (``xml = @XMLDocumentString@@@``).
  :class:`Xml` stores the raw text verbatim between the outer parentheses.
* The grammar's ``data = numericalData | propertyData | method | propertyList``
  production is ambiguous: a bare ``Name(number, ...)`` matches both
  ``method`` and a one-item ``propertyList``, and there is no rule
  disambiguating a response body that mixes plain numbers with named groups
  (e.g. ``ChangeToolAction(Switch, X(0), Y(0), Z(-8))`` seen in real
  implementations). :mod:`pyippdme.protocol.parser` resolves this by parsing
  any non-numeric, non-property-pair response body as a comma-separated list
  of :class:`NamedValue`/:class:`Argument` nodes (see ``Items`` below), which
  is a strict superset of both ``method`` and ``propertyList``.
* The standard never says how ``xml`` combines with ``data`` or with a
  ``property``/named group's own argument list, yet several real properties
  (e.g. ``Tool.Id()``, ``AdvDataStruct()``) return XML documents. This grammar
  adds ``xml`` as a top-level ``data`` alternative (a bare XML response, like
  :class:`StringValue`/:class:`NameValue` already cover the single-plain-value
  case) and, mirroring how :class:`Method` already resolves the identical
  ambiguity for command arguments, a mutually-exclusive ``xml`` field on
  :class:`NamedValue` for the "one named XML property" case.
* A *command* argument's own ``name(...)`` sub-group (the standard's informal
  ``property`` shorthand within a ``methodArgumentList``) is written as if it
  can only ever hold a flat list of numbers, but real commands need it to
  hold further named sub-groups and bare names too - e.g. ``ScanOnCurve``'s
  ``Format(X(),Y(),Z(),IJK(),tag)`` (Table 85), confirmed against a real
  captured wire line in the NIST/I++ DME reference test suite's
  ``ScanOnCurve.prg``. :mod:`pyippdme.protocol.parser` resolves this by
  giving a command argument's ``name(...)`` group the same fully recursive
  argument grammar as a *response*'s named group already has (they are
  otherwise identical productions), rather than a separate, artificially
  numbers-only one.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import ClassVar

from pyippdme.exceptions import IppDmeProtocolError

_BASIC_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]*$")
_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]*(\.[A-Za-z][A-Za-z0-9]*)*$")
_COMMAND_TAG_RE = re.compile(r"^\d{5}$")
_EVENT_TAG_RE = re.compile(r"^E\d{4}$")
_NUMBER_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([Ee][+-]?\d{1,3})?$")
_STRING_CHAR_RE = re.compile(r"^[ -~]*$")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise IppDmeProtocolError(message)


@dataclass(frozen=True, slots=True)
class Tag:
    """A 5-digit command tag, 00001-99999 (5.4.1). 00000 is reserved."""

    #: Largest tag number :meth:`of` accepts; also what
    #: :meth:`~pyippdme.client.IppDmeClient` cycles back to 1 after when
    #: allocating one, so the two can't drift apart.
    MAX: ClassVar[int] = 99999

    value: str

    def __post_init__(self) -> None:
        _require(_COMMAND_TAG_RE.match(self.value) is not None, f"Invalid tag {self.value!r}")
        _require(self.value != "00000", "Tag 00000 is reserved and must not be used")

    def to_wire(self) -> str:
        return self.value

    def __str__(self) -> str:
        return self.value

    @classmethod
    def of(cls, number: int) -> Tag:
        _require(1 <= number <= cls.MAX, f"Tag number {number} out of range 1-99999")
        return cls(f"{number:05d}")


@dataclass(frozen=True, slots=True)
class EventTag:
    """An event tag ``E0001``-``E9999``. ``E0000`` denotes unsolicited events."""

    #: Largest event tag number :meth:`of` accepts; also what
    #: :meth:`~pyippdme.client.IppDmeClient` cycles back to 1 after when
    #: allocating one (0 itself is never allocated there - it is reserved
    #: for :data:`UNSOLICITED`).
    MAX: ClassVar[int] = 9999

    value: str

    def __post_init__(self) -> None:
        _require(_EVENT_TAG_RE.match(self.value) is not None, f"Invalid event tag {self.value!r}")

    def to_wire(self) -> str:
        return self.value

    def __str__(self) -> str:
        return self.value

    @property
    def is_unsolicited(self) -> bool:
        return self.value == "E0000"

    @classmethod
    def of(cls, number: int) -> EventTag:
        _require(0 <= number <= cls.MAX, f"Event tag number {number} out of range 0-9999")
        return cls(f"E{number:04d}")

    UNSOLICITED: ClassVar[EventTag]  # set below


EventTag.UNSOLICITED = EventTag("E0000")

#: Either kind of tag that can prefix a command or response line.
TagLike = Tag | EventTag


def parse_tag(text: str) -> TagLike:
    """Parse either a :class:`Tag` or an :class:`EventTag` from its wire text."""
    if text.startswith("E"):
        return EventTag(text)
    return Tag(text)


@dataclass(frozen=True, slots=True)
class Number:
    """A numeric literal, at most 16 characters (5.3.1)."""

    text: str

    def __post_init__(self) -> None:
        _require(len(self.text) <= 16, f"Number {self.text!r} exceeds 16 characters")
        _require(_NUMBER_RE.match(self.text) is not None, f"Invalid number {self.text!r}")

    def to_wire(self) -> str:
        return self.text

    @property
    def value(self) -> float:
        return float(self.text)

    @classmethod
    def of(cls, value: float | int) -> Number:
        """Format ``value`` as wire text, guaranteed to fit the 16-character limit.

        Prefers Python's shortest round-trip ``repr()``, which is exact and
        usually well under the limit, but falls back to reduced-precision
        ``%g`` formatting for values (e.g. irrational trig results) whose
        exact repr would exceed it.
        """
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            raise IppDmeProtocolError(f"{value!r} cannot be represented as an I++ DME number")
        text = repr(float(value)) if isinstance(value, float) else str(value)
        if text.endswith(".0"):
            text = text[:-2]
        if len(text) <= 16:
            return cls(text)
        for precision in range(15, 0, -1):
            candidate = f"{value:.{precision}g}"
            if len(candidate) <= 16:
                return cls(candidate)
        raise IppDmeProtocolError(f"{value!r} cannot be represented within 16 characters")


@dataclass(frozen=True, slots=True)
class String:
    """A quoted string. Wire text may not itself contain a ``"`` character."""

    value: str

    def __post_init__(self) -> None:
        _require(
            _STRING_CHAR_RE.match(self.value) is not None, f"Invalid string content {self.value!r}"
        )
        _require('"' not in self.value, "Strings must not contain '\"' (no escaping is defined)")

    def to_wire(self) -> str:
        return f'"{self.value}"'


@dataclass(frozen=True, slots=True)
class BasicName:
    """A bare identifier used as an unquoted enum/name argument, e.g. ``Switch``."""

    value: str

    def __post_init__(self) -> None:
        _require(_BASIC_NAME_RE.match(self.value) is not None, f"Invalid name {self.value!r}")

    def to_wire(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class Xml:
    """Raw XML payload of a method, verbatim between its outer parentheses.

    The standard does not define this production (see module docstring), so
    no attempt is made to parse or validate the XML content itself.
    """

    raw: str

    def to_wire(self) -> str:
        return self.raw


@dataclass(frozen=True, slots=True)
class NamedValue:
    """A dotted name applied to a parenthesized argument list, or an XML payload.

    Covers both the ``property`` production (``name '(' numbers ')'``, used
    e.g. as a command argument like ``X(5)``) and nested named groups that
    appear inside response data (e.g. ``ChangeToolAction(Switch, X(0))``).

    ``xml`` mirrors :class:`Method`: a handful of properties are themselves
    XML documents (e.g. ``Tool.Id()``, 6.10.3; ``Tool.AdvDataStruct()``,
    6.17.2) - the standard's own grammar does not define how ``xml``
    combines with the rest of the ``data``/``property`` productions, so
    (matching how :class:`Method` already resolves the identical ambiguity
    for command arguments) it is a separate, mutually-exclusive field rather
    than a member of :data:`Argument`, which the ``XML`` token's own
    "consume to the closing paren" lexing does not allow to safely sit
    alongside sibling arguments anyway.
    """

    name: str
    args: tuple[Argument, ...] = ()
    xml: Xml | None = None

    def __post_init__(self) -> None:
        _require(_NAME_RE.match(self.name) is not None, f"Invalid name {self.name!r}")
        _require(
            self.xml is None or not self.args,
            "A named value cannot carry both an argument list and an XML payload",
        )

    def to_wire(self) -> str:
        body = (
            self.xml.to_wire()
            if self.xml is not None
            else ",".join(arg.to_wire() for arg in self.args)
        )
        return f"{self.name}({body})"


#: Any value that may appear as a method/property argument.
Argument = Number | String | NamedValue | EventTag | BasicName


@dataclass(frozen=True, slots=True)
class Method:
    """A method call: a name applied to an argument list or an XML payload."""

    name: str
    args: tuple[Argument, ...] = ()
    xml: Xml | None = None

    def __post_init__(self) -> None:
        _require(_BASIC_NAME_RE.match(self.name) is not None, f"Invalid method name {self.name!r}")
        _require(
            self.xml is None or not self.args,
            "A method cannot carry both an argument list and an XML payload",
        )

    def to_wire(self) -> str:
        body = (
            self.xml.to_wire() if self.xml is not None else ",".join(a.to_wire() for a in self.args)
        )
        return f"{self.name}({body})"


@dataclass(frozen=True, slots=True)
class Command:
    """A command line sent by the client: ``tag SP method CRLF``."""

    tag: TagLike
    method: Method

    def to_wire(self) -> str:
        return f"{self.tag.to_wire()} {self.method.to_wire()}"


@dataclass(frozen=True, slots=True)
class AckResponse:
    """Acknowledgement response: ``tag SP '&'``."""

    tag: TagLike

    def to_wire(self) -> str:
        return f"{self.tag.to_wire()} &"


@dataclass(frozen=True, slots=True)
class DoneResponse:
    """Transaction-complete response: ``tag SP '%'``."""

    tag: TagLike

    def to_wire(self) -> str:
        return f"{self.tag.to_wire()} %"


@dataclass(frozen=True, slots=True)
class NumericData:
    """A comma-separated list of bare numbers (``numericalData``)."""

    values: tuple[Number, ...]

    def to_wire(self) -> str:
        return ",".join(v.to_wire() for v in self.values)


@dataclass(frozen=True, slots=True)
class PropertyData:
    """Exactly two strings (``propertyData``), e.g. a property name/type pair."""

    first: String
    second: String

    def to_wire(self) -> str:
        return f"{self.first.to_wire()},{self.second.to_wire()}"


@dataclass(frozen=True, slots=True)
class NameValue:
    """A single bare (unquoted) ``name``-typed value as a whole data response.

    Section 5.3.5 defines ``name`` as "a special case of enum ... in case
    of returned data, it refers to a string, but without quotation marks",
    e.g. the ``Csy`` returned by ``GetCoordSystem()``. Like
    :class:`StringValue`, this has no direct alternative in the formal
    ``data`` production.
    """

    value: str

    def __post_init__(self) -> None:
        _require(_NAME_RE.match(self.value) is not None, f"Invalid name {self.value!r}")

    def to_wire(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class StringValue:
    """A single bare (unwrapped, "Kind U") string as a whole data response.

    Not a named production of the formal grammar: several commands (e.g.
    ``GetSupportedCommands()``, ``EnumNameSpaces()``) define a single
    unnamed string return value sent as "a separate message" per item, which
    the ``data`` production (``numericalData | propertyData | method |
    propertyList``) has no direct alternative for. This is the pragmatic
    reading used throughout the reference command classes in this project.
    """

    value: String

    def to_wire(self) -> str:
        return self.value.to_wire()


@dataclass(frozen=True, slots=True)
class Items:
    """A comma-separated list of named values, covering ``method``/``propertyList``.

    See the module docstring for why this generalization is needed.
    """

    values: tuple[NamedValue, ...]

    def to_wire(self) -> str:
        return ",".join(v.to_wire() for v in self.values)


#: The payload of a data response.
#: ``Xml`` covers commands whose sole returned value is itself an XML
#: document (e.g. ``AdvDataStruct()``, 6.17.2) - the standard's grammar
#: does not define this either (see the module docstring), so it is added
#: as a bare top-level alternative, mirroring how :class:`StringValue`/
#: :class:`NameValue` already cover the single-plain-value case.
DataPayload = NumericData | PropertyData | Items | StringValue | NameValue | Xml


@dataclass(frozen=True, slots=True)
class DataResponse:
    """Data response: ``tag SP '#' SP data``.

    The standard is internally inconsistent here. Its formal EBNF
    (5.2: ``dataResponse = (tag | eventTag) ' #' data``) puts no space
    between ``#`` and the payload, but its own prose two pages later (5.4.2:
    "If the response line contains more content, the eighth character must
    again be a space") requires exactly that space, and every real-world
    example in the standard follows the prose, not the EBNF. ``to_wire()``
    always emits it; :mod:`~pyippdme.protocol.parser` accepts either form on
    input.
    """

    tag: TagLike
    data: DataPayload

    def to_wire(self) -> str:
        return f"{self.tag.to_wire()} # {self.data.to_wire()}"


@dataclass(frozen=True, slots=True)
class ErrorResponse:
    """Error response: ``tag SP '!' SP 'Error(' F1 ',' F2 ',' F3 ',' Text ')'``."""

    tag: TagLike
    severity: int
    number: str
    cause: str
    text: str

    def __post_init__(self) -> None:
        _require(0 <= self.severity <= 9, f"Invalid severity {self.severity}")
        _require(
            re.match(r"^\d{4}$", self.number) is not None, f"Invalid error number {self.number!r}"
        )

    def to_wire(self) -> str:
        return (
            f"{self.tag.to_wire()} ! Error({self.severity},{self.number},"
            f'"{self.cause}","{self.text}")'
        )


#: Any response line the server may send.
Response = AckResponse | DoneResponse | DataResponse | ErrorResponse
