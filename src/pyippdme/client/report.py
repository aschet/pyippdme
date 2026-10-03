# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""The decoded form of a response made of named values such as ``X(1),Y(2),IJK(0,0,1)``."""

from __future__ import annotations

from collections.abc import Iterator, Mapping

from pyippdme.protocol.ast import BasicName, DataPayload, Items, NamedValue, Number, String

#: ``None`` stands for the ``NULL`` a server reports for information it cannot deliver (6.10.2).
ReportValue = float | str | tuple[float, ...] | None


class Report(Mapping[str, ReportValue]):
    """Named values by name: a single number is a ``float``, several numbers are a tuple.

    A report compares equal to a plain ``dict`` with the same content::

        report = await machine.cart_cmm.pt_meas(x=1, y=2, z=3, ijk=(0, 0, 1))
        report.number("X")     # 1.0
        report.vector("IJK")   # (0.0, 0.0, 1.0)
    """

    def __init__(self, values: Mapping[str, ReportValue]) -> None:
        self._values = dict(values)

    def __getitem__(self, name: str) -> ReportValue:
        return self._values[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __repr__(self) -> str:
        return f"Report({self._values!r})"

    def number(self, name: str) -> float:
        """Return ``name`` as a single number."""
        value = self._values[name]
        if isinstance(value, float):
            return value
        raise TypeError(f"{name!r} is not a single number: {value!r}")

    def vector(self, name: str) -> tuple[float, ...]:
        """Return ``name`` as a tuple of numbers, also when it holds only one."""
        value = self._values[name]
        if isinstance(value, float):
            return (value,)
        if isinstance(value, tuple):
            return value
        raise TypeError(f"{name!r} is not numeric: {value!r}")

    def text(self, name: str) -> str:
        """Return ``name`` as a string."""
        value = self._values[name]
        if isinstance(value, str):
            return value
        raise TypeError(f"{name!r} is not a string: {value!r}")


def _value(named: NamedValue) -> ReportValue:
    args = named.args
    if all(isinstance(arg, Number) for arg in args):
        numbers = tuple(arg.value for arg in args if isinstance(arg, Number))
        return numbers[0] if len(numbers) == 1 else numbers
    if len(args) == 1 and isinstance(args[0], String):
        return args[0].value
    if len(args) == 1 and isinstance(args[0], BasicName) and args[0].value == "NULL":
        return None
    raise TypeError(f"Expected numbers or one string for {named.name!r}")


def report_from_payload(payload: DataPayload) -> Report:
    """Decode an ``Items`` response into a :class:`Report`."""
    if not isinstance(payload, Items):
        raise TypeError(f"Expected an Items response, got {type(payload).__name__}")
    return Report({named.name: _value(named) for named in payload.values})
