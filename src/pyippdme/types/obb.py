# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Oriented bounding boxes, as ``Tool.CollisionVolume()`` reports them (6.20.2, Figures 46-51).

The standard's example (Figure 50) answers::

    Tool.CollisionVolume(OBB, 100.0, 30.0, 40.0, 500.0, 15.0, 20.0,
                              0.8944, 0.2683, 0.3578, -0.2873, 0.9578, 0.0, -0.3427, -0.1027, 0.9337,
                         OBB, 150.0, 10.0, 530.0, 50.0, 50.0, 500.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, ...)

Each box is the token ``OBB`` and 15 numbers (Figure 46): the vector ``C`` to the centre of the box
from the active tool, the extensions ``E`` of the box, and the three orthogonal unit vectors
``I``, ``J``, ``K`` that give its orientation, all in the coordinate system selected with
``SetCoordSystem``/``SetCsyTransformation``. The rotated tool of Figure 49 gets boxes that are
rotated with it.

The text does not say whether an extension is the full length or half of it. Figure 48's box
(``E`` = 90, 95, 30 around the centre 85, 10, 20) only covers the whole star stylus that the
figure draws if the extensions are half-lengths, so this package reads them that way, the usual
representation of an oriented box. Pure Python.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pyippdme.types.vec3 import Vec3

#: The token that starts each box in the answer.
OBB_TOKEN = "OBB"
#: Numbers per box after the token: ``C``, ``E``, ``I``, ``J``, ``K``.
OBB_NUMBERS = 15


@dataclass(frozen=True, slots=True)
class Obb:
    """A box with its own axes."""

    #: ``C``: the vector to the centre of the box from the active tool.
    center: Vec3
    #: ``E``: the extensions of the box along its axes, as half-lengths.
    extent: Vec3
    #: ``I``, ``J``, ``K``: the box's axes as orthogonal unit vectors (the rows of the matrix).
    axes: tuple[Vec3, Vec3, Vec3]

    def numbers(self) -> tuple[float, ...]:
        """The 15 numbers of the answer: ``C``, ``E``, ``I``, ``J``, ``K``."""
        return (*self.center, *self.extent, *self.axes[0], *self.axes[1], *self.axes[2])

    @classmethod
    def from_numbers(cls, values: tuple[float, ...]) -> Obb:
        if len(values) != OBB_NUMBERS:
            raise ValueError(f"an OBB has {OBB_NUMBERS} numbers, got {len(values)}")
        v = values
        return cls(
            (v[0], v[1], v[2]),
            (v[3], v[4], v[5]),
            ((v[6], v[7], v[8]), (v[9], v[10], v[11]), (v[12], v[13], v[14])),
        )

    def mapped(self, direction: Callable[[Vec3], Vec3]) -> Obb:
        """The box with its centre vector and axes turned by ``direction`` (a rotation)."""
        i, j, k = (direction(a) for a in self.axes)
        return Obb(direction(self.center), self.extent, (i, j, k))
