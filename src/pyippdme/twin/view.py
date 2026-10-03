# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""What a front end needs from a simulation, as a Protocol.

:mod:`pyippdme.gui` shows any object that satisfies :class:`SimulationView`;
:class:`~pyippdme.twin.twin.DigitalTwin` is the one this package ships. To show
your own simulation, implement these members (the protocol seams of
:mod:`pyippdme.server` are what actually answer the client).
"""

from __future__ import annotations

from collections import deque
from typing import Protocol

import numpy as np

from pyippdme.twin.twin import DrawItem, Listener, TwinSnapshot
from pyippdme.types.vec3 import Vec3


class SimulationView(Protocol):
    """What :class:`~pyippdme.gui.viewport.Viewport` draws; see the module docstring."""

    scene_version: int
    contacts: deque[Vec3]
    clouds: deque[np.ndarray]

    def snapshot(self) -> TwinSnapshot: ...
    def draw_items(self) -> list[DrawItem]: ...
    def bounds(self) -> tuple[Vec3, Vec3]: ...
    def add_listener(self, listener: Listener) -> None: ...
