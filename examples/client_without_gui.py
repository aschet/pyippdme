#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Drive a server from a program that owns its own thread, with the helpers the Qt client uses.

``ClientHost`` runs the connection on a background thread, ``recipes`` builds the command lines of
the usual tasks, and ``optical`` reads a sensor and acquires points. Any user interface can be put
on top of the same three.
"""

from __future__ import annotations

from pyippdme.client import optical, recipes
from pyippdme.client.host import ClientHost
from pyippdme.client.interaction import Completed, Failed, Received


def main() -> None:
    host = ClientHost()
    try:
        host.connect_embedded().result(10)  # a virtual CMM in this process; use host.connect(h, p)
        events: list[object] = []

        def show(text: str, event: object) -> None:
            if isinstance(event, Received):
                print(text, "->", event.payload.to_wire())
            elif isinstance(event, Failed):
                print(text, "failed:", event.error.number)
            events.append(event)

        ok = host.run_sequence(
            [
                "StartSession()",
                "Home()",
                "EnableUser()",
                recipes.goto_line(10, 20, 30),
                *recipes.status_lines(),
            ],
            show,
        ).result(10)
        print("all succeeded:", ok, "| completed:", sum(isinstance(e, Completed) for e in events))
        print(host.sensor_info().result(10))
        data = host.acquire(
            acquisition_type="Sweep", points=optical.path_points((0, 0, 0), (10, 0, 0), 3)
        ).result(10)
        print(len(data.points()), "points, box", optical.bounding_box(data.points()))
    finally:
        host.shutdown()


if __name__ == "__main__":
    main()
