# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""The strip above the 3D view: what the machine is doing, at a glance."""

from __future__ import annotations

from PySide6.QtWidgets import QHBoxLayout, QLabel, QWidget

from pyippdme.gui.widgets import Led, mono, separator
from pyippdme.twin.twin import TwinSnapshot

__all__ = ["StatusStrip"]


class StatusStrip(QWidget):
    """Lamps for server, client, homing, motion and safety; the tool; the position; an error."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        self.server = Led("Server stopped", "The I++ DME server of this window")
        self.client = Led("No client", "The program connected to the server")
        self.homed = Led("Not homed", "Home() has to run before the machine moves")
        self.motion = Led("Idle", "Whether the machine is moving")
        self.safety = Led("Safe", "Emergency stop and air supply")
        self.tool = QLabel("")
        self.position = QLabel("")
        mono(self.position, 10)
        self.banner = QLabel("")
        self.banner.setStyleSheet("color: #e74c3c; font-weight: bold;")
        for widget in (self.server, self.client, separator(), self.homed, self.motion, self.safety):
            layout.addWidget(widget)
        layout.addWidget(separator())
        layout.addWidget(self.tool)
        layout.addStretch(1)
        layout.addWidget(self.banner)
        layout.addWidget(self.position)

    def set_server(self, running: bool, address: str = "") -> None:
        if running:
            self.server.set_state("on", f"Listening on {address}")
        else:
            self.server.set_state("off", "Server stopped")

    def update_from(self, snap: TwinSnapshot) -> None:
        """Show the machine state of one moment."""
        if snap.peer:
            self.client.set_state("on", snap.peer)
        else:
            self.client.set_state("off", "No client")
        self.homed.set_state("on" if snap.homed else "off", "Homed" if snap.homed else "Not homed")
        if snap.changing_tool:
            self.motion.set_state("on", f"Changing to {snap.changing_tool}")
        else:
            self.motion.set_state(
                "on" if snap.moving else "off", "Moving" if snap.moving else "Idle"
            )
        if snap.estop:
            self.safety.set_state("error", "EMERGENCY STOP")
        elif not snap.air_ok:
            self.safety.set_state("error", "No air")
        else:
            self.safety.set_state("on", "Safe")
        qualified = "qualified" if snap.qualified else "not qualified"
        detached = " - stylus broke away" if snap.detached else ""
        self.tool.setText(f"Tool {snap.tool_name} ({snap.tool_mode}), {qualified}{detached}")
        self.banner.setText(snap.error)
        x, y, z = snap.position
        self.position.setText(f"X {x:9.3f}  Y {y:9.3f}  Z {z:9.3f}  R {snap.rotary:7.2f}")
