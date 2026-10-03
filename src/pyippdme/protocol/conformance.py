# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""The five conformance classes of Annex J.1 (deprecated).

A conformance class names the object-model components a typical machine has.
The standard deprecated the concept: a server's capabilities are found with
``GetSupportedCommands()`` and ``GetSupportedArguments(..)`` instead (4.3, Annex K).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ConformanceClass:
    """One conformance class: the classes of the machine's components (Annex J.1)."""

    description: str
    tool: str
    align_mode: str
    mover: str | None = None
    rotary_table: str | None = None
    cart_cmm: str = "CartCMM"
    tool_changer: str = "ToolChanger"

    @property
    def machine_class(self) -> str:
        """The ``GetMachineClass()`` string of a machine of this class (6.4.1)."""
        parts = (
            self.cart_cmm,
            self.tool,
            self.align_mode,
            self.mover,
            self.tool_changer,
            None,  # Part
            self.rotary_table,
        )
        return "_".join(part or "None" for part in parts)


CONFORMANCE_CLASSES: tuple[ConformanceClass, ...] = (
    ConformanceClass("Manual machine with tactile probe", "TouchTrigger", "Fixed"),
    ConformanceClass("CNC machine with tactile probe", "TouchTrigger", "Fixed", "Cartesian"),
    ConformanceClass(
        "CNC machine with tactile probe and CAA Head", "TouchTrigger", "Alignable_AB", "Cartesian"
    ),
    ConformanceClass("CNC machine with scanning probe", "Scanning", "Fixed", "Cartesian"),
    ConformanceClass(
        "CNC machine with scanning probe and rotary table",
        "Scanning",
        "Fixed",
        "Cartesian",
        "RotaryTable",
    ),
)
