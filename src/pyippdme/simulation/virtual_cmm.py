# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""A ready-to-run, fully simulated CMM for client development (no hardware required).

Real dimensional-metrology hardware is expensive, so a client implementer
needs something to develop and test against. :class:`VirtualCMM` is exactly
:class:`~pyippdme.server.IppDmeServer` with every built-in command class
enabled and sensible defaults for standalone use:

* ``machine_class`` reports the ``_VirtualCMM`` suffix 6.4.1
  reserves for exactly this case ("specifies whether the server provides a
  virtual CMM and not a real one"), and - unlike
  :class:`~pyippdme.server.IppDmeServer`'s fixed default - is derived from
  ``command_classes`` (see :func:`default_machine_class`), since a
  simulated machine's configuration is fully known from what was actually
  registered, unlike a real backend's.
* ``csy_store`` defaults to :class:`~pyippdme.types.csy.FileCsyStore`, so named
  coordinate systems saved during one run are still there the next time -
  matching 6.5.2's "must be stored persistently" requirement, unlike
  :class:`~pyippdme.server.IppDmeServer`'s own in-memory default (chosen
  there so a plain ``IppDmeServer()`` never touches disk unasked).

Everything else is inherited unchanged; pass ``backend=`` for custom
scanning geometry, or ``command_classes=`` to still trim which classes are
active, exactly as on :class:`~pyippdme.server.IppDmeServer`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from pyippdme.protocol.hooks import LineHook
from pyippdme.server import IppDmeServer
from pyippdme.server.backend import MachineBackend
from pyippdme.server.registry import CommandRegistry, component_name
from pyippdme.server.surface import SampleSurface
from pyippdme.simulation import DEFAULT_COMMAND_CLASSES
from pyippdme.simulation.backend import SimulatedBackend
from pyippdme.simulation.state import SimulationState
from pyippdme.types.csy import CsyStore, FileCsyStore

#: 6.4.1's naming scheme's four purely presence-based top-level objects:
#: each becomes its class name if the matching ``--components``/
#: ``command_classes`` entry is present, else "None". The Server (main
#: geometry) and Tool segments need their own logic below - this library
#: implements more than one possible value for each (CartCMM is currently
#: the only geometry class, but Tool can be TouchTrigger *or* Scanning,
#: 4.2 - "None" if this library's tool/scanning classes are unavailable
#: too). Alignment stays fixed at "Fixed": there's no other Alignment
#: class implemented here at all, so nothing to vary it against.
_OPTIONAL_MACHINE_CLASS_COMPONENTS: tuple[tuple[str, str], ...] = (
    ("mover", "Cartesian"),
    ("toolchanger", "ToolChanger"),
    ("part", "Part"),
    ("rotarytable", "RotaryTable"),
)


def default_machine_class(
    command_classes: Sequence[Callable[[CommandRegistry], None]],
) -> str:
    """Derive a 6.4.1 machine-class string from which ``command_classes`` are registered.

    Server becomes "CartCMM" if ``cartcmm`` is present, else "None". Tool
    becomes "Scanning" if ``scanning`` is present (4.2: a strictly more
    specific claim than plain single-point measurement), else "TouchTrigger"
    if ``tool`` is present, else "None" - note this can only report one of
    the two even when a machine implements both; 6.4.1's own answer for that
    ("if a server implements more than one machine class, multiple strings
    are sent") would need ``GetMachineClass()`` itself to return more than
    one value, which it doesn't here. Alignment stays fixed at "Fixed".
    Each of Mover/ToolChanger/Part/RotaryTable becomes its class name if
    present, else "None" (6.4.1), followed by the ``VirtualCMM`` suffix.
    Passing the full :data:`~pyippdme.simulation.DEFAULT_COMMAND_CLASSES`
    reproduces the same string :class:`VirtualCMM` has always defaulted to.
    """
    names = {component_name(fn) for fn in command_classes}
    server_segment = "CartCMM" if "cartcmm" in names else "None"
    if "scanning" in names:
        tool_segment = "Scanning"
    elif "tool" in names:
        tool_segment = "TouchTrigger"
    else:
        tool_segment = "None"
    optional_segments = [
        value if name in names else "None" for name, value in _OPTIONAL_MACHINE_CLASS_COMPONENTS
    ]
    return "_".join([server_segment, tool_segment, "Fixed", *optional_segments, "VirtualCMM"])


#: :func:`default_machine_class` for the full :data:`~pyippdme.simulation.DEFAULT_COMMAND_CLASSES`
#: set - :class:`VirtualCMM`'s ``machine_class`` when every built-in class is active.
VIRTUAL_CMM_MACHINE_CLASS = default_machine_class(DEFAULT_COMMAND_CLASSES)


class VirtualCMM(IppDmeServer[SimulationState]):
    """A fully simulated CMM, ready to ``serve_forever()`` with no further setup.

    ::

        import asyncio
        from pyippdme.simulation.virtual_cmm import VirtualCMM

        asyncio.run(VirtualCMM().serve_forever(host="127.0.0.1"))
    """

    def __init__(
        self,
        *,
        backend: MachineBackend | None = None,
        csy_store: CsyStore | None = None,
        machine_class: str | None = None,
        command_classes: Sequence[Callable[[CommandRegistry], None]] = DEFAULT_COMMAND_CLASSES,
        sample_surface: SampleSurface | None = None,
        on_line_received: LineHook | None = None,
        on_line_sent: LineHook | None = None,
        on_connect: Callable[[str], None] | None = None,
        on_disconnect: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__(
            backend=backend if backend is not None else SimulatedBackend(),
            csy_store=csy_store if csy_store is not None else FileCsyStore(),
            machine_class=(
                machine_class
                if machine_class is not None
                else default_machine_class(command_classes)
            ),
            command_classes=command_classes,
            state_factory=SimulationState,
            sample_surface=sample_surface,
            on_line_received=on_line_received,
            on_line_sent=on_line_sent,
            on_connect=on_connect,
            on_disconnect=on_disconnect,
        )
