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

import asyncio
from collections.abc import Callable, Sequence

from pyippdme.protocol.ast import Items, NamedValue, Number, String, Tag
from pyippdme.protocol.commands import CommandName
from pyippdme.protocol.hooks import LineHook
from pyippdme.protocol.network import TCP_NETWORK, Network
from pyippdme.server import IppDmeServer
from pyippdme.server import builders as server_builders
from pyippdme.server._util import generic_set_prop
from pyippdme.server.backend import MachineBackend
from pyippdme.server.motion import MotionModel, ToolHandler
from pyippdme.server.registry import CommandContext, CommandRegistry, component_name
from pyippdme.server.surface import RawSensor, SampleSurface
from pyippdme.simulation import DEFAULT_COMMAND_CLASSES
from pyippdme.simulation.backend import SimulatedBackend
from pyippdme.simulation.classes.cartcmm_class import pt_meas_fields
from pyippdme.simulation.classes.tool_class import TOOL_CATALOG, collection_root
from pyippdme.simulation.classes.toolchanger_class import activate_tool
from pyippdme.simulation.state import SimulationState
from pyippdme.types.csy import CsyStore, FileCsyStore
from pyippdme.types.toolcollection import find_node
from pyippdme.types.vec3 import Vec3

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
        network: Network = TCP_NETWORK,
        sample_surface: SampleSurface | None = None,
        raw_sensor: RawSensor | None = None,
        motion: MotionModel | None = None,
        tool_handler: ToolHandler | None = None,
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
            network=network,
            sample_surface=sample_surface,
            raw_sensor=raw_sensor,
            motion=motion,
            tool_handler=tool_handler,
            on_line_received=on_line_received,
            on_line_sent=on_line_sent,
            on_connect=on_connect,
            on_disconnect=on_disconnect,
        )

    # -- events the server sends on its own (5.5.3) ----------------------------------

    def _context(self, state: SimulationState) -> CommandContext[SimulationState]:
        return CommandContext(
            tag=Tag.of(1),
            state=state,
            registry=self.registry,
            backend=self.backend,
            csy_store=self.csy_store,
            cancel=asyncio.Event(),
            sample_surface=self.sample_surface,
            raw_sensor=self.raw_sensor,
            motion=self.motion,
            tool_handler=self.tool_handler,
            network=self.network,
        )

    async def key_press(self, key: str) -> bool:
        """Press a key on the virtual jog box; the client gets ``KeyPress(key)``.

        The event is sent only while the user is enabled (5.5.3); returns
        whether it was sent.
        """
        state = self.active_state
        if state is None or not state.mover.user_enabled:
            return False
        return await self.send_event(server_builders.key_press(key))

    async def clearance_point(self, x: float, y: float, z: float) -> bool:
        """Set a clearance point by hand: the machine moves there; the client gets ``GoTo(...)``."""
        return await self._point_event(
            (x, y, z), server_builders.clearance_point, direction=(0.0, 0.0, 0.0)
        )

    async def manual_point(self, x: float, y: float, z: float, ijk: Vec3 = (0.0, 0.0, 0.0)) -> bool:
        """Pick a point by hand: the client gets ``PtMeas(...)`` with the report fields."""
        return await self._point_event((x, y, z), server_builders.manual_point, direction=ijk)

    async def _point_event(
        self,
        position: Vec3,
        build: Callable[..., Items],
        *,
        direction: Vec3,
    ) -> bool:
        state = self.active_state
        if state is None or not state.mover.user_enabled:
            return False
        state.cart_cmm.position = position
        fields = pt_meas_fields(self._context(state), position, direction, CommandName.PT_MEAS)
        values = {f.name: tuple(a.value for a in f.args if isinstance(a, Number)) for f in fields}
        return await self.send_event(
            build(**{name: (v[0] if len(v) == 1 else v) for name, v in values.items()})
        )

    async def change_tool(self, tool_name: str) -> bool:
        """Change the tool at the machine: the client gets ``ChangeTool(ToolName)`` (5.5.3)."""
        state = self.active_state
        if state is None or tool_name not in TOOL_CATALOG:
            return False
        activate_tool(self._context(state), tool_name)
        return await self.send_event(server_builders.tool_changed(tool_name))

    async def data_acquire(
        self,
        acq_name: str,
        acquisition_type: str,
        settings_name: str,
        points: Sequence[tuple[Vec3, Vec3, Vec3]] = (),
    ) -> bool:
        """Take a measurement at the machine: the client gets ``DataAcquire(...)`` (6.15).

        Like the other unsolicited events it is sent only while the user is enabled.
        """
        state = self.active_state
        if state is None or not state.mover.user_enabled:
            return False
        return await self.send_event(
            server_builders.data_acquire_event(acq_name, acquisition_type, settings_name, points)
        )

    async def open_tool_collection(self, path: str) -> bool:
        """Open a tool collection at the machine: the client gets ``OpenToolCollection(path)``."""
        state = self.active_state
        if state is None or find_node(collection_root(), path) is None:
            return False
        return await self.send_event(server_builders.tool_collection_opened(path))

    async def set_property(self, name: str, *values: float | str) -> bool:
        """Change a property at the machine: the client gets ``SetProp(name(values))`` (5.5.3)."""
        state = self.active_state
        if state is None:
            return False
        argument = NamedValue(
            name, tuple(String(v) if isinstance(v, str) else Number.of(v) for v in values)
        )
        ctx = self._context(state)
        if not any(setter(ctx, argument) for setter in self.registry.property_setters()):
            generic_set_prop(ctx, (argument,))
        return await self.send_event(server_builders.property_set(name, *values))
