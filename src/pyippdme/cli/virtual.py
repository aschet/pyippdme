# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""The simulated machine behind ``serve`` and ``--virtual``: the minimal one, or the digital twin.

``ippdme serve``, ``ippdme client --virtual`` and ``ippdme tui --virtual`` take the same options
(:func:`add_arguments`) and build the same server (:func:`build_server`). Without options it is
the minimal :class:`~pyippdme.simulation.virtual_cmm.VirtualCMM`. ``--surface`` gives that one a
simple analytic sample, so ``PtMeas`` has something to touch. ``--twin`` (or any option that needs
it, like ``--sample``) switches to :class:`~pyippdme.twin.twin.DigitalTwin`: a machine with
limits, timed moves, collisions, tools, probe errors, CAD samples and optical sensors; it needs
``pip install pyippdme[twin]``.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pyippdme.protocol.network import TCP_NETWORK, Network
from pyippdme.server import IppDmeServer
from pyippdme.server.registry import CommandRegistry
from pyippdme.simulation import DEFAULT_COMMAND_CLASSES
from pyippdme.simulation.surface import CompositeSurface, parse_surface
from pyippdme.simulation.virtual_cmm import VirtualCMM
from pyippdme.types.csy import CsyStore, InMemoryCsyStore

ServerFactory = Callable[[Network], IppDmeServer[Any]]


@dataclass(slots=True)
class SimulationOptions:
    """What the command line asked of the simulated machine."""

    twin: bool = False
    machine: str | None = None
    preset: str | None = None
    rotary: bool = False
    samples: list[str] = field(default_factory=list)
    artifact: bool = False
    reference_sphere: bool = False
    time_scale: float = 1.0
    no_noise: bool = False
    seed: int | None = None
    surfaces: list[str] = field(default_factory=list)

    @property
    def uses_twin(self) -> bool:
        return bool(
            self.twin
            or self.machine
            or self.preset
            or self.rotary
            or self.samples
            or self.artifact
            or self.reference_sphere
        )


def add_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group(
        "simulated machine",
        "the minimal simulation by default; the digital twin (needs pyippdme[twin]) with --twin "
        "or any of the twin options",
    )
    group.add_argument("--twin", action="store_true", help="use the digital twin")
    group.add_argument(
        "--machine", help="twin: machine directory with a machine.toml, or a STEP assembly"
    )
    group.add_argument("--preset", help="twin: machine preset (see pyippdme.twin.spec.PRESETS)")
    group.add_argument("--rotary", action="store_true", help="twin: add a rotary table")
    group.add_argument(
        "--sample",
        action="append",
        default=[],
        dest="samples",
        metavar="FILE",
        help="twin: CAD file (STEP, IGES, STL, BREP) to put on the table; repeat for fixtures",
    )
    group.add_argument(
        "--check-artifact",
        action="store_true",
        dest="artifact",
        help="twin: put the check artefact on the table",
    )
    group.add_argument(
        "--reference-sphere",
        action="store_true",
        help="twin: put a qualification sphere on the table (tools need ReQualify on it)",
    )
    group.add_argument(
        "--time-scale",
        type=float,
        default=1.0,
        help="twin: playback speed of moves, 1 = real time, 0 = instant (default 1)",
    )
    group.add_argument("--no-noise", action="store_true", help="twin: no measuring errors")
    group.add_argument("--seed", type=int, help="twin: seed of the measuring errors")
    group.add_argument(
        "--surface",
        action="append",
        default=[],
        dest="surfaces",
        metavar="SPEC",
        help="minimal simulation: an analytic sample for PtMeas, plane:P:N, sphere:C:R or "
        "cylinder:P:AXIS:R with points as x,y,z (repeatable)",
    )


def options_from_args(args: argparse.Namespace) -> SimulationOptions:
    return SimulationOptions(
        twin=getattr(args, "twin", False),
        machine=getattr(args, "machine", None),
        preset=getattr(args, "preset", None),
        rotary=getattr(args, "rotary", False),
        samples=list(getattr(args, "samples", [])),
        artifact=getattr(args, "artifact", False),
        reference_sphere=getattr(args, "reference_sphere", False),
        time_scale=getattr(args, "time_scale", 1.0),
        no_noise=getattr(args, "no_noise", False),
        seed=getattr(args, "seed", None),
        surfaces=list(getattr(args, "surfaces", [])),
    )


def build_server(
    options: SimulationOptions,
    *,
    network: Network = TCP_NETWORK,
    csy_store: CsyStore | None = None,
    command_classes: Sequence[Callable[[CommandRegistry], None]] = DEFAULT_COMMAND_CLASSES,
    on_line_received: Callable[[str], None] | None = None,
    on_line_sent: Callable[[str], None] | None = None,
    on_connect: Callable[[str], None] | None = None,
    on_disconnect: Callable[[str], None] | None = None,
) -> IppDmeServer[Any]:
    """Build the simulated machine the options describe."""
    hooks = {
        "on_line_received": on_line_received,
        "on_line_sent": on_line_sent,
        "on_connect": on_connect,
        "on_disconnect": on_disconnect,
    }
    if options.uses_twin:
        return _build_twin(options, network, csy_store, command_classes, hooks)
    if options.surfaces:
        try:
            surfaces = [parse_surface(s) for s in options.surfaces]
        except ValueError as exc:
            raise SystemExit(str(exc)) from None
        surface = surfaces[0] if len(surfaces) == 1 else CompositeSurface(surfaces)
    else:
        surface = None
    return VirtualCMM(
        csy_store=csy_store,
        command_classes=command_classes,
        network=network,
        sample_surface=surface,
        **hooks,  # type: ignore[arg-type]
    )


def _build_twin(
    options: SimulationOptions,
    network: Network,
    csy_store: CsyStore | None,
    command_classes: Sequence[Callable[[CommandRegistry], None]],
    hooks: dict[str, Any],
) -> IppDmeServer[Any]:
    try:
        from pyippdme.twin import DigitalTwin, MachineModel
        from pyippdme.twin.artifact import build_check_artifact, build_reference_sphere
        from pyippdme.twin.cad import CadError
        from pyippdme.twin.objects import SceneObject
        from pyippdme.twin.spec import PRESETS
    except ImportError as exc:
        raise SystemExit("The digital twin needs OpenCASCADE: pip install pyippdme[twin]") from exc
    try:
        if options.machine:
            path = Path(options.machine)
            machine = (
                MachineModel.from_directory(path) if path.is_dir() else MachineModel.from_step(path)
            )
        else:
            preset = options.preset or "bridge-700"
            if preset not in PRESETS:
                raise SystemExit(f"unknown preset {preset!r}; choose from {', '.join(PRESETS)}")
            machine = MachineModel.default(preset, rotary=options.rotary)
        twin = DigitalTwin(machine, seed=options.seed, time_scale=options.time_scale)
        twin.noise_enabled = not options.no_noise
        if options.artifact:
            twin.place_sample(build_check_artifact())
        for i, sample in enumerate(options.samples):
            if i == 0 and not options.artifact:
                twin.load_sample(sample)
            else:
                twin.place_sample(SceneObject.from_file(sample, "fixture"), replace=False)
        if options.reference_sphere:
            twin.add_object(_place_reference(twin, build_reference_sphere()))
    except (CadError, OSError, ValueError, KeyError) as exc:
        raise SystemExit(f"cannot set up the digital twin: {exc}") from None
    server = twin.create_server(
        csy_store=csy_store, network=network, command_classes=command_classes, **hooks
    )
    server.twin = twin  # type: ignore[attr-defined]
    return server


def _place_reference(twin: Any, sphere: Any) -> Any:
    """Put the qualification sphere near the front left corner of the volume."""
    from pyippdme.twin import geometry

    spec = twin.machine.spec
    sphere.pose = geometry.translation(
        spec.travel[0] * 0.15, spec.travel[1] * 0.2, spec.table_top_z
    )
    return sphere


_factory: ServerFactory | None = None


def configure(options: SimulationOptions) -> None:
    """Make :func:`create_embedded` build the machine ``options`` describe."""
    global _factory
    _factory = lambda network: build_server(options, network=network)  # noqa: E731


def create_embedded(network: Network) -> IppDmeServer[Any]:
    """Create the in-process machine of ``--virtual``: as configured, or the minimal one."""
    if _factory is not None:
        return _factory(network)
    # A try-out machine does not leave saved coordinate systems in the user's home directory.
    return VirtualCMM(network=network, csy_store=InMemoryCsyStore())
