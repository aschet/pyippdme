# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""The simulated machine of ``serve`` and ``--virtual``: minimal with a sample, or the twin."""

from __future__ import annotations

import argparse

import pytest

from pyippdme import IppDmeMachine
from pyippdme.cli import virtual
from pyippdme.protocol.network import MemoryNetwork
from pyippdme.simulation.surface import CompositeSurface, parse_surface


def _args(*argv: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    virtual.add_arguments(parser)
    return parser.parse_args(list(argv))


async def _connect(server: object) -> tuple[IppDmeMachine, MemoryNetwork]:
    network = server.network  # type: ignore[attr-defined]
    port = await server.start()  # type: ignore[attr-defined]
    machine = await IppDmeMachine.connect("127.0.0.1", port, network=network)
    await machine.start_session()
    return machine, network


async def test_the_minimal_simulation_gets_an_analytic_sample_from_the_command_line() -> None:
    options = virtual.options_from_args(_args("--surface", "plane:0,0,5:0,0,1"))
    assert not options.uses_twin
    server = virtual.build_server(options, network=MemoryNetwork())
    machine, _ = await _connect(server)
    report = await machine.cart_cmm.pt_meas(x=1, y=2, z=5, ijk=(0, 0, 1))
    assert report.number("Z") == pytest.approx(5.0)
    await machine.close()
    await server.close()


def test_surfaces_are_parsed_and_combined() -> None:
    assert isinstance(parse_surface("sphere:1,2,3:4"), type(parse_surface("sphere:0,0,0:1")))
    plane = parse_surface("plane:0,0,0:0,0,1")
    sphere = parse_surface("sphere:0,0,10:2")
    both = CompositeSurface([plane, sphere])
    assert both.intersect((0.0, 0.0, 50.0), (0.0, 0.0, -1.0)) == pytest.approx((0.0, 0.0, 12.0))
    for bad in ("cube:1", "plane:0,0:0,0,1", "sphere:a,b,c:1"):
        with pytest.raises(ValueError, match="surface"):
            parse_surface(bad)
    with pytest.raises(SystemExit):
        virtual.build_server(virtual.SimulationOptions(surfaces=["nonsense"]))


def test_any_twin_option_selects_the_twin() -> None:
    assert virtual.options_from_args(_args("--check-artifact")).uses_twin
    assert virtual.options_from_args(_args("--sample", "a.step")).uses_twin
    assert virtual.options_from_args(_args("--twin")).uses_twin
    assert not virtual.options_from_args(_args("--no-noise")).uses_twin


async def test_the_twin_from_the_command_line_has_the_artifact_on_its_table() -> None:
    pytest.importorskip("OCP")
    options = virtual.options_from_args(
        _args("--check-artifact", "--time-scale", "0", "--no-noise", "--seed", "1")
    )
    server = virtual.build_server(options, network=MemoryNetwork())
    assert server.twin.artifact() is not None  # type: ignore[attr-defined]
    machine, _ = await _connect(server)
    await machine.dme.home()
    # The plate of the artefact stands on the table at z = -10; its top face is 25 mm higher.
    await machine.cart_cmm.go_to(z=200)  # lift first: a straight move would cut through the plate
    await machine.cart_cmm.go_to(x=440, y=320)
    await machine.cart_cmm.go_to(z=40)
    report = await machine.cart_cmm.pt_meas(x=440, y=320, z=15, ijk=(0, 0, 1))
    assert report.number("Z") == pytest.approx(15.0, abs=1e-6)
    await machine.close()
    await server.close()
    server.twin.toolkit.unregister()  # type: ignore[attr-defined]


async def test_a_cad_sample_and_the_hooks_come_from_the_command_line(tmp_path: object) -> None:
    pytest.importorskip("OCP")
    from pathlib import Path

    from pyippdme.twin import cad

    step = Path(str(tmp_path)) / "block.step"
    cad.write_step([cad.CadBody("b", cad.make_box(40, 30, 20))], step)
    lines: list[str] = []
    options = virtual.options_from_args(
        _args("--sample", str(step), "--reference-sphere", "--time-scale", "0")
    )
    server = virtual.build_server(options, network=MemoryNetwork(), on_line_received=lines.append)
    twin = server.twin  # type: ignore[attr-defined]
    assert {o.kind for o in twin.objects} == {"sample", "fixture"}
    machine, _ = await _connect(server)
    await machine.dme.home()
    await machine.tool.re_qualify()  # the reference sphere is there to qualify on
    assert twin.snapshot().qualified
    assert any("ReQualify" in line for line in lines)
    await machine.close()
    await server.close()
    twin.toolkit.unregister()
