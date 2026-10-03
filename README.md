# pyippdme

[![CI](https://github.com/aschet/pyippdme/actions/workflows/ci.yml/badge.svg)](https://github.com/aschet/pyippdme/actions/workflows/ci.yml)
[![Docs](https://github.com/aschet/pyippdme/actions/workflows/docs.yml/badge.svg)](https://aschet.github.io/pyippdme/)

A Python client and server package for I++ DME (Dimensional Measurement
Equipment interface, VDMA 8722), the line-based TCP protocol used to control
coordinate measuring machines (CMMs) and related equipment. Implements
VDMA 8722:2024-04 (I++ DME Version 2.5).

## Warning

This package has not been verified against real hardware. Commands sent to a
real machine may cause unexpected motion, collisions, or damage. Use it with
real equipment entirely at your own risk.

## Status

Alpha software. It is not complete, and the Python API, command-line
options and behavior may change at any time without notice or deprecation.
Implemented so far:

- Wire protocol: parsing and encoding, sessions, tags, events, prioritized
  execution and `AbortE()`, error handling.
- Client: `IppDmeClient`, a low-level client mirroring the wire protocol, and
  `IppDmeMachine`, a typed interface organized like the standard's object
  model.
- Server: `IppDmeServer`, a server you extend with your own command classes
  and machine backend.
- Simulation: `VirtualCMM`, a simulated CMM to develop and test clients
  against without hardware. It covers a large part of the standard
  (`Server`, `DME`, `CartCMM`, `TouchTrigger`, `Tool`, `ToolChanger`,
  `Scanning`, `FormTester`, `Mover`, `RotaryTable`, `Part`, raw data
  handling). Several scanning and alignment commands are only plausible
  responses, not real simulations.
- Networks: connections run over TCP by default, or over an in-memory
  network inside the process, without opening a port.
- Spy: a transparent proxy that logs the traffic between a client and a
  server.
- CLI and TUI: `ippdme client`, `ippdme serve`, `ippdme spy` and a
  full-screen `ippdme tui`.

Not implemented by the simulation: the deprecated `FeatureExtraction` class,
a second orthogonal rotary table, and the optional tool-direction and
rotary-table columns of `ScanOnCurve`.

## Installation

```bash
pip install -e . --group dev
# for the full-screen TUI:
pip install -e ".[tui]"
```

## Quickstart

```bash
# in-process simulated CMM, no separate server needed
ippdme client --virtual
ippdme tui --virtual

# or run a simulated server and connect to it
ippdme serve --host 127.0.0.1 --port 1294
ippdme client 127.0.0.1 1294
```

```
> StartSession()
> GoTo(X(10), Y(20))
> Get(X(), Y(), Z())
# X(10),Y(20),Z(0)
```

`ippdme client --file <script>` runs the commands in a file instead.

See `docs/index.md` for the Python API and `examples/` for runnable scripts.

## Development

```bash
pip install -e . --group dev --group docs
pre-commit install
pytest --cov=pyippdme
pre-commit run --all-files
```
