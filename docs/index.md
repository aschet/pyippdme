<!--
SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>

SPDX-License-Identifier: MPL-2.0
-->

# pyippdme

A Python client and server package for I++ DME (Dimensional Measurement
Equipment interface, VDMA 8722), the line-based TCP protocol used to control
coordinate measuring machines (CMMs) and related equipment.

This is alpha software. It is not complete, and its interfaces are not
stable and may change at any time.

This package has not been verified against real hardware. Commands sent to a
real machine may cause unexpected motion, collisions, or damage. Use it with
real equipment entirely at your own risk.

## Quickstart: server

`VirtualCMM` is a fully simulated CMM with every built-in command class
enabled and named coordinate systems persisted to disk:

```python
import asyncio
from pyippdme import VirtualCMM

async def main() -> None:
    await VirtualCMM().serve_forever(host="127.0.0.1", port=1294)

asyncio.run(main())
```

From the command line: `ippdme serve --host 127.0.0.1 --port 1294`. See
`examples/virtual_cmm_server.py` and `examples/raw_data_handling.py` in the
repository.

A plain `IppDmeServer` registers no commands. Pass the command classes you
want, for example the two every conformant server needs:

```python
from pyippdme import IppDmeServer
from pyippdme.server.classes import register_dme_class, register_server_class

server = IppDmeServer(command_classes=[register_server_class, register_dme_class])
```

## Quickstart: client

The low-level `IppDmeClient` sends/receives raw protocol nodes and mirrors
the wire format directly:

```python
import asyncio
from pyippdme import CommandName, IppDmeClient
from pyippdme.client import builders

async def main() -> None:
    client = await IppDmeClient.connect("127.0.0.1", 1294)
    await client.start_session()
    (version,) = await client.call(CommandName.GET_DME_VERSION)
    print(version)
    await client.call(CommandName.GO_TO, *builders.go_to(x=10))
    await client.end_session()
    await client.close()

asyncio.run(main())
```

`CommandName` (a `StrEnum`) names every command the built-in classes
register, and `ParameterName` names the argument/response identifiers
inside them (`X` in `GoTo(X(10))`, `Center`/`IJK` in `GoToOnCircle(...)`,
...) - a plain string like `"GoTo"`/`"X"` still works everywhere too, both
are there so this layer's own users don't have to retype/risk mistyping
one by hand. `pyippdme.client.builders` goes a step further for commands whose
argument *shape* (not just names) is easy to get wrong - `builders.go_to(x=10)`
builds the same `NamedValue(...)` tuple as the example above by hand;
`IppDmeMachine` itself uses the same functions internally, so a raw-client
user and this layer's own typed methods build requests identically.

`IppDmeMachine` wraps the same connection with typed methods and return
values instead, grouped the way VDMA 8722's own object model groups them
(`.server`, `.dme`, `.cart_cmm`, `.tool`, `.found_tool`, `.tool_changer`, `.scanning`,
`.form_tester`, `.mover`, `.rotary_table`, `.part`, `.raw_data`, `.feature_extraction`)
- see `pyippdme.client.model` for what each namespace covers:

```python
import asyncio
from pyippdme import IppDmeMachine

async def main() -> None:
    machine = await IppDmeMachine.connect("127.0.0.1", 1294)
    await machine.start_session()
    await machine.cart_cmm.go_to(x=10, y=20)
    print(await machine.cart_cmm.get_position())
    async for point in machine.scanning.scan_on_line(
        (0, 0, 0), (10, 0, 0), (0, 0, 1), step_width=2.0
    ):
        print(point)
    await machine.end_session()
    await machine.close()

asyncio.run(main())
```

Anything `IppDmeMachine` doesn't wrap is still reachable via
`machine.client.call(...)` directly.

The namespaces cover the commands and properties of VDMA 8722, including the
deprecated `FeatureExtraction` class. For example `machine.server` has the
session, error and property commands, including `clear_all_errors()`.

Moves and measurements take everything the standard allows as arguments:

```python
from pyippdme.client.model import ToolAlignment

await machine.cart_cmm.go_to(x=10, r=90, alignment=ToolAlignment((0, 0, 1)), sync=True)
report = await machine.cart_cmm.pt_meas(x=1, y=2, z=3, ijk=(0, 0, 1))
```

A response made of named values, such as the result of `pt_meas()` or `get()`,
is a `Report`: a single number is a `float` and several numbers are a tuple.
`report.number("X")` and `report.vector("IJK")` give the typed value.

A scan yields `(x, y, z)` points. To get other values for each point, such as the
quality or the surface direction, use `machine.scanning.reporting(...)`:

```python
scan = machine.scanning.reporting("X", "Y", "Z", "Q", "IJK")
async for point in scan.scan_on_line((0, 0, 0), (10, 0, 0), (0, 0, 1), step_width=2.0):
    print(point.number("Q"), point.vector("IJK"))
```

Each method sends its command as soon as it is called and returns a handle.
Awaiting the handle waits until the command is done and gives its typed result.
You can also check the acknowledgement first:

```python
call = machine.cart_cmm.pt_meas(x=1, y=2, z=3)  # command sent
await call.acknowledged()                       # the server received it
report = await call                             # a Report

scan = machine.scanning.scan_on_line((0, 0, 0), (10, 0, 0), (0, 0, 1), step_width=2.0)
await scan.acknowledged()
async for point in scan:                        # points as they arrive
    print(point)
```

An Ack only means the server received the command. A command that then fails
is reported after the Ack and raises `IppDmeServerError` when you await the
handle, not from `acknowledged()`.

The standard (5.4.3) does not let a client send a command before the previous one
was acknowledged. The client therefore sends commands one after another, also when
you start several at once; commands for prioritized execution (names ending in `E`,
such as `AbortE()`) go out at once, so that they can get past a running command.

For the raw responses instead of the parsed result, `await call.transaction()`
gives the `Transaction` of the command that was sent:

```python
call = machine.cart_cmm.pt_meas(x=1, y=2, z=3)
transaction = await call.transaction()
raw = await transaction.wait_complete()         # tuple of DataPayload nodes
```

## Events from the server

A server can tell the client that something happened without being asked
(5.5.3): a key pressed on the jog box, a clearance point or a manual point, a tool
changed or a property set at the machine, a tool collection opened.
`machine.server.events()` yields them as `KeyPress`, `ClearancePoint`,
`ManualPoint`, `ToolChanged`, `PropertyChanged` and `ToolCollectionOpened`
(`pyippdme.client.events`), and anything else as `UnknownEvent`:

```python
async for event in machine.server.events():
    if isinstance(event, KeyPress):
        print(event.key)
```

A `VirtualCMM` sends them with `key_press()`, `clearance_point()`, `manual_point()`,
`change_tool()`, `open_tool_collection()` and `set_property()`. A custom
`IppDmeServer` uses `send_event()` with the builders in `pyippdme.server.builders`.

## Raw data

All three ways to get raw data (file, shared memory, binary socket) deliver the
scanpoints binary format ESBF of Annex C.2. `pyippdme.rawdata.formats` reads and writes it
and the older SBF (`unpack_esbf`, `pack_esbf`, `unpack_sbf`, `pack_sbf`), and
`pyippdme.rawdata.transfer.read_samples` reads the stream of a binary socket.

## In-process connections

Everything that opens or accepts a connection (`IppDmeClient`, `IppDmeMachine`,
`IppDmeServer`, `VirtualCMM`, `Spy`) takes a `network` argument. The default is
TCP. A `MemoryNetwork` keeps the connections inside the process, so a server and
its clients can talk without opening a port, which is handy for tests and for
simulation:

```python
import asyncio
from pyippdme import IppDmeMachine, VirtualCMM
from pyippdme.protocol.network import MemoryNetwork

async def main() -> None:
    network = MemoryNetwork()
    server = VirtualCMM(network=network)
    port = await server.start()
    machine = await IppDmeMachine.connect("virtual", port, network=network)
    await machine.start_session()
    print(await machine.cart_cmm.get_position())
    await machine.close()
    await server.close()

asyncio.run(main())
```

A `MemoryNetwork` ignores host names and matches connections by port only. The
network is a small protocol (`open_connection` and `start_server`, mirroring
asyncio), so other implementations can be plugged in the same way.

## Spy

`Spy` is a proxy between a client and a server that reports every line it
relays (`on_message`). It only observes unless you give it an `intercept`
function, which is awaited for each received line and returns what to send on
instead: the same bytes to forward it unchanged, other bytes to alter it,
`None` to drop it, or a list of lines to replace it with several. Lines are raw
bytes ending in `\r\n`. `ippdme spy` never filters anything.

```python
from pyippdme.spy import Spy, SpyDirection, SpyMessage

async def intercept(message: SpyMessage) -> bytes | None:
    if message.direction is SpyDirection.TO_SERVER and b"GoTo" in message.line:
        return None  # drop every GoTo
    return message.line

spy = Spy("192.168.1.50", 1294, intercept=intercept)
```

A dropped or altered command can leave the client waiting for a response the
server never sends. The spy does not invent one.

## Static command catalog

Building your own tooling on top of this package - a REPL/TUI completion
engine, a GUI command palette, a linter for hand-written scripts - and want
to know what commands and arguments *exist* without a live connection?
`pyippdme.simulation.catalog.BUILTIN_COMMANDS` is a static, read-only catalog
of every command this package's built-in classes register, computed once
at import time - no server instance, no connection, no network:

```python
from pyippdme.simulation.catalog import BUILTIN_COMMANDS

info = BUILTIN_COMMANDS["GoTo"]
print([p.name for p in info.arguments])  # ['Positions', 'Sync'] - argument-name hints
```

This is deliberately distinct from a live server's *actual* capabilities
(`GetSupportedCommands`/`GetSupportedArguments(CommandName)`, 6.4.1, over a
real connection - the authoritative answer for that specific server, which
may register only a subset of the built-in classes, or add its own
proprietary commands `BUILTIN_COMMANDS` has no way to know about).
`build_command_catalog(command_classes)` builds the same kind of catalog
for a custom class subset, e.g. matching a particular
`IppDmeServer(command_classes=[...])` configuration.

## Digital twin and simulator window

`VirtualCMM` stays the small, GUI-free simulation for the command line. For a
simulation you can see, `pyippdme.twin` builds on the same protocol server and adds
a physical model, and `pyippdme.gui` shows it in a Qt window. A client connects over
TCP with the normal I++ DME protocol and gets the answers of that model.

```bash
pip install "pyippdme[gui]"          # PySide6 and OpenCASCADE (cadquery-ocp)
ippdme gui --start --sample part.step
```

In the window you can load a sample from a STEP, IGES, STL or BREP file, place it on the
table or the rotary table (position, rotation, "rest on table"), add box and cylinder
fixtures, switch machine presets, change the playback speed, the measuring noise and the
part temperature, jog by hand, and watch the protocol lines. Measured points and scan
point clouds appear in the 3D view.

What the twin simulates behind the protocol:

- **Motion.** `GoTo`, `Step` and `PtMeas` take time with a trapezoidal speed profile (maximum
  vector speed and acceleration of the machine), `AbortE()` stops them, and the machine
  must be homed first (error 1011 otherwise). A target outside the machine volume is
  rejected with 1008.
- **Collisions.** The stylus, probe holder, table, rotary table, the moving machine parts,
  the sample and fixtures are checked against each other with exact OpenCASCADE
  distances. A collision stops the machine where it hit and reports 2504. During
  `PtMeas` the tip is the sensor, so touching the part is not a collision.
- **Probing.** `PtMeas` rays hit the exact B-rep of the CAD sample, with a measuring
  error that follows the machine's MPE_E = A + L/K and MPE_P (a third of the MPE as one
  standard deviation) and a thermal expansion error for a part that is not at 20 °C.
- **Scanning.** `ScanOnLine` and the other scanning commands run at the scanning speed
  and show the probe moving; line scans follow the CAD surface along the probing
  direction when it is within a few millimetres.
- **Raw data.** `DataAcquire` is answered by a laser line scanner (`LineScanner`) that
  ray-casts the CAD sample with noise and dropouts, for `SingleShot`, `MultiShot` and
  `Sweep`.
- **Tools.** The four tools of the catalog have a physical stylus (ball, stem, holder)
  that is drawn and used for collisions; `ChangeTool` changes it, and `AlignTool` turns
  its stem.

### Machines from STEP files

The machine is a set of STEP bodies that move with the axes. Export the default machine to
see the layout, edit or replace the files, and load the directory again:

```python
from pyippdme.twin import MachineModel

MachineModel.default("bridge-900", rotary=True).export("my_cmm")   # machine.toml + STEP files
machine = MachineModel.from_directory("my_cmm")
```

```toml
[machine]
preset = "bridge-700"          # start from a preset, then override
travel = [900, 1200, 700]
max_speed = 520
acceleration = 1200
[machine.accuracy]
a_um = 1.6
k = 350
[machine.rotary]
origin = [450, 600, -10]       # a rotary table, centre on its top
speed = 90

[[component]]
name = "table"
step = "granite.step"           # no step: the generated body is used
[[component]]
name = "bridge"
step = "bridge.step"
moves_with = ["y"]
[[component]]
name = "carriage"
step = "carriage.step"
moves_with = ["x", "y"]
[[component]]
name = "quill"
step = "quill.step"
moves_with = ["x", "y", "z"]
[[component]]
name = "turntable"
step = "turntable.step"
rotates = true
```

A single STEP assembly works as well (`MachineModel.from_step`, or *Load machine…*): its parts
are recognised by name (table/granite, bridge/portal, carriage/slide, quill/spindle/ram,
rotary/turntable). Travel ranges and the machine zero that you do not give are estimated
from the geometry (`machine.spec.derived` lists what was estimated), so check them.
Coordinates are machine coordinates: origin at the home position, Z up, the table
surface 10 mm below zero.

The presets give representative numbers for bridge machines (measuring range, MPE_E =
A + L/K, axis speed and acceleration in the range of public manufacturer data sheets);
they are not a model of one specific machine. Take the numbers of your machine from its
data sheet.

### Your own simulation in the window

The window shows anything that satisfies `pyippdme.twin.view.SimulationView` (a
snapshot, a list of meshes with poses, and listeners). What answers the client is
a set of seams of the protocol server, each a small Protocol that you can implement
yourself and pass to `IppDmeServer` or `VirtualCMM`:

| Seam | Used by | Without it |
| --- | --- | --- |
| `pyippdme.server.motion.MotionModel` | every move: time, limits, collisions | moves are instant |
| `pyippdme.server.surface.SampleSurface` | `PtMeas` | the commanded point is reported |
| `pyippdme.server.surface.RawSensor` | `DataAcquire` | points are made up from the request |
| `pyippdme.server.backend.MachineBackend` | scanning commands | nominal path |

`DigitalTwin.create_server()` shows how the twin wires them. `pyippdme.twin.host.ServerHost`
runs a server on a background thread so that a GUI keeps its main thread.

Limits of the twin: the 3D view is a software renderer without OpenGL, so very large meshes
are thinned while the camera moves; the tool catalog is the fixed set of
`pyippdme.simulation.classes.tool_class.TOOL_CATALOG`; homing is instant; stylus bending,
probe pre-travel, lobing and temperature of the machine itself are not modelled; the
bridge and carriage are only checked against the sample and fixtures, not against the
stylus.

## Deviations and open points

Where this package departs from VDMA 8722:2024-04, or has to guess:

- The server acknowledges every command at once. The standard (5.4.3) lets a server
  delay the Ack until it can accept more commands.
- `VirtualCMM` does not require `Home()` before it moves, has no machine volume
  (error 2500) and no collisions (2504), and never reports 1014.
  Every measuring tool accepts every measuring and scanning command, so the
  error 2002 ("Type of probe does not allow this operation") never occurs.
- Coordinate system transformations are stored and returned, but not applied to
  coordinates: the transformation chain is defined by Figure 12, a diagram that the
  text of the standard does not contain.
- `GetXtdErrStatus()` reports active errors as `ActiveError()` and `Severity()` data
  lines. Table 15 can also be read as asking for error responses.
- `GetChangeToolAction()` answers `Argument(Switch),X(0),Y(0),Z(0)`. The standard
  lists the action as an unnamed value, followed by named ones.
- `GetRawDataShaMem()` sends offset and size as decimal numbers, although Table 101 calls
  them "hex coded", and `GetRawDataFile()` sends the URL as a quoted string
  although Table 103 calls it a name.
- Values that Tables 66, 70 and 32 call unnamed (`ER`, `Q`, `IJKAct`) are sent named in
  the response to `PtMeas` and `Get`, because the grammar cannot mix unnamed and
  named values in one response.
- `VirtualCMM` reports one machine class, although 6.4.1 allows several.
- The standard gives no examples for these encodings, so they are guesses: the
  `pi,pj,pk`, `si,sj,sk` and `R()` items of the `ScanOnCurve` format, the
  `include`/`exclude` flag of `ROI`, and the `Acqs(..)`, `ROIs(..)`, `QEPs(S(..))` and
  `GeoElem()` arguments of `FeatureExtract`.
- The deprecated `FeatureExtraction` class (Annex J.2) is in the client but not
  simulated.

## Interactive shell

```bash
ippdme client 127.0.0.1 1294
```

Type bare method calls (`StartSession()`, `GoTo(X(10), Y(20))`, ...) and see
the raw `&`/`#`/`%`/`!` responses as they arrive. This is deliberately a
barebones tool - just enough to see whether a server is answering, not a
full shell: no meta-commands, no tab-completion, no reconnecting
mid-session (the connection is fixed for the whole run). Also stdlib only
(no `prompt_toolkit` or similar), so `pyippdme` never pulls in an
interactive-shell dependency for a user of the API only - line editing is
just `input()` with `readline`-provided history on platforms that have it.
Unsolicited (`E0000`) events print as they arrive throughout the session.

Don't have a server running yet? `ippdme client --virtual` (and `ippdme tui
--virtual`, the full-screen equivalent) starts an in-process `VirtualCMM`
and connects to it automatically - no separate `ippdme serve` in another
terminal needed. It's torn down again when you quit.

Want to run a file of commands instead of typing them? `ippdme client
127.0.0.1 1294 --file session.iscript` runs every line non-interactively
(same syntax as typed interactively; blank lines and `#` comments are
allowed and copied through verbatim, and a leading wire tag like `00047
GoTo(...)` is accepted and ignored) and prints the same Ack/Data/Done/Error
output - this project's own take on the NIST/I++ DME reference test
suite's `.prg`/`.res` file pairs; see `pyippdme.cli.script` for how and why
the format differs. `--file` and `--virtual` combine too, for a quick
"does my script still work" check against a throwaway simulated server.

`client`, `serve`, and `spy` all share one `--session-log <file>` flag
that records every wire line sent/received, plus connect/disconnect, in a
single unified format (timestamp, then `>`/`<` for traffic direction or
`CONNECTED`/`DISCONNECTED`, then the raw tagged wire text) - see
`pyippdme.cli.session_log`. They also share `--commands-file <file>`,
which records just the commands actually run (tag-stripped, one per
line), in exactly the syntax `ippdme client --file` accepts - so a
session can be replayed later.

## Extending the server

Add commands from other VDMA 8722 classes via `server.registry.command(...)`.

For your own proprietary commands, VDMA 8722 6.1 requires a two-character
company namespace prefix (e.g. a fictional company "XX" would add
`XXMyCommand(...)`); use `command_proprietary`/`register_proprietary`
instead of `command`/`register` so that convention is built and validated
for you rather than typed by hand:

```python
from pyippdme import IppDmeServer
from pyippdme.protocol.signature import DataType, Parameter
from pyippdme.server.classes import register_dme_class, register_server_class

server = IppDmeServer(command_classes=[register_server_class, register_dme_class])

@server.registry.command_proprietary(
    "XX", "MyCommand", arguments=(Parameter("Value", DataType.INT),)
)
async def _my_command(ctx, args):
    ...
```

The same two-character convention applies to properties and parameters
(e.g. `GoTo(XXMyParameter, ...)`); build those names with
`pyippdme.proprietary_name("XX", "MyParameter")` too.

`arguments` is optional (a command registered without it still works,
`GetSupportedArguments` just can't describe it) but works the same way for
`command`/`command_proprietary` as it does for `register`/`register_proprietary`
- pass it so `GetSupportedArguments("XXMyCommand")` can answer for real
instead of falling back to "0506 Argument not supported".

```{toctree}
:hidden:

api
```
