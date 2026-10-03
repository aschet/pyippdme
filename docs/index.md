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
(`.server`, `.dme`, `.cart_cmm`, `.tool`, `.tool_changer`, `.scanning`,
`.form_tester`, `.mover`, `.rotary_table`, `.part`, `.raw_data`) - see
`pyippdme.client.model` for what each namespace covers:

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
