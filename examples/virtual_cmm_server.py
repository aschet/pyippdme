#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Run a fully simulated CMM to develop and test client code against.

Real dimensional-metrology hardware is expensive; VirtualCMM is a complete,
in-process stand-in with every built-in command class enabled (coordinate
queries and motion, coordinate-system transformations, tool parameter
blocks, scanning, form-tester alignment, mover/temperature-compensation
commands, a rotary table, and raw-data acquisition/retrieval) - see
pyippdme.server.virtual_cmm for what each one covers and doesn't.

    python examples/virtual_cmm_server.py
    python examples/virtual_cmm_server.py --port 1294 --csy-dir ~/.pyippdme/csy

Then, in another terminal::

    ippdme client 127.0.0.1 1294

Equivalent to ``ippdme serve``; this script exists as a
copy-pasteable starting point for embedding a VirtualCMM in your own code
(e.g. as a pytest fixture for testing a client application), not because
the CLI can't already do this.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
from pathlib import Path

from pyippdme.server.virtual_cmm import VirtualCMM

from pyippdme.types.csy import FileCsyStore


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1294)
    parser.add_argument(
        "--csy-dir",
        type=Path,
        default=None,
        help="directory for persisted coordinate systems (default: ~/.pyippdme/csy)",
    )
    args = parser.parse_args()

    server = VirtualCMM(csy_store=FileCsyStore(args.csy_dir) if args.csy_dir else None)
    print(f"VirtualCMM listening on {args.host}:{args.port} (Ctrl+C to stop)")
    await server.serve_forever(args.host, args.port)


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(main())
