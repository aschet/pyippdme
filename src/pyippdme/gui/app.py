# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Entry point of the simulator window (``ippdme gui``)."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ippdme gui", description="Virtual CMM with a visible digital twin"
    )
    parser.add_argument("--host", default="127.0.0.1", help="address the server listens on")
    parser.add_argument("--port", type=int, default=1294, help="port the server listens on")
    parser.add_argument(
        "--machine", help="machine directory with a machine.toml, or a STEP assembly"
    )
    parser.add_argument("--sample", help="CAD file (STEP/IGES/STL/BREP) to place on the table")
    parser.add_argument("--start", action="store_true", help="start the server at once")
    args = parser.parse_args(list(argv) if argv is not None else None)

    from PySide6.QtWidgets import QApplication

    from pyippdme.gui.main_window import MainWindow
    from pyippdme.twin import DigitalTwin, MachineModel

    app = QApplication(sys.argv[:1])
    machine = None
    if args.machine:
        path = Path(args.machine)
        machine = (
            MachineModel.from_directory(path) if path.is_dir() else MachineModel.from_step(path)
        )
    twin = DigitalTwin(machine)
    if args.sample:
        twin.load_sample(args.sample)
    window = MainWindow(twin)
    window.host_edit.setText(args.host)
    window.port_spin.setValue(args.port)
    window.show()
    if args.start:
        window.toggle_server()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
