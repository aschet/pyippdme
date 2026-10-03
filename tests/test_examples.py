# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""The scripts in ``examples/`` must keep working."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

EXAMPLES = Path(__file__).parent.parent / "examples"


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"example_{name}", EXAMPLES / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["virtual_cmm_server", "raw_data_handling"])
def test_example_imports(name: str) -> None:
    assert hasattr(_load(name), "main")


async def test_raw_data_example_runs(capsys: pytest.CaptureFixture[str]) -> None:
    await _load("raw_data_handling").main()
    output = capsys.readouterr().out
    for technology in ("[File]", "[SharedMemory]", "[BinarySocket]"):
        assert technology in output
