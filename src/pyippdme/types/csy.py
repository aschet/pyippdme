# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Coordinate-system transformation math and persistence (6.5.1/6.5.2).

A CartCMM defines several coordinate systems (CSYs) that sit below the base
``MachineCsy`` in a transformation chain (``MachineCsy`` -> ... ->
``PartCsy``, 6.5.1, Figure 12 - not reproduced here, it is a
diagram). The transformation from a parent CSY to a child one is a rigid
motion: a proper Euler-angle rotation (angles ``Phi``, ``Theta``, ``Psi``
about axes Z, X', Z'' respectively, in that composition order) followed by a
translation, i.e. ``p_parent = R(Phi, Theta, Psi) @ p_child + t``.

This module provides that transform as a reusable, tested building block
(:class:`CoordinateTransform`) plus a pluggable persistence seam
(:class:`CsyStore`, mirroring :class:`~pyippdme.server.backend.MachineBackend`) for
the named coordinate systems ``SaveNamedCsyTransformation``/
``LoadCoordSystem`` and friends require to survive "in another session and
after rebooting" (6.5.2).

Scope note: :mod:`pyippdme.simulation.classes.cartcmm_class`'s command handlers store
and return these transforms faithfully, but the simulated ``Get``/``GoTo``
handlers do *not* apply them to convert reported/commanded coordinates
between CSYs - that would require committing to a specific interpretation of
the full transformation chain across all six CSYs, which Figure 12 (a
diagram this library cannot extract from the PDF) defines but the
surrounding prose does not fully spell out. A real integrator wiring this
library to actual hardware is expected to call :meth:`CoordinateTransform.apply`
themselves when composing the chain their machine actually implements.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt

from pyippdme.types.vec3 import Vec3, from_array, to_array

#: The CSY names ``SetCsyTransformation``/``GetCsyTransformation`` accept (6.5.2,
#: Tables 35/38): child CSYs whose transformation relative to their parent in
#: the chain can be queried/replaced live, for the current connection only.
LIVE_TRANSFORM_NAMES = (
    "PartCsy",
    "JogDisplayCsy",
    "JogMoveCsy",
    "SensorCsy",
    "MoveableMachineCsy",
    "MultipleArmCsy",
    "RotaryTableFixCsy",
)


@dataclass(frozen=True, slots=True)
class CoordinateTransform:
    """A rigid transform from a parent CSY to a child CSY (6.5.1).

    ``psi``/``phi`` are normalized modulo 360 on construction, per 6.5.2's
    "Psi and Phi are unlimited but 'normalized' by the server (modulo 360)".
    ``theta`` is left as given; callers enforce its ``[0, 180]`` range
    (error 1007 "Theta out of range") since only they know whether that
    should be a hard rejection or a silent clamp.
    """

    x0: float
    y0: float
    z0: float
    theta: float
    psi: float
    phi: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "psi", self.psi % 360.0)
        object.__setattr__(self, "phi", self.phi % 360.0)

    @classmethod
    def identity(cls) -> CoordinateTransform:
        return cls(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    @property
    def is_theta_in_range(self) -> bool:
        return 0.0 <= self.theta <= 180.0

    def rotation_matrix(self) -> npt.NDArray[np.float64]:
        """Compute the proper Euler rotation matrix ``Rz(Phi) @ Rx(Theta) @ Rz(Psi)``."""
        phi, theta, psi = (math.radians(a) for a in (self.phi, self.theta, self.psi))
        return _rotation_z(phi) @ _rotation_x(theta) @ _rotation_z(psi)

    def apply(self, point: Vec3) -> Vec3:
        """Transform ``point`` from the child CSY into the parent CSY: ``R @ p + t``."""
        rotated = self.rotation_matrix() @ to_array(point)
        return from_array(rotated + np.array((self.x0, self.y0, self.z0)))

    def inverse(self) -> CoordinateTransform:
        """Compute the transform from the parent CSY back to the child CSY."""
        r_inv = self.rotation_matrix().T  # a rotation matrix is orthogonal: R^-1 == R^T
        t_inv = -(r_inv @ np.array((self.x0, self.y0, self.z0)))
        phi, theta, psi = _euler_angles_from_matrix(r_inv)
        return CoordinateTransform(
            float(t_inv[0]), float(t_inv[1]), float(t_inv[2]), theta, psi, phi
        )


def _rotation_z(angle: float) -> npt.NDArray[np.float64]:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rotation_x(angle: float) -> npt.NDArray[np.float64]:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _euler_angles_from_matrix(r: npt.NDArray[np.float64]) -> tuple[float, float, float]:
    """Recover ``(phi, theta, psi)`` in degrees from a Z-X'-Z'' rotation matrix."""
    theta = math.degrees(math.acos(np.clip(r[2, 2], -1.0, 1.0)))
    if abs(r[2, 2]) < 1.0 - 1e-9:
        phi = math.degrees(math.atan2(r[0, 2], -r[1, 2]))
        psi = math.degrees(math.atan2(r[2, 0], r[2, 1]))
    else:
        # Gimbal lock (theta == 0 or 180): phi and psi are not individually
        # observable, only their sum/difference is. Attribute the whole
        # rotation to phi, leaving psi at 0.
        phi = math.degrees(math.atan2(r[1, 0], r[0, 0]))
        psi = 0.0
    return phi, theta, psi


class CsyStore(Protocol):
    """Pluggable persistence for named coordinate systems (6.5.2 "stored persistently").

    Mirrors :class:`~pyippdme.server.backend.MachineBackend`: :class:`IppDmeServer
    <pyippdme.server.IppDmeServer>` defaults to :class:`InMemoryCsyStore`
    (matching its default of :class:`~pyippdme.server.backend.SimulatedBackend` -
    no surprise disk I/O out of the box), and a real deployment that needs
    the standard's persistence guarantee passes a :class:`FileCsyStore`
    (or its own implementation, e.g. backed by a database) explicitly.
    """

    async def save(self, name: str, transform: CoordinateTransform) -> None: ...

    async def load(self, name: str) -> CoordinateTransform | None: ...

    async def delete(self, name: str) -> bool:
        """Delete ``name``; return whether it existed."""
        ...

    async def names(self) -> tuple[str, ...]: ...


class InMemoryCsyStore:
    """A :class:`CsyStore` that only lasts for the process lifetime."""

    def __init__(self) -> None:
        self._transforms: dict[str, CoordinateTransform] = {}

    async def save(self, name: str, transform: CoordinateTransform) -> None:
        self._transforms[name] = transform

    async def load(self, name: str) -> CoordinateTransform | None:
        return self._transforms.get(name)

    async def delete(self, name: str) -> bool:
        return self._transforms.pop(name, None) is not None

    async def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._transforms))


class FileCsyStore:
    """A :class:`CsyStore` that persists each named transform as a JSON file.

    ``Name`` is client-controlled (wire type ``[string]``, so it may contain
    ``/`` or ``..``); :meth:`_path_for` rejects anything that is not a plain
    file-safe name rather than trusting it as a path component.
    """

    def __init__(self, directory: Path | None = None) -> None:
        self.directory = directory if directory is not None else Path.home() / ".pyippdme" / "csy"

    async def save(self, name: str, transform: CoordinateTransform) -> None:
        path = self._path_for(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "x0": transform.x0,
            "y0": transform.y0,
            "z0": transform.z0,
            "theta": transform.theta,
            "psi": transform.psi,
            "phi": transform.phi,
        }
        path.write_text(json.dumps(payload, indent=2))

    async def load(self, name: str) -> CoordinateTransform | None:
        path = self._path_for(name)
        if not path.is_file():
            return None
        data = json.loads(path.read_text())
        return CoordinateTransform(**data)

    async def delete(self, name: str) -> bool:
        path = self._path_for(name)
        if not path.is_file():
            return False
        path.unlink()
        return True

    async def names(self) -> tuple[str, ...]:
        if not self.directory.is_dir():
            return ()
        return tuple(sorted(p.stem for p in self.directory.glob("*.json")))

    def _path_for(self, name: str) -> Path:
        if not name or any(c in name for c in '/\\:*?"<>|') or name in (".", ".."):
            raise ValueError(f"{name!r} is not a valid coordinate system name")
        return self.directory / f"{name}.json"
