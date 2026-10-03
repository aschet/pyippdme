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
and return these transforms faithfully, but the minimal simulated ``Get``/``GoTo``
handlers do *not* apply them to convert reported/commanded coordinates
between CSYs: they have no machine coordinates to convert to. The chain across the
CSYs is defined by Figure 12 (a diagram this library cannot extract from the PDF); this module
commits to one reading of it, :data:`CSY_CHAIN` and :func:`chain_matrix`, which
:mod:`pyippdme.twin` applies. A machine with a different chain can compose its own with
:meth:`CoordinateTransform.apply`.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
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


#: The transformation chain of a CartCMM (6.5.1, Figure 12), from the machine to the part.
#: Coordinates of ``PartCsy`` pass through ``RotaryTableVarCsy``, ``RotaryTableFixCsy``,
#: ``MoveableMachineCsy`` and ``MultipleArmCsy`` into ``MachineCsy``. Each CSY is placed relative to
#: the one before it in this tuple with ``SetCsyTransformation`` (6.5.2), and
#: ``SetCoordSystem`` chooses where a client enters the chain. The offset of the tool is
#: included in all of them. ``RotaryTableVarCsy`` and ``MoveableMachineCsy`` are in the figure
#: for consistency (the rotary table's angle and movable measuring equipment).
CSY_CHAIN = (
    "MachineCsy",
    "MultipleArmCsy",
    "MoveableMachineCsy",
    "RotaryTableFixCsy",
    "RotaryTableVarCsy",
    "PartCsy",
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
        """Return the rotation from child to parent coordinates: ``Rz(Psi) @ Rx(Theta) @ Rz(Phi)``.

        The standard (6.5.1, Figure 12 text) defines the transformation from the parent to the
        child CSY as ``p' = R(phi, theta, psi) . (p - t)`` with::

            R = [[ cos phi, sin phi, 0], [-sin phi, cos phi, 0], [0, 0, 1]]
              . [[1, 0, 0], [0, cos theta, sin theta], [0, -sin theta, cos theta]]
              . [[ cos psi, sin psi, 0], [-sin psi, cos psi, 0], [0, 0, 1]]

        Those are rotations of the axes (the transpose of the usual rotation matrices), so the
        child-to-parent direction this method returns is the transpose of ``R``, which is
        ``Rz(psi) Rx(theta) Rz(phi)`` in the usual notation (``phi`` and ``psi`` swap places).
        """
        phi, theta, psi = (math.radians(a) for a in (self.phi, self.theta, self.psi))
        return _rotation_z(psi) @ _rotation_x(theta) @ _rotation_z(phi)

    def parent_to_child(self, point: Vec3) -> Vec3:
        """Transform ``point`` from the parent CSY into the child CSY: ``R . (p - t)`` (6.5.1)."""
        t = np.array((self.x0, self.y0, self.z0))
        return from_array(self.rotation_matrix().T @ (to_array(point) - t))

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


def transform_matrix(transform: CoordinateTransform) -> npt.NDArray[np.float64]:
    """Return the 4x4 homogeneous matrix of ``transform`` (child to parent coordinates)."""
    m = np.eye(4)
    m[:3, :3] = transform.rotation_matrix()
    m[:3, 3] = (transform.x0, transform.y0, transform.z0)
    return m


def chain_matrix(
    active: str,
    transforms: Mapping[str, CoordinateTransform],
    rotary_var: npt.NDArray[np.float64] | None = None,
) -> npt.NDArray[np.float64]:
    """Return the 4x4 matrix that maps points of the ``active`` CSY into ``MachineCsy``.

    ``transforms`` holds the live transformations of the CSYs of :data:`CSY_CHAIN` by name (a
    CSY without one is placed at its parent). ``RotaryTableVarCsy`` is not set by a command
    but follows the rotary table: pass its 4x4 matrix as ``rotary_var``, or ``None`` while that
    calculation is not enabled. An unknown ``active`` name (``MachineCsy`` included) is the
    identity.
    """
    if active not in CSY_CHAIN:
        return np.eye(4)
    result = np.eye(4)
    for name in CSY_CHAIN[1 : CSY_CHAIN.index(active) + 1]:
        if name == "RotaryTableVarCsy":
            step = rotary_var if rotary_var is not None else np.eye(4)
        else:
            transform = transforms.get(name)
            step = transform_matrix(transform) if transform is not None else np.eye(4)
        result = result @ step
    return result


@dataclass(frozen=True, slots=True)
class CsyContext:
    """The coordinate system a client works in, and how it sits in the machine (6.5.1, 6.5.2).

    This is what a server needs to turn the coordinates of a command into machine
    coordinates and a result back into the client's: the CSY the client activated with
    ``SetCoordSystem``, the live transformations it set with ``SetCsyTransformation`` and
    ``LoadCoordSystem`` (by CSY name, each relative to the one before it in
    :data:`CSY_CHAIN`), and, while the rotary table's own system is calculated, that
    system's 4x4 matrix::

        context = CsyContext("PartCsy", {"PartCsy": CoordinateTransform(100, 0, 0, 0, 0, 90)})
        machine_point = context.to_machine((10, 0, 0))
        client_point = context.to_client(machine_point)
        probing = context.direction_to_machine((0, 0, 1))   # IJK, tool axis: rotation only

    Points and directions are separate because a direction is not moved by a translation.
    """

    active: str = "MachineCsy"
    transforms: Mapping[str, CoordinateTransform] = field(default_factory=dict)
    #: Matrix of ``RotaryTableVarCsy`` (the rotary table's angle) while it is calculated.
    rotary_var: npt.NDArray[np.float64] | None = None

    def matrix(self) -> npt.NDArray[np.float64]:
        """Return the 4x4 matrix mapping the active CSY into ``MachineCsy``."""
        return chain_matrix(self.active, self.transforms, self.rotary_var)

    def to_machine(self, point: Vec3) -> Vec3:
        """Convert a point of the active CSY to machine coordinates."""
        m = self.matrix()
        return from_array(m[:3, :3] @ to_array(point) + m[:3, 3])

    def to_client(self, point: Vec3) -> Vec3:
        """Convert a machine point to the active CSY."""
        m = self.matrix()
        return from_array(m[:3, :3].T @ (to_array(point) - m[:3, 3]))

    def direction_to_machine(self, vector: Vec3) -> Vec3:
        """Convert a direction of the active CSY to machine coordinates (rotation only)."""
        return from_array(self.matrix()[:3, :3] @ to_array(vector))

    def direction_to_client(self, vector: Vec3) -> Vec3:
        """Convert a machine direction to the active CSY (rotation only)."""
        return from_array(self.matrix()[:3, :3].T @ to_array(vector))


def _rotation_z(angle: float) -> npt.NDArray[np.float64]:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rotation_x(angle: float) -> npt.NDArray[np.float64]:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _euler_angles_from_matrix(r: npt.NDArray[np.float64]) -> tuple[float, float, float]:
    """Recover ``(phi, theta, psi)`` in degrees from ``Rz(psi) Rx(theta) Rz(phi)``."""
    theta = math.degrees(math.acos(np.clip(r[2, 2], -1.0, 1.0)))
    if abs(r[2, 2]) < 1.0 - 1e-9:
        psi = math.degrees(math.atan2(r[0, 2], -r[1, 2]))
        phi = math.degrees(math.atan2(r[2, 0], r[2, 1]))
    else:
        # Gimbal lock: only one combination of phi and psi shows. At theta = 0 the rotation is
        # Rz(psi + phi), at theta = 180 it is Rz(psi - phi) mirrored; attribute it to one angle.
        angle = math.degrees(math.atan2(r[1, 0], r[0, 0]))
        phi, psi = (angle, 0.0) if r[2, 2] > 0 else (0.0, angle)
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
