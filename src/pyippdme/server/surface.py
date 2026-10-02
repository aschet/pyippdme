# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""The pluggable interface between ``PtMeas`` and a real (or synthetic) part.

Without a surface, ``PtMeas`` (6.12.1) simply reports whatever
position the client commanded - there is nothing for the probe to actually
touch, so approach/search/retract distances and the ``IJK`` probing
direction are accepted but have no effect (see
:mod:`pyippdme.simulation.classes.cartcmm_class`'s module docstring).
:class:`SampleSurface` is the seam that changes that: a
:class:`~pyippdme.server.IppDmeServer`/:class:`~pyippdme.simulation.virtual_cmm.VirtualCMM`
configured with one gets a real, if synthetic, part for ``PtMeas`` to
measure against - a measured point can genuinely diverge from the commanded
one, and "no surface found" becomes a real, reachable error (``1006``)
instead of something that can never happen. This module holds only the
interface itself - the concrete geometry (planes, spheres, cylinders) lives
in :mod:`pyippdme.simulation.surface`, which depends on this module, not the
other way around; a real integrator with an actual CAD/mesh representation
of the part implements this same one-method Protocol directly instead.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pyippdme.types.vec3 import Vec3


@runtime_checkable
class SampleSurface(Protocol):
    """A synthetic surface a probing ray can intersect (structural - no inheritance needed).

    ``direction`` need not be normalized. Returns the nearest intersection
    point with ``t >= 0`` along ``direction`` from ``origin``, or ``None``
    if the ray never meets the surface (including when the surface is
    entirely behind ``origin``). Implementations don't need to know or care
    how far the caller intends to search - that bound is applied by the
    caller (:mod:`pyippdme.simulation.classes.cartcmm_class`), against
    ``Tool``/``Part``'s ``Approach``/``Search`` properties.
    """

    def intersect(self, origin: Vec3, direction: Vec3) -> Vec3 | None: ...
