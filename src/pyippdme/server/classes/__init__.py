# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""The two command classes VDMA 8722 mandates for every conformant server (6.3/6.4).

Each module here exposes a single ``register(registry)`` function that adds
its commands to a :class:`~pyippdme.server.registry.CommandRegistry`. Command names
are flat on the wire (the standard does not prefix them by class), so these
simply add handlers to the same registry regardless of which class formally
defines the command in the standard.

:mod:`server_class` (``Server``: session lifecycle, error handling, and the
generic ``SetProp``/``GetProp``/``EnumProp``/``EnumAllProp`` property
dispatch other classes plug into) and :mod:`dme_class` (``DME``:
``GetDMEVersion``/``GetMachineClass``/``GetSupportedCommands``/
``GetSupportedArguments``) are the two classes every conformant server must
support. Every other built-in class (``CartCMM``/``Tool``/``Scanning``/...)
lives in :mod:`pyippdme.simulation.classes`.
"""

from pyippdme.server.classes.dme_class import register as register_dme_class
from pyippdme.server.classes.server_class import register as register_server_class

__all__ = [
    "register_dme_class",
    "register_server_class",
]
