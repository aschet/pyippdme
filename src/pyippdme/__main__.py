# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Lets ``python -m pyippdme`` run the ``ippdme`` CLI without it being installed as a script."""

from pyippdme.cli.main import main

if __name__ == "__main__":
    main()
