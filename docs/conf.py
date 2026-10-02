# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Sphinx configuration for pyippdme."""

from __future__ import annotations

from pyippdme import __version__

project = "pyippdme"
copyright = "2026, Thomas Ascher"
author = "Thomas Ascher"
release = __version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

autodoc_member_order = "bysource"
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}
# Some names are legitimately importable at two paths (e.g. ServerError, both
# pyippdme.ServerError and pyippdme.protocol.errors.ServerError); Sphinx warns
# on the resulting ambiguous unqualified reference when resolving a type hint
# for such a name, even though the reference itself resolves correctly.
suppress_warnings = ["ref.python"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

html_theme = "furo"
