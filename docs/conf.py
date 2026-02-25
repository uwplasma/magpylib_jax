"""Sphinx configuration for magpylib_jax."""

from __future__ import annotations

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath("../src"))

project = "magpylib_jax"
author = "uwplasma"
copyright = f"{datetime.now().year}, {author}"
release = "0.1.0"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_static_path = ["_static"]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

autodoc_typehints = "description"
