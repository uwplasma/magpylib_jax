"""Sphinx configuration for magpylib_jax."""

from __future__ import annotations

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath("../src"))

from magpylib_jax import __version__

project = "magpylib_jax"
author = "uwplasma"
copyright = f"{datetime.now().year}, {author}"
release = __version__

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_static_path = ["_static"]

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
]
myst_heading_anchors = 3
# Render $...$ and $$...$$ as math (MathJax) rather than literal text.
myst_dmath_double_inline = True

autodoc_typehints = "description"
autosectionlabel_prefix_document = True
