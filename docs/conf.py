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
    "sphinx.ext.intersphinx",
    "sphinx_design",
    "sphinx_copybutton",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "jax": ("https://docs.jax.dev/en/latest", None),
}
# Don't fail the build if an intersphinx inventory is unreachable.
intersphinx_disabled_reftypes = ["*"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = "magpylib_jax"
html_static_path = ["_static"]
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#2f6fd0",
        "color-brand-content": "#2f6fd0",
    },
    "dark_css_variables": {
        "color-brand-primary": "#6ea8fe",
        "color-brand-content": "#6ea8fe",
    },
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/uwplasma/magpylib_jax",
            "html": "",
            "class": "fa-brands fa-github",
        },
    ],
}

# Strip the >>> / ... and $ prompts when copying code blocks.
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

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
# Only label pages/sections down to h2, so repeated h3 headings (e.g. the many
# "### Added"/"### Fixed" in the changelog) don't collide.
autosectionlabel_maxdepth = 2
