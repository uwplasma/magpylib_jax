"""Top-level names that ease dropping ``magpylib_jax`` in for ``magpylib``.

Import ``magpylib_jax as magpy`` and most field-computation scripts run
unchanged: the source classes, ``Collection``, ``Sensor``, ``getB/getH/getJ/getM``,
``getFT``, ``show``, and ``mu_0`` all match magpylib's public surface. This module
supplies the few remaining top-level names. Display/config internals
(``graphics``, the full ``defaults`` tree, ``func``, ``core`` field functions) are
only partially shimmed — see the compatibility notes in the docs.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

from magpylib_jax.constants import MU0

#: Vacuum permeability in SI (T·m/A), matching ``scipy.constants.mu_0``.
mu_0 = float(MU0)

#: Plotting backends supported by :func:`magpylib_jax.show` (matplotlib only).
SUPPORTED_PLOTTING_BACKENDS = ("matplotlib",)

#: Minimal stand-in for ``magpylib.defaults``. Common code reads/sets
#: ``defaults.display.backend``; deeper attributes are not modelled.
defaults = SimpleNamespace(
    display=SimpleNamespace(backend="matplotlib"),
)


@contextmanager
def show_context(*args, **kwargs):
    """Compatibility no-op for ``magpylib.show_context``.

    magpylib batches ``show()`` calls made inside this context into one figure;
    here each ``show()`` renders independently. The context yields so existing
    ``with magpy.show_context(): ...`` blocks run without error.
    """
    yield None
