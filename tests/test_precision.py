"""Precision contract: the library follows the user's JAX x64 setting.

magpylib_jax must NOT mutate the global ``jax.config`` on import and must not
hard-code float64. By default (x64 off) fields are float32 with no truncation
warning; enabling x64 gives float64 parity with magpylib. These run in
subprocesses because ``conftest.py`` enables x64 for the rest of the suite.
"""

import subprocess
import sys
import textwrap


def _run(code: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_import_does_not_enable_x64_and_is_warning_free():
    out = _run(
        """
        import warnings
        warnings.simplefilter("error")  # any float64-truncation warning -> failure
        import jax
        import magpylib_jax as mpj
        assert jax.config.jax_enable_x64 is False, "import must not enable x64"
        b = mpj.magnet.Cuboid(polarization=(0, 0, 1.0), dimension=(1, 1, 1)).getB((2.0, 0, 0))
        assert b.dtype == jax.numpy.float32, b.dtype
        print("OK", b.dtype)
        """
    )
    assert out == "OK float32"


def test_x64_opt_in_gives_float64():
    out = _run(
        """
        import jax
        jax.config.update("jax_enable_x64", True)
        import magpylib_jax as mpj
        b = mpj.magnet.Cuboid(polarization=(0, 0, 1.0), dimension=(1, 1, 1)).getB((2.0, 0, 0))
        assert b.dtype == jax.numpy.float64, b.dtype
        print("OK", b.dtype)
        """
    )
    assert out == "OK float64"
