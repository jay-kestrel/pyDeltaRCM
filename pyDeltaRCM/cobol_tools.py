"""ctypes adapter for the COBOL weighted-pick kernel (the ported routine).

The COBOL kernel is a pure numeric function of the buffers it receives:
a batched weighted pick -- the arithmetic inside
:func:`pyDeltaRCM.shared_tools.random_pick` (float32 cumulative weights
followed by a searchsorted-left scan) -- over ``k`` rows of ``n`` weights
each. Random numbers are never drawn in COBOL (port contract C5): the
Python layer draws one uniform per row from the checkpointed MT19937
stream exactly as the jitted reference does (``uniform(0, row_total)``)
and hands the kernel raw doubles.

Routing is explicit: setting ``PYDELTARCM_COBOL=1`` selects this backend
for the ported routine and takes precedence over the shared
``PYDELTARCM_NATIVE`` variable (which continues to govern the fused Rust
water phase). Setting it without a built shared object raises
``FileNotFoundError`` with build instructions rather than silently
falling back to the Python path (port contract C2).
"""
from __future__ import annotations

import ctypes
import os
import subprocess
from pathlib import Path

import numpy as np

from pyDeltaRCM.shared_tools import get_random_uniform

_ENV_VAR = "PYDELTARCM_COBOL"
_LIB_NAME = "pydeltarcm_cobol.so"
# the D8+center weight window is 9; the kernel's cutoff storage is
# sized for this bound
_MAX_COLS = 64

_f32p = ctypes.POINTER(ctypes.c_float)
_f64p = ctypes.POINTER(ctypes.c_double)
_i32p = ctypes.POINTER(ctypes.c_int32)
_lib_cache = None


def cobol_active():
    """True when the COBOL backend is selected via the environment."""
    return os.environ.get(_ENV_VAR) == "1"


def pick_backend():
    """Backend selected for the ported pick: ``"cobol"`` or ``"python"``.

    ``PYDELTARCM_COBOL=1`` takes precedence over the shared
    ``PYDELTARCM_NATIVE`` dispatch variable.
    """
    return "cobol" if cobol_active() else "python"


def library_path():
    """Where the built kernel shared object is expected."""
    return Path(__file__).parent / "lib" / _LIB_NAME


def gnucobol_version():
    """GnuCOBOL version string for benchmark provenance, or None."""
    try:
        out = subprocess.check_output(["cobc", "--version"], text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    # First line is "cobc (GnuCOBOL) 3.2.0"; later lines carry the
    # compiler-toolchain versions, so parse the GnuCOBOL line only.
    for line in out.splitlines():
        if "(GnuCOBOL)" in line:
            return line.split()[-1]
    stripped = out.strip().splitlines()
    return stripped[0] if stripped else None


def load_library():
    """Load (once) and return the COBOL kernel shared object.

    Raises FileNotFoundError naming the build step when the backend was
    requested but not built -- never fall back silently.
    """
    global _lib_cache
    if _lib_cache is not None:
        return _lib_cache
    path = library_path()
    if not path.exists():
        raise FileNotFoundError(
            f"{_ENV_VAR}=1 but the COBOL backend is not built -- run "
            "`make -C cobol` in the repository root (requires GnuCOBOL "
            f"3.x, e.g. apt gnucobol3); expected {path}"
        )
    # RTLD_GLOBAL is load-bearing: the kernel CALLs "cobol_faddf" and
    # friends, which libcob resolves through dlsym(RTLD_DEFAULT); a
    # default RTLD_LOCAL load hides the helper symbols and the runtime
    # aborts with "module 'cobol_faddf' not found" (validated in the
    # dorado port, PR #32).
    lib = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    lib.pydeltarcm_pick_cobol.argtypes = [_i32p, _i32p, _f32p, _f64p, _i32p]
    lib.pydeltarcm_pick_cobol.restype = ctypes.c_int
    _lib_cache = lib
    return lib


def _ptr(arr, ctype):
    return arr.ctypes.data_as(ctypes.POINTER(ctype))


def _reference_pick_rows(rows, cum, uniforms):
    """NumPy reference for the batched pick (the ported routine).

    Per row: float32 cumulative weights (already computed as ``cum``),
    one ``uniform(0, total)`` draw (already drawn as ``uniforms``), then
    ``np.searchsorted`` in numpy's promoted double domain. Given identical
    cutoffs and uniforms this is bit-identical to the jitted
    ``shared_tools.random_pick``: numba's cumsum and searchsorted match
    numpy bit-for-bit and numba's MT19937 draws match numpy legacy
    RandomState in lockstep (probed on this head).
    """
    picks = np.empty(rows.shape[0], dtype=np.int32)
    for p in range(rows.shape[0]):
        total = cum[p, -1]
        if not total > 0:
            raise ValueError(
                "prob must contain a positive total probability "
                f"(row {p})"
            )
        picks[p] = np.searchsorted(cum[p], uniforms[p])
    return picks


def _cobol_pick_rows(rows, uniforms):
    """COBOL backend for the batched pick (kernel recomputes the cumsum)."""
    lib = load_library()
    k = np.array([rows.shape[0]], dtype=np.int32)
    n = np.array([rows.shape[1]], dtype=np.int32)
    flat = np.ascontiguousarray(rows.reshape(-1))
    out = np.empty(rows.shape[0], dtype=np.int32)
    rc = lib.pydeltarcm_pick_cobol(
        _ptr(k, ctypes.c_int32),
        _ptr(n, ctypes.c_int32),
        _ptr(flat, ctypes.c_float),
        _ptr(uniforms, ctypes.c_double),
        _ptr(out, ctypes.c_int32),
    )
    if rc < 0:
        # kernel encodes the offending 1-based row as -rc
        raise ValueError(
            "prob must contain a positive total probability "
            f"(row {-rc - 1})"
        )
    if rc != 0:
        raise RuntimeError(f"cobol random_pick failed, rc={rc}")
    return out


def pick_rows(prob, uniforms=None):
    """Batched weighted pick: the ported ``random_pick``, dispatched.

    ``prob`` is a ``(k, n)`` array of weight rows -- or a single ``(n,)``
    row, which returns a scalar index. Any dtype is canonicalized to
    contiguous float32 at the adapter (the model's weight fields are
    float32; no float64 promotion enters the arithmetic).

    ``uniforms`` optionally supplies the ``(k,)`` raw doubles the kernel
    consumes (port contract C5). When omitted, one uniform per row is
    drawn from the checkpointed MT19937 stream as
    ``uniform(0, row_total)`` -- the same call, in the same order, the
    jitted reference makes.

    Returns an int32 array of picked indices (0-based, searchsorted-left
    against the float32 cumulative weights). Raises ValueError when a
    row's total is not positive (NaN included), with the reference's
    message, and IndexError when a uniform exceeds every cutoff (the
    reference's out-of-bounds array index).

    Parity domain: weight rows must be non-negative with a positive
    total -- the model's weight fields are sums of squared normalized
    gradients, so the float32 cumsum is non-decreasing and
    ``searchsorted`` (binary search, which requires sorted cutoffs) is
    well-defined and bit-equivalent to the kernel's linear scan. Inputs
    violating that contract (negative weights) are outside both the
    model's and this port's parity guarantees.
    """
    rows = np.ascontiguousarray(prob, dtype=np.float32)
    single = rows.ndim == 1
    if single:
        rows = rows.reshape(1, -1)
    if rows.ndim != 2:
        raise ValueError("prob must be a 1-D or 2-D array of weight rows")
    if rows.shape[1] > _MAX_COLS:
        raise ValueError(
            f"at most {_MAX_COLS} weights per row supported "
            f"(D8 window is 9), got {rows.shape[1]}"
        )

    cum = np.cumsum(rows, axis=1)
    # validate every row before drawing so the stream is never consumed
    # with a non-finite limit (the reference aborts at the first such row)
    for p in range(rows.shape[0]):
        if not cum[p, -1] > 0:
            raise ValueError(
                "prob must contain a positive total probability "
                f"(row {p})"
            )
    if uniforms is None:
        uniforms = np.array(
            [get_random_uniform(float(cum[p, -1])) for p in range(rows.shape[0])],
            dtype=np.float64,
        )
    else:
        uniforms = np.ascontiguousarray(uniforms, dtype=np.float64)
        if uniforms.shape != (rows.shape[0],):
            raise ValueError(
                f"uniforms shape {uniforms.shape} does not match "
                f"{rows.shape[0]} weight rows"
            )

    if pick_backend() == "cobol":
        picks = _cobol_pick_rows(rows, uniforms)
    else:
        picks = _reference_pick_rows(rows, cum, uniforms)
    if np.any(picks >= rows.shape[1]):
        raise IndexError(
            "pick index out of bounds: uniform exceeded every cutoff"
        )
    if single:
        return int(picks[0])
    return picks
