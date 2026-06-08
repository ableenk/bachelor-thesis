"""Persistence landscapes and their L_p norms (Bubenik, 2015).

Given a persistence diagram ``Dgm = {(b_i, d_i)}_i`` with ``b_i < d_i``, define
the **tent function** of the i-th point as

    lambda_{(b, d)}(t) = max(0, min(t - b, d - t)),     t in R.

For each ``t in R``, order the values ``{lambda_{(b_i, d_i)}(t)}_i`` in
decreasing order; the ``k``-th value is denoted ``lambda_k^f(t)``. The sequence

    Lambda_q(f) = { lambda_k^f }_{k >= 1}

is the **persistence landscape** of ``f`` in degree ``q``. We compute it on a
user-supplied grid ``t in T`` and equip it with the discrete ``L_p`` norms

    ||Lambda(f) - Lambda(g)||_p
        = ( sum_{k} sum_{t in T} | lambda_k^f(t) - lambda_k^g(t) |^p * dt )^{1/p},
    ||Lambda(f) - Lambda(g)||_inf
        = sup_{k, t} | lambda_k^f(t) - lambda_k^g(t) |.

By design the landscape is a 1-Lipschitz function of the diagram, so the
deterministic stability targeted in the thesis
``||Lambda_0(f) - Lambda_0(g)||_p <= C_p ||f - g||_inf`` becomes amenable
to direct empirical verification.

Conventions
-----------
Diagrams are expected in **(-f)-coordinates** (sublevel filtration, ``b < d``).
A helper is provided to convert from f-coordinates if needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# --------------------------------------------------------------------------- #
# Coordinate convention helpers
# --------------------------------------------------------------------------- #


def diagram_f_to_neg(dgm_f: np.ndarray) -> np.ndarray:
    """Convert a diagram from ``f``-coords (birth >= death) to ``-f``-coords (b < d)."""
    out = np.column_stack([-dgm_f[:, 0], -dgm_f[:, 1]])
    # ensure b < d (swap any defective rows just in case)
    bad = out[:, 0] > out[:, 1]
    if np.any(bad):
        out[bad] = out[bad][:, ::-1]
    return out


def diagram_neg_to_f(dgm_neg: np.ndarray) -> np.ndarray:
    """Convert a diagram from ``-f``-coords back to ``f``-coords."""
    return np.column_stack([-dgm_neg[:, 0], -dgm_neg[:, 1]])


# --------------------------------------------------------------------------- #
# Landscape evaluation on a grid
# --------------------------------------------------------------------------- #


@dataclass
class PersistenceLandscape:
    """Persistence landscape sampled on a regular grid.

    Attributes
    ----------
    values : (K, T) ndarray
        ``values[k - 1, j] = lambda_k(t_j)`` for ``k = 1, ..., K``.
    grid   : (T,) ndarray
        Sample points ``t_j``.
    """

    values: np.ndarray
    grid: np.ndarray

    @property
    def n_layers(self) -> int:
        return self.values.shape[0]


def compute_landscape(
    diagram: np.ndarray,
    grid: np.ndarray,
    n_layers: Optional[int] = None,
    drop_infinite: bool = True,
) -> PersistenceLandscape:
    """Compute the persistence landscape on a grid.

    Parameters
    ----------
    diagram      : (m, 2) array in ``(-f)``-coordinates with ``b_i < d_i``.
                   Infinite-death points are silently dropped if ``drop_infinite``.
    grid         : 1D array of evaluation points (sorted ascending).
    n_layers     : how many landscape layers ``lambda_k`` to return.
                   Defaults to ``len(diagram)`` (with infinities dropped).
    drop_infinite: drop points with non-finite birth/death.
    """
    dgm = np.asarray(diagram, dtype=np.float64).reshape(-1, 2)
    if drop_infinite:
        dgm = dgm[np.isfinite(dgm).all(axis=1)]

    grid = np.asarray(grid, dtype=np.float64)
    T = grid.shape[0]

    if dgm.shape[0] == 0:
        K = n_layers or 1
        return PersistenceLandscape(values=np.zeros((K, T)), grid=grid)

    # (m, T) tent matrix:  L_{i, j} = max(0, min(t_j - b_i, d_i - t_j))
    b = dgm[:, 0:1]
    d = dgm[:, 1:2]
    left = grid[None, :] - b
    right = d - grid[None, :]
    tents = np.maximum(0.0, np.minimum(left, right))   # (m, T)

    # sort each column in decreasing order
    tents_sorted = np.sort(tents, axis=0)[::-1, :]     # (m, T)

    m = tents_sorted.shape[0]
    K = n_layers if n_layers is not None else m
    if K <= m:
        values = tents_sorted[:K, :]
    else:
        values = np.zeros((K, T))
        values[:m, :] = tents_sorted
    return PersistenceLandscape(values=values, grid=grid)


# --------------------------------------------------------------------------- #
# Norms
# --------------------------------------------------------------------------- #


def landscape_norm(L: PersistenceLandscape, p: float = 2.0) -> float:
    """Discrete ``L_p`` norm of a landscape ``Lambda`` (treated as an element of L^p(R x N))."""
    dt = _grid_dt(L.grid)
    if np.isinf(p):
        return float(np.max(np.abs(L.values))) if L.values.size else 0.0
    return float(((np.abs(L.values) ** p).sum() * dt) ** (1.0 / p))


def landscape_distance(
    L1: PersistenceLandscape, L2: PersistenceLandscape, p: float = 2.0
) -> float:
    """Discrete ``L_p`` distance between two landscapes on a *common* grid."""
    if L1.grid.shape != L2.grid.shape or not np.allclose(L1.grid, L2.grid):
        raise ValueError("landscapes must share the same evaluation grid")

    K = max(L1.n_layers, L2.n_layers)
    v1 = _pad_layers(L1.values, K)
    v2 = _pad_layers(L2.values, K)
    diff = np.abs(v1 - v2)
    dt = _grid_dt(L1.grid)
    if np.isinf(p):
        return float(diff.max()) if diff.size else 0.0
    return float(((diff ** p).sum() * dt) ** (1.0 / p))


def _grid_dt(grid: np.ndarray) -> float:
    if grid.size < 2:
        return 1.0
    return float(grid[1] - grid[0])


def _pad_layers(values: np.ndarray, K: int) -> np.ndarray:
    if values.shape[0] >= K:
        return values[:K]
    pad = np.zeros((K - values.shape[0], values.shape[1]), dtype=values.dtype)
    return np.vstack([values, pad])


# --------------------------------------------------------------------------- #
# Convenience: auto-grid from a diagram
# --------------------------------------------------------------------------- #


def auto_grid(
    diagrams: list[np.ndarray] | np.ndarray,
    n_points: int = 512,
    pad: float = 0.05,
) -> np.ndarray:
    """Build a common grid covering the finite support of one or more diagrams."""
    if isinstance(diagrams, np.ndarray):
        diagrams = [diagrams]
    finite = [d[np.isfinite(d).all(axis=1)] for d in diagrams]
    finite = [d for d in finite if d.size > 0]
    if not finite:
        return np.linspace(-1.0, 1.0, n_points)
    all_pts = np.vstack(finite)
    lo = float(all_pts.min())
    hi = float(all_pts.max())
    span = max(hi - lo, 1e-6)
    return np.linspace(lo - pad * span, hi + pad * span, n_points)
