"""Persistent homology of superlevel sets of a PL function on a triangulation.

Following the convention of the thesis we encode the **superlevel filtration**
of ``f`` as the **sublevel filtration of ``-f``** (Section 2.5). Concretely:

* every vertex ``v`` enters the filtration at time ``-f(v)``;
* every edge ``[u, v]`` enters at time ``max(-f(u), -f(v))``
  i.e. the lower-star filtration value of the edge w.r.t. ``-f``.

This is exactly the **lower-star filtration** of ``-f`` on the 1-skeleton of
the triangulation, which fully determines persistent homology in degree
``q = 0`` (connected components) -- the main case studied in the thesis.

We compute persistence using :mod:`ripser`, which accepts a (sparse) distance
matrix whose diagonal entries are vertex births and off-diagonal entries are
edge births. The output diagram lives in (-f)-coordinates; we additionally
provide a helper that flips the sign back to ``f``-coordinates if the user
prefers to think in terms of superlevel-set thresholds ``t = -s``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.sparse import coo_matrix
from ripser import ripser

from .triangulation import Triangulation2D


# --------------------------------------------------------------------------- #
# Lower-star filtration via sparse distance matrix
# --------------------------------------------------------------------------- #


def _lower_star_sparse_matrix(
    nodal_neg_f: np.ndarray, edges: np.ndarray
) -> coo_matrix:
    """Build a sparse "distance" matrix encoding the lower-star filtration of -f.

    Parameters
    ----------
    nodal_neg_f : (V,) array of vertex births (here, values of -f at the vertices).
    edges       : (E, 2) array of unique mesh edges.

    Returns
    -------
    A (V, V) sparse COO matrix where:
      * diagonal entries are vertex births,
      * off-diagonal entries are edge births = max of the two endpoint births.
    """
    V = nodal_neg_f.shape[0]
    if edges.size == 0:
        rows = np.arange(V)
        cols = np.arange(V)
        vals = nodal_neg_f
    else:
        edge_vals = np.maximum(nodal_neg_f[edges[:, 0]], nodal_neg_f[edges[:, 1]])
        rows = np.concatenate([np.arange(V), edges[:, 0], edges[:, 1]])
        cols = np.concatenate([np.arange(V), edges[:, 1], edges[:, 0]])
        vals = np.concatenate([nodal_neg_f, edge_vals, edge_vals])
    return coo_matrix((vals, (rows, cols)), shape=(V, V))


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def persistence_h0_superlevel(
    nodal_values: np.ndarray,
    mesh: Triangulation2D,
    return_in_f_coords: bool = True,
) -> np.ndarray:
    """Persistence diagram in degree 0 of the **superlevel filtration** of ``f``.

    Parameters
    ----------
    nodal_values : (V,) array of nodal values ``f(v_j)`` defining the PL function.
    mesh         : underlying triangulation; only ``mesh.edges`` is used (q = 0
                   homology is determined by the 1-skeleton).
    return_in_f_coords : if True, return the diagram in the original
                   ``f``-coordinates as ``[(birth_f, death_f), ...]``. With the
                   superlevel convention, a feature is *born* at a high threshold
                   ``t`` and *dies* at a lower threshold, so ``birth_f >= death_f``.
                   If False, the diagram is returned in ``(-f)``-coordinates
                   (``birth_{-f} <= death_{-f}``), matching ripser's raw output.

    Notes
    -----
    Infinite death times (the persistent component of the whole connected
    domain) are preserved as ``+inf`` (in ``-f``-coords) or ``-inf`` (in
    ``f``-coords).
    """
    neg = -np.asarray(nodal_values, dtype=np.float64)
    M = _lower_star_sparse_matrix(neg, mesh.edges)
    res = ripser(M, distance_matrix=True, maxdim=0)
    dgm = res["dgms"][0]  # (n_pts, 2) array in (-f)-coords

    if not return_in_f_coords:
        return np.asarray(dgm, dtype=np.float64)

    # Flip back: t = -s, so a (birth_s, death_s) with birth_s <= death_s
    # becomes (birth_t, death_t) = (-birth_s, -death_s) with birth_t >= death_t.
    out = np.column_stack([-dgm[:, 0], -dgm[:, 1]])
    return out


def diagram_finite_part(dgm: np.ndarray) -> np.ndarray:
    """Drop points with infinite death (or infinite birth in flipped coords)."""
    finite = np.isfinite(dgm).all(axis=1)
    return dgm[finite]


def bottleneck_distance(dgm1: np.ndarray, dgm2: np.ndarray) -> float:
    """Bottleneck distance between two diagrams (thin wrapper around ``persim``)."""
    from persim import bottleneck

    return float(bottleneck(diagram_finite_part(dgm1), diagram_finite_part(dgm2)))
