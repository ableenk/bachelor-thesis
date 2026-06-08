"""Shape-regular triangulation of a 2D covariate domain and PL interpolation.

For the geometric layer of the thesis we approximate a function f : X -> R
by its piecewise-linear (P1) interpolant on a triangulation K_h, where

    f_h |_simplex   is affine,
    f_h(v_j) = f(v_j)  for every vertex v_j of K_h.

For low-dimensional X = [0, 1]^d (the regime stressed in the thesis) we use a
uniform structured triangulation, which is automatically shape-regular and
whose mesh size h equals the grid step. Optionally we also expose a
Delaunay-based triangulation of arbitrary 2D point clouds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.spatial import Delaunay


# --------------------------------------------------------------------------- #
# Triangulation container
# --------------------------------------------------------------------------- #


@dataclass
class Triangulation2D:
    """Container for a 2D triangulation.

    Attributes
    ----------
    vertices  : (V, 2) ndarray of vertex coordinates.
    simplices : (T, 3) ndarray of vertex indices for each triangle.
    edges     : (E, 2) ndarray of *unique* edges (i < j), used for lower-star
                filtration assembly.
    h         : mesh size (longest edge length).
    """

    vertices: np.ndarray
    simplices: np.ndarray
    edges: np.ndarray
    h: float

    @property
    def n_vertices(self) -> int:
        return self.vertices.shape[0]


def _build_unique_edges(simplices: np.ndarray) -> np.ndarray:
    """Extract the sorted unique edges of a 2D triangulation."""
    a = simplices[:, [0, 1]]
    b = simplices[:, [1, 2]]
    c = simplices[:, [0, 2]]
    edges = np.vstack([a, b, c])
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    return edges


def _max_edge_length(vertices: np.ndarray, edges: np.ndarray) -> float:
    if edges.size == 0:
        return 0.0
    diffs = vertices[edges[:, 0]] - vertices[edges[:, 1]]
    return float(np.linalg.norm(diffs, axis=1).max())


# --------------------------------------------------------------------------- #
# Uniform structured triangulation of a 2D box
# --------------------------------------------------------------------------- #


def uniform_triangulation_2d(
    n_per_side: int,
    domain: np.ndarray = np.array([[0.0, 1.0], [0.0, 1.0]]),
) -> Triangulation2D:
    """Uniform structured triangulation of ``[x0, x1] x [y0, y1]``.

    Each cell of an ``n_per_side x n_per_side`` grid is split into two triangles
    by the diagonal ``(i, j)-(i+1, j+1)``. The resulting family
    ``{K_h}`` with ``h = max(dx, dy) * sqrt(2)`` is shape-regular.
    """
    if n_per_side < 2:
        raise ValueError("n_per_side must be at least 2")

    (x0, x1), (y0, y1) = domain[0], domain[1]
    xs = np.linspace(x0, x1, n_per_side)
    ys = np.linspace(y0, y1, n_per_side)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    vertices = np.column_stack([XX.ravel(), YY.ravel()])

    def vid(i: int, j: int) -> int:
        return j * n_per_side + i

    tris = []
    for j in range(n_per_side - 1):
        for i in range(n_per_side - 1):
            v00 = vid(i, j)
            v10 = vid(i + 1, j)
            v01 = vid(i, j + 1)
            v11 = vid(i + 1, j + 1)
            tris.append([v00, v10, v11])
            tris.append([v00, v11, v01])
    simplices = np.asarray(tris, dtype=np.int64)
    edges = _build_unique_edges(simplices)
    h = _max_edge_length(vertices, edges)
    return Triangulation2D(vertices=vertices, simplices=simplices, edges=edges, h=h)


# --------------------------------------------------------------------------- #
# Delaunay triangulation of an arbitrary 2D point cloud
# --------------------------------------------------------------------------- #


def delaunay_triangulation_2d(points: np.ndarray) -> Triangulation2D:
    """Delaunay triangulation of a 2D point cloud (useful for sample-based meshes)."""
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must be of shape (N, 2)")
    tri = Delaunay(points)
    simplices = tri.simplices.astype(np.int64)
    edges = _build_unique_edges(simplices)
    h = _max_edge_length(points, edges)
    return Triangulation2D(vertices=np.asarray(points, dtype=np.float64),
                           simplices=simplices, edges=edges, h=h)


# --------------------------------------------------------------------------- #
# Piecewise-linear interpolation on a triangulation
# --------------------------------------------------------------------------- #


def pl_interpolate(
    f: Callable[[np.ndarray], np.ndarray],
    mesh: Triangulation2D,
) -> np.ndarray:
    """Evaluate ``f`` at the vertices of ``mesh`` to define its PL interpolant.

    Returns the array of nodal values ``(f_h(v_j))_j``. The PL function ``f_h``
    is fully determined by these nodal values together with ``mesh.simplices``.
    """
    return np.asarray(f(mesh.vertices), dtype=np.float64)


def pl_evaluate(
    nodal_values: np.ndarray,
    mesh: Triangulation2D,
    X: np.ndarray,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Evaluate a PL function (specified by its nodal values) at points ``X``.

    Points outside the convex hull of the mesh receive ``fill_value``.
    """
    tri = Delaunay(mesh.vertices)
    simplex_idx = tri.find_simplex(X)
    out = np.full(X.shape[0], fill_value, dtype=np.float64)

    inside = simplex_idx >= 0
    if not np.any(inside):
        return out

    # barycentric coordinates inside each found simplex
    s_idx = simplex_idx[inside]
    transform = tri.transform[s_idx]                       # (m, 3, 2)
    b = np.einsum("mij,mj->mi", transform[:, :2, :], X[inside] - transform[:, 2, :])
    bary = np.column_stack([b, 1.0 - b.sum(axis=1)])       # (m, 3)
    verts = tri.simplices[s_idx]                           # (m, 3)
    out[inside] = np.einsum("mi,mi->m", bary, nodal_values[verts])
    return out


def sup_norm_error(
    f: Callable[[np.ndarray], np.ndarray],
    nodal_values: np.ndarray,
    mesh: Triangulation2D,
    n_test: int = 5000,
    seed: Optional[int] = 0,
) -> float:
    """Monte-Carlo estimate of ``||f_h - f||_inf`` on the convex hull of the mesh.

    Mainly used for empirical verification of the ``h^alpha`` interpolation
    error bound stated in Section 4.1 of the thesis.
    """
    rng = np.random.default_rng(seed)
    lo = mesh.vertices.min(axis=0)
    hi = mesh.vertices.max(axis=0)
    Xtest = rng.uniform(lo, hi, size=(n_test, mesh.vertices.shape[1]))
    fh = pl_evaluate(nodal_values, mesh, Xtest, fill_value=np.nan)
    valid = np.isfinite(fh)
    if not np.any(valid):
        return float("nan")
    return float(np.max(np.abs(fh[valid] - f(Xtest[valid]))))
