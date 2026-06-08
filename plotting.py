"""Plotting helpers for the topological CATE pipeline."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation as MplTri

from .triangulation import Triangulation2D
from .landscape import PersistenceLandscape


# --------------------------------------------------------------------------- #
# Functions on the covariate space
# --------------------------------------------------------------------------- #


def plot_function_2d(
    f: Callable[[np.ndarray], np.ndarray],
    domain: np.ndarray = np.array([[0.0, 1.0], [0.0, 1.0]]),
    resolution: int = 200,
    title: str = "",
    ax: Optional[plt.Axes] = None,
    levels: int = 20,
    cmap: str = "viridis",
) -> plt.Axes:
    (x0, x1), (y0, y1) = domain
    xs = np.linspace(x0, x1, resolution)
    ys = np.linspace(y0, y1, resolution)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel()])
    ZZ = f(pts).reshape(resolution, resolution)

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    im = ax.contourf(XX, YY, ZZ, levels=levels, cmap=cmap)
    plt.colorbar(im, ax=ax)
    ax.set_aspect("equal")
    ax.set_title(title)
    return ax


def plot_pl_function(
    nodal_values: np.ndarray,
    mesh: Triangulation2D,
    title: str = "",
    ax: Optional[plt.Axes] = None,
    cmap: str = "viridis",
    show_mesh: bool = False,
) -> plt.Axes:
    """Plot a PL function on a 2D triangulation."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    mtri = MplTri(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.simplices)
    im = ax.tricontourf(mtri, nodal_values, levels=20, cmap=cmap)
    plt.colorbar(im, ax=ax)
    if show_mesh:
        ax.triplot(mtri, color="w", lw=0.2, alpha=0.5)
    ax.set_aspect("equal")
    ax.set_title(title)
    return ax


# --------------------------------------------------------------------------- #
# Persistence diagrams and landscapes
# --------------------------------------------------------------------------- #


def plot_diagram(
    diagram: np.ndarray,
    title: str = "Persistence diagram",
    ax: Optional[plt.Axes] = None,
    in_f_coords: bool = True,
) -> plt.Axes:
    """Plot a persistence diagram. Use ``in_f_coords=False`` for sublevel coords."""
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))

    finite = np.isfinite(diagram).all(axis=1)
    pts_fin = diagram[finite]
    pts_inf = diagram[~finite]

    if pts_fin.size:
        ax.scatter(pts_fin[:, 0], pts_fin[:, 1], s=25, alpha=0.8)

    if in_f_coords:
        # superlevel: birth >= death; "diagonal" is birth = death; persistence = b - d
        if pts_fin.size:
            lo = float(pts_fin.min())
            hi = float(pts_fin.max())
        else:
            lo, hi = 0.0, 1.0
        span = max(hi - lo, 1e-3)
        lo -= 0.05 * span
        hi += 0.05 * span
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlabel("birth (threshold $t$, superlevel)")
        ax.set_ylabel("death")
        # for finite-inf points (deaths at -inf in f-coords), draw at lower edge
        if pts_inf.size:
            ax.scatter(pts_inf[:, 0], np.full(pts_inf.shape[0], lo),
                       marker="^", color="red", s=40, label="infinite")
            ax.legend()
    else:
        if pts_fin.size:
            lo = float(pts_fin.min())
            hi = float(pts_fin.max())
        else:
            lo, hi = 0.0, 1.0
        span = max(hi - lo, 1e-3)
        lo -= 0.05 * span
        hi += 0.05 * span
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlabel("birth (sublevel $-f$)")
        ax.set_ylabel("death")
        if pts_inf.size:
            ax.scatter(pts_inf[:, 0], np.full(pts_inf.shape[0], hi),
                       marker="^", color="red", s=40, label="infinite")
            ax.legend()

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return ax


def plot_landscape(
    landscape: PersistenceLandscape,
    max_layers: int = 5,
    title: str = "Persistence landscape",
    ax: Optional[plt.Axes] = None,
    *,
    amplitude_minmax: Optional[float] = None,
    amplitude_bootstrap: Optional[float] = None,
    eps_n: Optional[float] = None,
) -> plt.Axes:
    """Plot a persistence landscape with optional scale-reference overlays.

    Parameters
    ----------
    landscape : :class:`PersistenceLandscape`
        Landscape to render.
    max_layers : int
        Maximum number of landscape layers (``lambda_k``) to draw.
    title, ax : standard matplotlib hooks.
    amplitude_minmax : float, optional
        ``max - min`` amplitude of ``tau_hat`` on a grid.  When provided,
        a red dashed horizontal line is drawn at ``amplitude_minmax / 2``
        (the theoretical landscape ceiling).
    amplitude_bootstrap : float, optional
        Robust bootstrap-based amplitude (e.g. mean of ``Q_99 - Q_01`` across
        bootstrap replicates).  When provided, a blue dotted horizontal line
        is drawn at ``amplitude_bootstrap / 2``.
    eps_n : float, optional
        Bootstrap sup-discrepancy estimate of ``|| tau_hat - E tau_hat ||_inf``.
        When provided, the band ``[0, eps_n]`` is shaded in grey -- tents
        not exceeding this band are likely attributable to estimator variance
        rather than genuine topology.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.5))

    values = landscape.values
    
    # noise band first so that landscape lines sit on top of it
    if eps_n is not None and eps_n > 0:
        ax.axhspan(
            0.0,
            float(eps_n),
            color="lightgray",
            alpha=0.5,
            label=f"noise band ($\\varepsilon_n={eps_n:.3g}$)",
            zorder=0,
        )

    K = min(max_layers, landscape.n_layers)
    for k in range(K):
        if np.any(values[k] > 0):
            ax.plot(landscape.grid, values[k], label=f"$\\lambda_{{{k + 1}}}$")

    # amplitude reference lines: ceiling for lambda_k is amplitude/2
    if amplitude_minmax is not None:
        ceiling = float(amplitude_minmax) / 2.0
        ax.axhline(
            ceiling,
            linestyle="--",
            color="tab:red",
            linewidth=1.2,
            label=f"min-max amp/2 = {ceiling:.3g}",
        )
    if amplitude_bootstrap is not None:
        ceiling_b = float(amplitude_bootstrap) / 2.0
        ax.axhline(
            ceiling_b,
            linestyle=":",
            color="tab:blue",
            linewidth=1.2,
            label=f"bootstrap amp/2 = {ceiling_b:.3g}",
        )

    ax.set_xlabel("$t$ (in $-f$ coords)")
    ax.set_ylabel("$\\lambda_k(t)$")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return ax
