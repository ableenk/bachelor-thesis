"""Synthetic data generators for the heterogeneous partially linear model.

The thesis works with the model

    Y = g_0(X) + A * tau_0(X) + U,        E[U | X, A] = 0,
    A | X  ~  Bernoulli(e_0(X)),

where X lives in a bounded domain Omega in R^d (here a square [0, 1]^2 by default).

We provide two data generators:

* :func:`make_topology_dgp`   - custom DGP with a controllable topology of the
  high-effect region (one blob / two blobs / annulus). Returns also the ground-truth
  ``tau_0`` so that we can compare estimated and true persistence landscapes.

* :func:`make_doubleml_data`  - thin wrapper around ``doubleml.datasets`` that
  produces a partially-linear-model dataset and packs it into our common dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional

import numpy as np


# --------------------------------------------------------------------------- #
# Common container
# --------------------------------------------------------------------------- #


@dataclass
class CateDataset:
    """Container for a generated dataset.

    Attributes
    ----------
    X : (n, d) ndarray
        Covariates.
    A : (n,) ndarray of {0, 1}
        Binary treatment indicator.
    Y : (n,) ndarray
        Observed outcome.
    tau0 : callable or None
        Ground-truth CATE function ``tau_0(X) -> (n,)``. ``None`` if unknown.
    g0 : callable or None
        Baseline outcome function ``g_0(X) -> (n,)``.
    e0 : callable or None
        Propensity score ``e_0(X) -> (n,)``.
    domain : (d, 2) ndarray
        Per-coordinate axis-aligned bounding box of the covariate space.
    """

    X: np.ndarray
    A: np.ndarray
    Y: np.ndarray
    tau0: Optional[Callable[[np.ndarray], np.ndarray]] = None
    g0: Optional[Callable[[np.ndarray], np.ndarray]] = None
    e0: Optional[Callable[[np.ndarray], np.ndarray]] = None
    domain: Optional[np.ndarray] = None

    @property
    def n(self) -> int:
        return self.X.shape[0]

    @property
    def d(self) -> int:
        return self.X.shape[1]


# --------------------------------------------------------------------------- #
# Custom DGP with controllable topology
# --------------------------------------------------------------------------- #


def _bump(x: np.ndarray, center: np.ndarray, scale: float) -> np.ndarray:
    """Gaussian bump centered at ``center`` with bandwidth ``scale``."""
    diff = x - center
    return np.exp(-0.5 * np.sum(diff ** 2, axis=-1) / scale ** 2)


def make_tau0(
    topology: Literal[
        "single", "two_blobs", "annulus", "three_blobs", "four_blobs"
    ] = "two_blobs",
    amplitude: float = 1.0,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a tau_0 function with the desired topology of its superlevel sets.

    All variants live on [0, 1]^2 and are smooth (C^infty), hence trivially
    Hoelder-regular and tame.

    Presets ``'three_blobs'`` and ``'four_blobs'`` use Gaussian bumps of
    *distinct* heights placed sufficiently far apart so that the superlevel
    sets break into the prescribed number of connected components over a
    non-trivial range of thresholds. Concretely:

    * ``'three_blobs'`` gives one infinite H_0 class plus 2 finite ones in
      Dgm_0 (and therefore 2 non-zero tents in Lambda_0) of distinct heights;
    * ``'four_blobs'`` gives one infinite H_0 class plus 3 finite ones in
      Dgm_0 (3 non-zero tents in Lambda_0) of strictly decreasing heights.
    """

    if topology == "single":
        c = np.array([0.5, 0.5])

        def tau0(X: np.ndarray) -> np.ndarray:
            return amplitude * _bump(X, c, scale=0.18)

    elif topology == "two_blobs":
        c1 = np.array([0.30, 0.35])
        c2 = np.array([0.72, 0.68])

        def tau0(X: np.ndarray) -> np.ndarray:
            return amplitude * (
                _bump(X, c1, scale=0.12) + 0.9 * _bump(X, c2, scale=0.14)
            )

    elif topology == "annulus":
        c = np.array([0.5, 0.5])

        def tau0(X: np.ndarray) -> np.ndarray:
            r = np.linalg.norm(X - c, axis=-1)
            # ring of radius ~0.3, width ~0.07
            return amplitude * np.exp(-((r - 0.3) ** 2) / (2 * 0.07 ** 2))

    elif topology == "three_blobs":
        # three well-separated bumps of distinct heights; the global maximum
        # gives the infinite H_0 class, the other two contribute one finite
        # point each to Dgm_0 with persistences ~ (0.75, 0.50) respectively.
        c1 = np.array([0.25, 0.30]); a1 = 1.00
        c2 = np.array([0.72, 0.30]); a2 = 0.75
        c3 = np.array([0.50, 0.75]); a3 = 0.50
        scale = 0.10

        def tau0(X: np.ndarray) -> np.ndarray:
            return amplitude * (
                a1 * _bump(X, c1, scale)
                + a2 * _bump(X, c2, scale)
                + a3 * _bump(X, c3, scale)
            )

    elif topology == "four_blobs":
        # four well-separated bumps of strictly decreasing heights; expect
        # one infinite H_0 class plus 3 finite ones with persistences
        # roughly (0.80, 0.60, 0.40).
        c1 = np.array([0.22, 0.22]); a1 = 1.00
        c2 = np.array([0.78, 0.22]); a2 = 0.80
        c3 = np.array([0.22, 0.78]); a3 = 0.60
        c4 = np.array([0.78, 0.78]); a4 = 0.40
        scale = 0.09

        def tau0(X: np.ndarray) -> np.ndarray:
            return amplitude * (
                a1 * _bump(X, c1, scale)
                + a2 * _bump(X, c2, scale)
                + a3 * _bump(X, c3, scale)
                + a4 * _bump(X, c4, scale)
            )

    else:
        raise ValueError(f"unknown topology={topology!r}")

    return tau0


def _logistic(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def make_topology_dgp(
    n: int = 2000,
    topology: Literal[
        "single", "two_blobs", "annulus", "three_blobs", "four_blobs"
    ] = "two_blobs",
    noise_sd: float = 0.3,
    amplitude: float = 1.0,
    seed: Optional[int] = 0,
) -> CateDataset:
    """Generate data from a heterogeneous PLM with prescribed CATE topology.

    Parameters
    ----------
    n        : sample size.
    topology : shape of the high-effect region (see :func:`make_tau0`).
    noise_sd : standard deviation of the residual U.
    amplitude: peak amplitude of tau_0.
    seed     : RNG seed.
    """
    rng = np.random.default_rng(seed)

    X = rng.uniform(0.0, 1.0, size=(n, 2))

    tau0 = make_tau0(topology=topology, amplitude=amplitude)

    def g0(x: np.ndarray) -> np.ndarray:
        # smooth nonlinear baseline -- independent of tau0's geometry
        return np.sin(2 * np.pi * x[:, 0]) * np.cos(2 * np.pi * x[:, 1]) * 0.5

    def e0(x: np.ndarray) -> np.ndarray:
        # mildly heterogeneous propensity bounded away from {0, 1}
        return _logistic(0.5 * (x[:, 0] - 0.5) + 0.3 * (x[:, 1] - 0.5))

    pi = e0(X)
    A = rng.binomial(1, pi).astype(np.float64)
    U = rng.normal(0.0, noise_sd, size=n)
    Y = g0(X) + A * tau0(X) + U

    domain = np.array([[0.0, 1.0], [0.0, 1.0]])
    return CateDataset(X=X, A=A, Y=Y, tau0=tau0, g0=g0, e0=e0, domain=domain)


# --------------------------------------------------------------------------- #
# DoubleML wrapper
# --------------------------------------------------------------------------- #


def make_doubleml_data(
    n: int = 2000,
    dim_x: int = 5,
    seed: Optional[int] = 0,
) -> CateDataset:
    """Generate a partially-linear dataset via ``doubleml.datasets.make_plr_CCDDHNR2018``.

    Note
    ----
    The default DoubleML PLR generator imposes a *constant* treatment effect
    (typically ``alpha = 0.5``), so ``tau0`` is a constant function. We expose it
    here mostly to demonstrate compatibility with the DoubleML ecosystem; for
    studying topological summaries of *heterogeneous* effects, prefer
    :func:`make_topology_dgp`.

    The underlying CCDDHNR2018 DGP accesses ``x[:, 0]``, ``x[:, 1]`` and
    ``x[:, 2]`` and therefore requires ``dim_x >= 3``; we enforce this here
    with a clear error message instead of letting the upstream call raise
    an unhelpful ``IndexError``.
    """
    if dim_x < 3:
        raise ValueError(
            f"make_doubleml_data requires dim_x >= 3 (got {dim_x}); the "
            "underlying DoubleML CCDDHNR2018 DGP uses the first three covariates."
        )

    try:
        from doubleml.datasets import make_plr_CCDDHNR2018  # doubleml < 0.10
    except ImportError:                                      # doubleml >= 0.10
        from doubleml.plm.datasets import make_plr_CCDDHNR2018

    if seed is not None:
        np.random.seed(seed)

    alpha = 0.5
    df = make_plr_CCDDHNR2018(
        n_obs=n,
        dim_x=dim_x,
        alpha=alpha,
        return_type="DataFrame",
    )

    x_cols = [c for c in df.columns if c.startswith("X")]
    X = df[x_cols].to_numpy(dtype=np.float64)
    A = df["d"].to_numpy(dtype=np.float64)
    Y = df["y"].to_numpy(dtype=np.float64)

    def tau0_const(x: np.ndarray) -> np.ndarray:
        return np.full(x.shape[0], alpha, dtype=np.float64)

    lo = X.min(axis=0)
    hi = X.max(axis=0)
    domain = np.stack([lo, hi], axis=1)

    return CateDataset(
        X=X, A=A, Y=Y, tau0=tau0_const, g0=None, e0=None, domain=domain
    )
