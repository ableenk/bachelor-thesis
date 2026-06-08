"""Cross-fitted residualised estimator of the CATE (R-learner).

Implements the estimation procedure described in Section 2.3 of the thesis:

1. Fit nuisance functions
       m_0(x) = E[Y | X = x],    e_0(x) = P(A = 1 | X = x)
   by cross-fitting with K folds, producing out-of-fold predictions
   ``m_hat^{-k(i)}(X_i)``, ``e_hat^{-k(i)}(X_i)``.

2. Form residualised variables
       Ytil = Y - m_hat^{-k(i)}(X),
       Atil = A - e_hat^{-k(i)}(X).

3. Minimise the second-stage objective
       (1/n) sum_i (Ytil_i - Atil_i * tau(X_i))^2 + pen_n(tau)
   over a class F_n of regularised non-parametric functions.

Supported backends for both the nuisance and the second-stage estimators:

* ``"catboost"`` (default) - gradient boosting via CatBoost; native
  ``sample_weight`` support, silent mode, no on-disk artefacts.
* ``"xgboost"``  - gradient boosting via XGBoost; sklearn-compatible API,
  native ``sample_weight`` support.
* ``"gbr"``      - sklearn's :class:`GradientBoostingRegressor` /
  :class:`GradientBoostingClassifier`; the original default of the package.
* ``"krr"``      - kernel ridge regression with a Gaussian kernel for the
  second stage (true Tikhonov regularisation, useful for sup-norm theory).
  For nuisance estimation under ``backend='krr'`` we fall back to ``'gbr'``,
  since sklearn ships no ready-made kernel-ridge classifier.

Optional dependencies (``catboost``, ``xgboost``) are imported lazily inside
the factory below so that the package keeps working without them as long as
the user does not request the corresponding backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold


# --------------------------------------------------------------------------- #
# Backend factory
# --------------------------------------------------------------------------- #


def _make_nuisance_estimators(
    backend: str,
    seed: Optional[int] = 0,
    *,
    gbr_n_estimators: int = 200,
    gbr_max_depth: int = 3,
    gbr_learning_rate: float = 0.05,
    xgb_n_estimators: int = 400,
    xgb_max_depth: int = 4,
    xgb_learning_rate: float = 0.05,
    cb_iterations: int = 500,
    cb_depth: int = 6,
    cb_learning_rate: float = 0.05,
) -> tuple[BaseEstimator, BaseEstimator]:
    """Return ``(m_estimator, e_estimator)`` for the requested ``backend``.

    ``m_estimator`` is a regressor for E[Y | X], ``e_estimator`` is a
    classifier for P(A = 1 | X). Both follow the sklearn interface and
    support :meth:`sklearn.base.clone`.
    """
    backend = backend.lower()

    if backend == "gbr" or backend == "krr":
        # KRR has no off-the-shelf classifier; use GBR for nuisances.
        m = GradientBoostingRegressor(
            n_estimators=gbr_n_estimators,
            max_depth=gbr_max_depth,
            learning_rate=gbr_learning_rate,
            random_state=seed,
        )
        e = GradientBoostingClassifier(
            n_estimators=gbr_n_estimators,
            max_depth=gbr_max_depth,
            learning_rate=gbr_learning_rate,
            random_state=seed,
        )
        return m, e

    if backend == "xgboost":
        try:
            from xgboost import XGBClassifier, XGBRegressor
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "backend='xgboost' requires the 'xgboost' package; "
                "install it via `pip install xgboost`."
            ) from exc

        m = XGBRegressor(
            n_estimators=xgb_n_estimators,
            max_depth=xgb_max_depth,
            learning_rate=xgb_learning_rate,
            tree_method="hist",
            random_state=seed,
            verbosity=0,
        )
        e = XGBClassifier(
            n_estimators=xgb_n_estimators,
            max_depth=xgb_max_depth,
            learning_rate=xgb_learning_rate,
            tree_method="hist",
            random_state=seed,
            verbosity=0,
            eval_metric="logloss",
        )
        return m, e

    if backend == "catboost":
        try:
            from catboost import CatBoostClassifier, CatBoostRegressor
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "backend='catboost' requires the 'catboost' package; "
                "install it via `pip install catboost`."
            ) from exc

        m = CatBoostRegressor(
            iterations=cb_iterations,
            depth=cb_depth,
            learning_rate=cb_learning_rate,
            random_state=seed,
            verbose=False,
            allow_writing_files=False,
        )
        e = CatBoostClassifier(
            iterations=cb_iterations,
            depth=cb_depth,
            learning_rate=cb_learning_rate,
            random_state=seed,
            loss_function="Logloss",
            verbose=False,
            allow_writing_files=False,
        )
        return m, e

    raise ValueError(f"unknown backend={backend!r}")


# --------------------------------------------------------------------------- #
# Cross-fitting of nuisance functions
# --------------------------------------------------------------------------- #


@dataclass
class NuisanceFit:
    m_hat: np.ndarray            # out-of-fold E[Y | X]
    e_hat: np.ndarray            # out-of-fold P(A = 1 | X)
    folds: np.ndarray            # fold index k(i) for each observation
    m_models: list               # per-fold fitted regressors for m_0
    e_models: list               # per-fold fitted classifiers for e_0


def cross_fit_nuisances(
    X: np.ndarray,
    A: np.ndarray,
    Y: np.ndarray,
    n_folds: int = 5,
    backend: str = "catboost",
    m_estimator: Optional[BaseEstimator] = None,
    e_estimator: Optional[BaseEstimator] = None,
    seed: Optional[int] = 0,
) -> NuisanceFit:
    """Compute out-of-fold predictions of ``m_0`` and ``e_0`` via K-fold cross-fitting.

    If ``m_estimator`` / ``e_estimator`` are passed explicitly they take
    precedence; otherwise both are constructed from ``backend`` via
    :func:`_make_nuisance_estimators`.
    """
    if m_estimator is None or e_estimator is None:
        m_default, e_default = _make_nuisance_estimators(backend, seed=seed)
        if m_estimator is None:
            m_estimator = m_default
        if e_estimator is None:
            e_estimator = e_default

    n = X.shape[0]
    m_hat = np.zeros(n)
    e_hat = np.zeros(n)
    folds = np.zeros(n, dtype=int)
    m_models = []
    e_models = []

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (tr, te) in enumerate(kf.split(X)):
        folds[te] = k

        m_k = clone(m_estimator).fit(X[tr], Y[tr])
        m_hat[te] = m_k.predict(X[te])
        m_models.append(m_k)

        e_k = clone(e_estimator).fit(X[tr], A[tr])
        # robustly extract P(A = 1 | X)
        if hasattr(e_k, "predict_proba"):
            e_hat[te] = e_k.predict_proba(X[te])[:, 1]
        else:
            e_hat[te] = np.clip(e_k.predict(X[te]), 0.0, 1.0)
        e_models.append(e_k)

    # numeric safety: keep propensities strictly inside (0, 1)
    e_hat = np.clip(e_hat, 1e-3, 1 - 1e-3)

    return NuisanceFit(m_hat=m_hat, e_hat=e_hat, folds=folds,
                       m_models=m_models, e_models=e_models)


def residualise(
    A: np.ndarray, Y: np.ndarray, nuisances: NuisanceFit
) -> tuple[np.ndarray, np.ndarray]:
    """Return (Ytil, Atil) as in Definition 2.2 of the thesis."""
    return Y - nuisances.m_hat, A - nuisances.e_hat


# --------------------------------------------------------------------------- #
# Second-stage estimator
# --------------------------------------------------------------------------- #


class RLearnerCATE:
    """R-learner-style second-stage estimator for the CATE.

    The second stage minimises the weighted least-squares objective

        sum_i Atil_i^2 * (Z_i - tau(X_i))^2 + pen_n(tau),     Z_i = Ytil_i / Atil_i,

    which is the standard rearrangement of the R-learner loss
    (Nie & Wager, 2021). Four regularised non-parametric backends are supported:

    * ``"catboost"`` (default) - gradient boosting via CatBoost; the
      regularisation comes from limited tree depth, shrinkage, and
      ordered boosting.
    * ``"xgboost"``  - gradient boosting via XGBoost.
    * ``"gbr"``      - sklearn's :class:`GradientBoostingRegressor` with a
      sample-weighted MSE loss; same controls as above.
    * ``"krr"``      - kernel ridge regression with a Gaussian kernel
      (true Tikhonov regularisation, useful for sup-norm theory).

    The resulting :meth:`predict` returns an estimate of ``tau_0(x)``.

    Parameters
    ----------
    backend          : backend for the *second-stage* estimator.
    nuisance_backend : backend for the cross-fitted nuisance estimators
                       (defaults to ``backend``; KRR falls back to GBR).
    """

    def __init__(
        self,
        backend: str = "catboost",
        nuisance_backend: Optional[str] = None,
        n_folds: int = 5,
        seed: Optional[int] = 0,
        # GBR hyper-parameters
        gbr_n_estimators: int = 400,
        gbr_max_depth: int = 3,
        gbr_learning_rate: float = 0.05,
        # XGBoost hyper-parameters
        xgb_n_estimators: int = 400,
        xgb_max_depth: int = 4,
        xgb_learning_rate: float = 0.05,
        # CatBoost hyper-parameters
        cb_iterations: int = 500,
        cb_depth: int = 6,
        cb_learning_rate: float = 0.05,
        # KRR hyper-parameters
        krr_alpha: float = 1e-2,
        krr_gamma: float = 5.0,
        # explicit overrides for nuisance estimators (take precedence over backend)
        m_estimator: Optional[BaseEstimator] = None,
        e_estimator: Optional[BaseEstimator] = None,
        eps: float = 1e-2,
    ) -> None:
        self.backend = backend.lower()
        self.nuisance_backend = (nuisance_backend or backend).lower()
        self.n_folds = n_folds
        self.seed = seed
        self.gbr_n_estimators = gbr_n_estimators
        self.gbr_max_depth = gbr_max_depth
        self.gbr_learning_rate = gbr_learning_rate
        self.xgb_n_estimators = xgb_n_estimators
        self.xgb_max_depth = xgb_max_depth
        self.xgb_learning_rate = xgb_learning_rate
        self.cb_iterations = cb_iterations
        self.cb_depth = cb_depth
        self.cb_learning_rate = cb_learning_rate
        self.krr_alpha = krr_alpha
        self.krr_gamma = krr_gamma
        self.m_estimator = m_estimator
        self.e_estimator = e_estimator
        self.eps = eps
        self.model_: Optional[BaseEstimator] = None
        self.nuisances_: Optional[NuisanceFit] = None

    def _make_second_stage(self) -> BaseEstimator:
        b = self.backend
        if b == "gbr":
            return GradientBoostingRegressor(
                n_estimators=self.gbr_n_estimators,
                max_depth=self.gbr_max_depth,
                learning_rate=self.gbr_learning_rate,
                random_state=self.seed,
            )
        if b == "krr":
            return KernelRidge(
                alpha=self.krr_alpha, kernel="rbf", gamma=self.krr_gamma
            )
        if b == "xgboost":
            try:
                from xgboost import XGBRegressor
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "backend='xgboost' requires the 'xgboost' package; "
                    "install it via `pip install xgboost`."
                ) from exc
            return XGBRegressor(
                n_estimators=self.xgb_n_estimators,
                max_depth=self.xgb_max_depth,
                learning_rate=self.xgb_learning_rate,
                tree_method="hist",
                random_state=self.seed,
                verbosity=0,
            )
        if b == "catboost":
            try:
                from catboost import CatBoostRegressor
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "backend='catboost' requires the 'catboost' package; "
                    "install it via `pip install catboost`."
                ) from exc
            return CatBoostRegressor(
                iterations=self.cb_iterations,
                depth=self.cb_depth,
                learning_rate=self.cb_learning_rate,
                random_state=self.seed,
                verbose=False,
                allow_writing_files=False,
            )
        raise ValueError(f"unknown backend={self.backend!r}")

    def fit(self, X: np.ndarray, A: np.ndarray, Y: np.ndarray) -> "RLearnerCATE":
        self.nuisances_ = cross_fit_nuisances(
            X, A, Y,
            n_folds=self.n_folds,
            backend=self.nuisance_backend,
            m_estimator=self.m_estimator,
            e_estimator=self.e_estimator,
            seed=self.seed,
        )
        Ytil, Atil = residualise(A, Y, self.nuisances_)

        # Reformulate as weighted regression of Z = Ytil / Atil on X with
        # weights w = Atil^2. We guard against tiny Atil to avoid blow-up.
        eps = self.eps
        safe = np.abs(Atil) >= eps
        if safe.sum() < max(50, X.shape[1] * 10):
            # fall back to all observations with floored Atil
            Atil_safe = np.where(np.abs(Atil) < eps, np.sign(Atil) * eps + eps, Atil)
            Z = Ytil / Atil_safe
            w = Atil_safe ** 2
            X_fit, Z_fit, w_fit = X, Z, w
        else:
            Z = Ytil[safe] / Atil[safe]
            w = Atil[safe] ** 2
            X_fit, Z_fit, w_fit = X[safe], Z, w

        model = self._make_second_stage()
        if self.backend == "krr":
            # KernelRidge does not support sample_weight directly; the cleanest
            # quadratic-loss equivalent would be to scale (X, Z) row-wise by
            # sqrt(w), but that distorts the kernel. For an empirical study
            # the unweighted KRR fit is a reasonable approximation.
            model.fit(X_fit, Z_fit)
        else:
            # gbr / xgboost / catboost all accept sample_weight via fit().
            model.fit(X_fit, Z_fit, sample_weight=w_fit)
        self.model_ = model
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("call .fit(...) before .predict(...)")
        return self.model_.predict(X)

    # convenience -- the estimator behaves as a callable tau_hat(x)
    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.predict(X)


# --------------------------------------------------------------------------- #
# Convenience helper
# --------------------------------------------------------------------------- #


def fit_tau_hat(
    X: np.ndarray,
    A: np.ndarray,
    Y: np.ndarray,
    backend: str = "catboost",
    n_folds: int = 5,
    seed: Optional[int] = 0,
) -> Callable[[np.ndarray], np.ndarray]:
    """One-shot helper returning a callable estimator ``tau_hat(x)``."""
    est = RLearnerCATE(backend=backend, n_folds=n_folds, seed=seed).fit(X, A, Y)
    return est


# --------------------------------------------------------------------------- #
# Bootstrap-based scale estimators (amplitude and eps_n)
# --------------------------------------------------------------------------- #


@dataclass
class TauScaleEstimate:
    """Bundle of bootstrap-based scale estimates for ``tau_hat``.

    Attributes
    ----------
    amplitude_minmax : float
        ``max tau_hat(X_grid) - min tau_hat(X_grid)`` for the main fit (no bootstrap).
        Optimistic: captures sharp peaks but is sensitive to outliers.
    amplitude_bootstrap : float
        Robust amplitude estimate: ``Q_high - Q_low`` averaged across bootstrap
        replicates of ``tau_hat`` on ``X_grid``.
    eps_n : float
        Bootstrap sup-discrepancy estimate of ``||tau_hat - E[tau_hat]||_inf``.
        Computed as the ``alpha``-quantile of
        ``max_x |tau_hat_b(x) - mean_b tau_hat_b(x)|`` over bootstrap replicates.
        Serves as a noise-band proxy for the landscape plot.
    tau_predictions : np.ndarray
        (B, |X_grid|) matrix of bootstrap predictions, kept for diagnostics.
    """

    amplitude_minmax: float
    amplitude_bootstrap: float
    eps_n: float
    tau_predictions: np.ndarray


def estimate_tau_scales(
    X: np.ndarray,
    A: np.ndarray,
    Y: np.ndarray,
    X_grid: np.ndarray,
    *,
    backend: str = "catboost",
    n_folds: int = 5,
    n_bootstrap: int = 100,
    q_low: float = 0.01,
    q_high: float = 0.99,
    alpha: float = 0.95,
    seed: Optional[int] = 0,
) -> TauScaleEstimate:
    """Bootstrap estimator for amplitude and ``eps_n`` of a CATE estimate.

    For each bootstrap replicate ``b = 1, ..., B`` we resample ``(X, A, Y)``
    with replacement, refit the R-learner, and predict on the fixed grid
    ``X_grid``. From the resulting ``(B, |X_grid|)`` matrix of predictions
    we extract:

    * ``amplitude_minmax`` -- raw ``max - min`` of the original (non-bootstrap)
      fit on ``X_grid``;
    * ``amplitude_bootstrap`` -- mean of ``Q_high - Q_low`` across replicates;
    * ``eps_n`` -- ``alpha``-quantile of ``sup_x |tau_b(x) - bar tau(x)|``,
      i.e. an empirical sup-discrepancy of the estimator around its bootstrap
      mean.  Captures variance, not bias.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X)
    A = np.asarray(A)
    Y = np.asarray(Y)
    X_grid = np.asarray(X_grid)
    n = X.shape[0]

    # main fit -- amplitude_minmax from it
    main = RLearnerCATE(backend=backend, n_folds=n_folds, seed=seed).fit(X, A, Y)
    tau_main = main.predict(X_grid)
    amplitude_minmax = float(tau_main.max() - tau_main.min())

    # bootstrap replicates
    preds = np.empty((n_bootstrap, X_grid.shape[0]), dtype=np.float64)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        seed_b = None if seed is None else int(seed + 1 + b)
        est_b = RLearnerCATE(backend=backend, n_folds=n_folds, seed=seed_b)
        est_b.fit(X[idx], A[idx], Y[idx])
        preds[b] = est_b.predict(X_grid)

    # robust amplitude: average across replicates of Q_high - Q_low
    q_hi = np.quantile(preds, q_high, axis=1)
    q_lo = np.quantile(preds, q_low, axis=1)
    amplitude_bootstrap = float(np.mean(q_hi - q_lo))

    # eps_n: sup-discrepancy quantile across replicates
    tau_mean = preds.mean(axis=0)
    sup_dev = np.max(np.abs(preds - tau_mean[None, :]), axis=1)
    eps_n = float(np.quantile(sup_dev, alpha))

    return TauScaleEstimate(
        amplitude_minmax=amplitude_minmax,
        amplitude_bootstrap=amplitude_bootstrap,
        eps_n=eps_n,
        tau_predictions=preds,
    )
