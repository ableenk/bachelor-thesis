# Persistence Landscapes for Heterogeneous Treatment Effects: Stability and Consistency under Piecewise-Linear Approximation

Code for the bachelor's thesis (Apollonov N.A., HSE Faculty of Mathematics).

This repository implements an end-to-end numerical pipeline accompanying the
mathematical framework of the thesis:

1. **Data generation.** Synthetic data from a heterogeneous partially linear model
   `Y = g0(X) + A*tau0(X) + U` with a known ground-truth CATE `tau0`.
   Generated via [`DoubleML`](https://docs.doubleml.org/) data generators
   plus a custom generator with a controllable topology of the high-effect region.
2. **CATE estimation.** Cross-fitted residualisation (Robinson / R-learner) followed
   by a regularised non-parametric second-stage regressor.
3. **Geometric layer.** A shape-regular triangulation of the covariate domain
   (`scipy.spatial.Delaunay`) and piecewise-linear interpolation of the estimated
   `tau_hat` onto the triangulation vertices.
4. **Topological layer.** Persistent homology of the **superlevel-set filtration**
   of the PL function, encoded as the **sublevel filtration of `-f`** and computed
   with [`ripser`](https://ripser.scikit-tda.org/) via a lower-star distance matrix.
5. **Persistence landscapes.** Functional summaries
   `Lambda_q(f) = {lambda_k^f}_{k>=1}` of the persistence diagrams with
   discrete `L_p` and `L_inf` norms, in the spirit of Bubenik (2015).

## Layout

```
.
├── requirements.txt
├── README.md
├── project_proposal_ApollonovNA.pdf
├── src/
│   ├── __init__.py
│   ├── data.py            # synthetic data generators (incl. DoubleML)
│   ├── cate.py            # cross-fitted residualisation + 2nd-stage estimator
│   ├── triangulation.py   # shape-regular triangulation + PL interpolation
│   ├── topology.py        # lower-star persistent homology via ripser
│   ├── landscape.py       # persistence landscapes and Lp norms
│   └── plotting.py        # visualisation helpers
└── notebooks/
    └── main.ipynb         # end-to-end demonstrative pipeline
```

## Quickstart

```bash
pip install -r requirements.txt
jupyter notebook notebooks/main.ipynb
```

The notebook reproduces, on synthetic 2D data:

* Generation of the ground-truth CATE with a prescribed topological profile.
  Available `topology` presets in `data.make_topology_dgp`:
  `single`, `two_blobs`, `annulus`, `three_blobs`, `four_blobs`.
  The latter two place several Gaussian bumps of distinct heights and produce
  ground-truth landscapes `Lambda_0(tau_0)` with 2 (resp. 3) non-zero tents of
  decreasing height, which makes convergence checks more informative.
* R-learner-style estimation of `tau_hat`. Available `RLearnerCATE` backends:
  `catboost` (default), `xgboost`, `gbr`, `krr`. Both the nuisance and the
  second-stage estimators are configurable independently.
* Triangulation, PL interpolation, and computation of `Lambda_0(tau_hat)`.
* Empirical stability check
  `||Lambda_0(f) - Lambda_0(g)||_p  vs.  ||f - g||_inf`,
  illustrating the deterministic bound targeted in the thesis.
* Monte-Carlo convergence
  `||Lambda_0(f_{n,h}) - Lambda_0(tau_0)||_p = O_P(r_n + h^alpha)`,
  including a head-to-head comparison at moderate and large sample sizes
  on the richer `three_blobs` / `four_blobs` ground-truth topologies.
