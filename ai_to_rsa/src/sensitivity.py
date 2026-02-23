from __future__ import annotations

"""Review-oriented sensitivity analyses.

This module provides small, dependency-light helpers used by the pipeline to
produce additional robustness checks commonly requested in review, including:

..."""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .modeling import add_const, condition_dummies, interactions_with_series
from .plotting import save_figure, set_grayscale_style
from .stats import brier_score, cluster_covariance, fit_binom_glm, log_likelihood_from_probs, tidy_coef_table, parse_cluster_cols


@dataclass(frozen=True)
class ClusterRobustGLM:
    link: str
    cluster_col: str
    n_clusters: int
    n_obs: int
    x_cols: List[str]
    llf: float


def fit_cluster_robust_glm(
    df: pd.DataFrame,
    *,
    y_col: str = "y",
    x_cols: Sequence[str],
    cluster_cols: Union[str, Sequence[str]] = "worker",
    link: str = "logit",
) -> Tuple[ClusterRobustGLM, object, pd.DataFrame]:
    """Fit a binomial GLM and return a cluster-robust coefficient table."""

    x_cols = list(x_cols)
    y = df[y_col].to_numpy(dtype=float)
    X = add_const(df[x_cols])

    res = fit_binom_glm(y, X, link=link)
    ccols = [c for c in parse_cluster_cols(cluster_cols) if c in df.columns]
    if not ccols:
        ccols = ["worker"]
    clusters_obj = df[ccols] if len(ccols) > 1 else df[ccols[0]]

    cov, G = cluster_covariance(res, clusters_obj)
    coef = tidy_coef_table(res, cov, df_t=max(int(G) - 1, 1))

    meta = ClusterRobustGLM(
        link=str(link),
        cluster_col="+".join(ccols),
        n_clusters=int(G),
        n_obs=int(df.shape[0]),
        x_cols=x_cols,
        llf=float(res.llf),
    )
    return meta, res, coef


def _safe_logit(p: np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def calibration_curve_binned(
    y: np.ndarray,
    p: np.ndarray,
    *,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> pd.DataFrame:
    """Compute a standard reliability curve table (no external deps).

    Returns one row per bin with:
        n, p_mean, y_mean, p_lo, p_hi
    """

    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = p[mask]

    if y.size == 0:
        return pd.DataFrame(
            {
                "bin": [],
                "n": [],
                "p_mean": [],
                "y_mean": [],
                "p_lo": [],
                "p_hi": [],
            }
        )

    n_bins = int(max(n_bins, 1))
    strat = str(strategy).strip().lower()
    if strat not in {"uniform", "quantile"}:
        raise ValueError("strategy must be 'uniform' or 'quantile'")

    if strat == "quantile":
        # Quantile binning can collapse when many identical probabilities exist;
        # we guard by uniquifying edges.
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(p, qs)
        edges = np.unique(edges)
        if edges.size < 2:
            edges = np.array([0.0, 1.0])
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)

    # Bin index in [0, n_bins-1]
    bin_idx = np.digitize(p, edges[1:-1], right=False)

    rows: List[Dict] = []
    for b in range(int(np.max(bin_idx)) + 1):
        m = bin_idx == b
        if not np.any(m):
            continue
        pb = p[m]
        yb = y[m]
        rows.append(
            {
                "bin": int(b),
                "n": int(pb.size),
                "p_mean": float(np.mean(pb)),
                "y_mean": float(np.mean(yb)),
                "p_lo": float(np.min(pb)),
                "p_hi": float(np.max(pb)),
            }
        )

    return pd.DataFrame(rows).sort_values("bin").reset_index(drop=True)


def expected_calibration_error(calib: pd.DataFrame) -> float:
    """ECE computed from a binned calibration curve table."""

    if calib is None or calib.shape[0] == 0:
        return float("nan")
    n = calib["n"].to_numpy(dtype=float)
    w = n / max(float(np.sum(n)), 1.0)
    gap = np.abs(calib["y_mean"].to_numpy(dtype=float) - calib["p_mean"].to_numpy(dtype=float))
    return float(np.sum(w * gap))


def calibration_intercept_slope(y: np.ndarray, p: np.ndarray) -> Tuple[float, float]:
    """Per-(sub)set calibration intercept & slope via logistic recalibration.

    We fit:
        y ~ 1 + logit(p)
    and return (intercept, slope).
    """

    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = p[mask]
    if y.size < 10:
        return (float("nan"), float("nan"))

    x = _safe_logit(p)
    X = add_const(pd.DataFrame({"logit_p": x}))
    res = fit_binom_glm(y, X, link="logit")
    params = np.asarray(res.params, dtype=float)
    if params.size < 2:
        return (float("nan"), float("nan"))
    return float(params[0]), float(params[1])


def calibration_by_group(
    df: pd.DataFrame,
    *,
    y_col: str,
    p_col: str,
    group_col: str,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calibration curve + per-group summary (Brier, ECE, slope/intercept)."""

    rows_curve: List[pd.DataFrame] = []
    rows_sum: List[Dict] = []
    for g, sub in df.groupby(group_col, dropna=False):
        y = sub[y_col].to_numpy(dtype=float)
        p = sub[p_col].to_numpy(dtype=float)
        curve = calibration_curve_binned(y, p, n_bins=n_bins, strategy=strategy)
        curve.insert(0, "group", str(g))
        rows_curve.append(curve)

        rows_sum.append(
            {
                "group": str(g),
                "n": int(np.isfinite(y).sum()),
                "brier": brier_score(y, p),
                "ece": expected_calibration_error(curve),
                "calib_intercept": calibration_intercept_slope(y, p)[0],
                "calib_slope": calibration_intercept_slope(y, p)[1],
            }
        )

    curve_df = pd.concat(rows_curve, axis=0, ignore_index=True) if rows_curve else pd.DataFrame([])
    sum_df = pd.DataFrame(rows_sum) if rows_sum else pd.DataFrame([])
    return curve_df, sum_df


def plot_calibration_curves(
    curve_df: pd.DataFrame,
    *,
    base_path: str,
    xlabel: str = "Mean predicted probability",
    ylabel: str = "Empirical accuracy",
    title: Optional[str] = None,
    dpi: int = 800,
) -> None:
    """Plot reliability curves for multiple groups in grayscale."""

    import matplotlib.pyplot as plt

    set_grayscale_style()

    fig, ax = plt.subplots(figsize=(5.5, 5.0))

    ax.plot([0, 1], [0, 1], linestyle="--", color="0.4", linewidth=1.0)

    if curve_df is not None and curve_df.shape[0] > 0:
        groups = sorted(curve_df["group"].astype(str).unique().tolist())
        # Deterministic grayscale levels
        grays = np.linspace(0.15, 0.85, num=max(len(groups), 1))
        for i, g in enumerate(groups):
            sub = curve_df.loc[curve_df["group"].astype(str) == str(g)].sort_values("bin")
            ax.plot(
                sub["p_mean"].to_numpy(dtype=float),
                sub["y_mean"].to_numpy(dtype=float),
                marker="o",
                markersize=3,
                linewidth=1.0,
                color=str(float(grays[i])),
                label=str(g),
            )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(str(title))
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=7, loc="lower right")

    save_figure(fig, base_path, dpi=dpi)


def add_condition_interactions(
    df: pd.DataFrame,
    *,
    condition_categories: Sequence[str],
    cols: Sequence[str],
    dummy_prefix: str = "cond",
) -> Tuple[pd.DataFrame, List[str]]:
    """Add condition dummies and a list of z×condition interaction columns."""

    dum = condition_dummies(df, categories=condition_categories, prefix=dummy_prefix)
    ints = []
    for c in cols:
        ints.append(interactions_with_series(dum, df[c], c, dummy_prefix=f"{dummy_prefix}_"))
    ints_df = pd.concat(ints, axis=1) if ints else pd.DataFrame(index=df.index)
    out = pd.concat([df, dum, ints_df], axis=1)
    return out, list(dum.columns) + list(ints_df.columns)


def fit_mixed_effect_logit_random_intercepts(
    df: pd.DataFrame,
    *,
    formula: str,
    vc_formulas: Dict[str, str],
    fit_method: str = "vb",
) -> Tuple[pd.DataFrame, pd.DataFrame, object]:
    """Fit a (Bayesian) binomial mixed model with random intercept(s).

    Statsmodels provides BinomialBayesMixedGLM, which we use as a
    reviewer-friendly random-intercept sensitivity check.
    """

    # IMPORTANT:
    # BinomialBayesMixedGLM.from_formula uses patsy to build the fixed-effect
    # and variance-component design matrices. Patsy will silently drop rows
    # containing NA *separately* for each design matrix depending on the
    # variables referenced.
    #
    # In our pipeline, the fixed-effects formula does not reference `worker`
    # or `game`, while the vc formulas do. If any rows have missing worker/game
    # IDs, patsy would drop those rows for the vc matrix but not for the fixed
    # effects, leading to a shape-mismatch error such as:
    #   ValueError: operands could not be broadcast together with shapes ...
    #
    # To make this reviewer-proof, we pre-align the rows kept by patsy across
    # the fixed + all vc formulas, then refit on the common complete-case subset.
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    import patsy

    # Determine the common row set implied by the fixed formula and all vc formulas.
    y_mat, x_mat = patsy.dmatrices(formula, df, return_type="dataframe")
    idx = y_mat.index
    for _name, vc_fml in vc_formulas.items():
        z = patsy.dmatrix(vc_fml, df, return_type="dataframe")
        idx = idx.intersection(z.index)

    if len(idx) < 10:
        raise ValueError(
            f"Too few rows after aligning NA dropping across formulas (n={len(idx)}). "
            "Check missingness in the grouping variables referenced by vc_formulas."
        )

    df_sub = df.loc[idx].copy()

    model = BinomialBayesMixedGLM.from_formula(formula, vc_formulas, df_sub)
    if str(fit_method).lower() == "map":
        res = model.fit_map()
    else:
        res = model.fit_vb()

    # Fixed effects
    fe_names = list(getattr(res.model, "exog_names", []))
    fe_mean = np.asarray(getattr(res, "fe_mean", []), dtype=float)
    fe_sd = np.asarray(getattr(res, "fe_sd", []), dtype=float)
    fe_z = fe_mean / np.where(fe_sd == 0, np.nan, fe_sd)
    fe = pd.DataFrame({"term": fe_names, "coef_mean": fe_mean, "coef_sd": fe_sd, "z": fe_z})
    fe.insert(0, "n_obs", int(df_sub.shape[0]))

    # Variance components (log standard deviations)
    vcp_names = list(getattr(res.model, "vcp_names", []))
    vcp_mean = np.asarray(getattr(res, "vcp_mean", []), dtype=float)
    vcp_sd = np.asarray(getattr(res, "vcp_sd", []), dtype=float)
    sd_mean = np.exp(vcp_mean)
    sd_sd = sd_mean * vcp_sd  # delta-method (approx)

    vcp = pd.DataFrame(
        {
            "component": vcp_names,
            "log_sd_mean": vcp_mean,
            "log_sd_sd": vcp_sd,
            "sd_mean_approx": sd_mean,
            "sd_sd_delta": sd_sd,
        }
    )

    vcp.insert(0, "n_obs", int(df_sub.shape[0]))

    return fe, vcp, res
