from __future__ import annotations

"""Scale-adjusted (heteroskedastic) logit for reviewer-facing sensitivity analyses.

Motivation
----------
In logistic models, apparent group differences in coefficients can arise from
*scale heterogeneity* (different residual variance / unobserved heterogeneity),
not only from "true" coefficient differences. A common diagnostic is to fit a
"heterogeneous choice" / "scale-adjusted" logit where the scale is modeled as a
function of group indicators, and then re-test interaction blocks.

Model
-----
For observation i:
    t_i = (X_i @ beta) / s_i
    s_i = exp(Z_i @ gamma)   (baseline group has Z_i = 0 => s_i = 1)
    p_i = logistic(t_i)

We implement MLE with analytic gradient and analytic Hessian for stability.
Cluster-robust sandwich covariance is computed from per-observation scores.

This module is dependency-light (SciPy + NumPy + pandas) and does not rely on
specialized heteroskedastic logit classes (which are not available in
statsmodels<=0.14).

The main pipeline uses this module only as a sensitivity check.
"""

from dataclasses import dataclass
from itertools import combinations
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from .modeling import add_const, condition_dummies, interactions_with_series
from .stats import parse_cluster_cols, tidy_coef_table, wald_joint


# ----------------------------
# Utilities: clustering helpers
# ----------------------------

def _coerce_cluster_series(
    x: Union[pd.Series, np.ndarray, Sequence],
    *,
    name: str = "cluster",
) -> pd.Series:
    s = x if isinstance(x, pd.Series) else pd.Series(x)
    if s.name is None:
        s = s.rename(name)
    return s.astype("string").fillna("(missing)")


def _intersection_codes(df_clusters: pd.DataFrame) -> Tuple[np.ndarray, int]:
    """Row-wise intersection cluster codes + number of unique clusters."""
    mi = pd.MultiIndex.from_frame(df_clusters.astype("string"), names=list(df_clusters.columns))
    codes, uniq = pd.factorize(mi, sort=False)
    return codes.astype(int), int(len(uniq))


def _cluster_sandwich_cov_oneway(
    score_obs: np.ndarray,
    hess_inv: np.ndarray,
    groups: Union[pd.Series, np.ndarray, Sequence],
    *,
    use_correction: bool = True,
) -> Tuple[np.ndarray, int]:
    """One-way cluster-robust covariance for generic MLE."""
    gser = _coerce_cluster_series(groups, name="cluster").astype("category")
    g = gser.cat.codes.to_numpy(dtype=int)
    G = int(np.unique(g).size)

    n, k = score_obs.shape

    # Sum scores within cluster
    Sg = np.zeros((G, k), dtype=float)
    np.add.at(Sg, g, score_obs)

    B = Sg.T @ Sg
    cov = hess_inv @ B @ hess_inv

    if use_correction and G > 1:
        # Match statsmodels' default finite-sample correction in cov_cluster.
        denom = max(n - k, 1)
        cov *= (G / (G - 1)) * ((n - 1) / denom)

    return cov, G


def cluster_sandwich_cov(
    score_obs: np.ndarray,
    hess_inv: np.ndarray,
    clusters: Union[pd.Series, np.ndarray, Sequence, pd.DataFrame, Sequence[Sequence]],
    *,
    use_correction: bool = True,
) -> Tuple[np.ndarray, int]:
    """Cluster-robust covariance (one-way or multiway; CGM inclusion–exclusion)."""

    if clusters is None:
        # IID covariance (observed information inverse)
        return np.asarray(hess_inv, dtype=float), int(score_obs.shape[0])

    if isinstance(clusters, pd.DataFrame):
        cdf = clusters.copy()
    elif isinstance(clusters, (list, tuple)) and len(clusters) > 0 and not isinstance(clusters, (pd.Series, np.ndarray)):
        cdf = pd.DataFrame({f"c{i}": _coerce_cluster_series(c, name=f"c{i}") for i, c in enumerate(clusters)})
    else:
        cdf = None

    # One-way
    if cdf is None or cdf.shape[1] <= 1:
        return _cluster_sandwich_cov_oneway(score_obs, hess_inv, clusters, use_correction=use_correction)

    # Multiway (Cameron–Gelbach–Miller)
    keep_cols: List[str] = []
    g_counts: List[int] = []
    for c in list(cdf.columns):
        nuniq = int(_coerce_cluster_series(cdf[c], name=c).nunique(dropna=False))
        if nuniq > 1:
            keep_cols.append(c)
            g_counts.append(nuniq)

    if len(keep_cols) <= 1:
        col = keep_cols[0] if keep_cols else cdf.columns[0]
        return _cluster_sandwich_cov_oneway(score_obs, hess_inv, cdf[col], use_correction=use_correction)

    cdf = cdf[keep_cols].copy()
    n_dims = int(cdf.shape[1])
    n_clusters_eff = int(min(g_counts)) if g_counts else int(cdf.iloc[:, 0].nunique(dropna=False))

    cov_total = np.zeros((score_obs.shape[1], score_obs.shape[1]), dtype=float)

    for r in range(1, n_dims + 1):
        sign = 1.0 if (r % 2 == 1) else -1.0
        for dims in combinations(range(n_dims), r):
            sub = cdf.iloc[:, list(dims)]
            if sub.shape[1] == 1:
                cov_s, _G = _cluster_sandwich_cov_oneway(score_obs, hess_inv, sub.iloc[:, 0], use_correction=use_correction)
            else:
                codes, _G = _intersection_codes(sub)
                cov_s, _G2 = _cluster_sandwich_cov_oneway(score_obs, hess_inv, codes, use_correction=use_correction)
            cov_total += sign * cov_s

    return cov_total, n_clusters_eff


# ----------------------------
# Core MLE
# ----------------------------

@dataclass(frozen=True)
class ScaleAdjustedLogitFit:
    """Fitted scale-adjusted logit (generic MLE container compatible with tidy_coef_table)."""
    params: pd.Series
    llf: float
    n_obs: int
    n_params: int
    converged: bool
    message: str

    cov_iid: np.ndarray
    cov_cluster: np.ndarray
    n_clusters: int

    # Design metadata (for debugging / plotting)
    utility_cols: List[str]
    scale_cols: List[str]

    def coef_table(self, *, use_cluster: bool = True) -> pd.DataFrame:
        cov = self.cov_cluster if use_cluster else self.cov_iid
        df_t = max(int(self.n_clusters) - 1, 1) if use_cluster else None
        return tidy_coef_table(self, cov, df_t=df_t)


def _loglike_grad_hess(
    theta: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (llf, score, hess_ll, p, t, a) for log-likelihood."""
    n = y.size
    p_beta = X.shape[1]
    p_gamma = Z.shape[1]

    beta = theta[:p_beta]
    gamma = theta[p_beta:]

    eta = X @ beta
    a = np.exp(-(Z @ gamma)) if p_gamma else np.ones(n, dtype=float)  # 1/scale
    t = eta * a
    p = expit(t)
    r = y - p
    w = p * (1.0 - p)

    # ll = sum(y*t - log(1+exp(t)))
    ll_i = y * t - np.logaddexp(0.0, t)
    llf = float(np.sum(ll_i))

    # score
    Xs = X * a[:, None]
    score_beta = Xs.T @ r
    if p_gamma:
        rt = -(r * t)
        score_gamma = Z.T @ rt
        score = np.concatenate([score_beta, score_gamma], axis=0)
    else:
        score = score_beta

    # Hessian of llf (analytic)
    H_bb = -(Xs.T @ (w[:, None] * Xs))

    if p_gamma:
        # beta-gamma: sum_i c_i outer(Xs_i, Z_i), c_i = -r_i + w_i*t_i
        c = (-r + w * t)
        H_bg = Xs.T @ (c[:, None] * Z)

        # gamma-gamma: sum_i d_i outer(Z_i, Z_i), d_i = r_i*t_i - w_i*t_i^2
        d = (r * t - w * (t * t))
        H_gg = Z.T @ (d[:, None] * Z)

        top = np.concatenate([H_bb, H_bg], axis=1)
        bot = np.concatenate([H_bg.T, H_gg], axis=1)
        H = np.concatenate([top, bot], axis=0)
    else:
        H = H_bb

    H = 0.5 * (H + H.T)
    return llf, score, H, p, t, a


def fit_scale_adjusted_logit(
    y: Union[np.ndarray, Sequence[float]],
    X: pd.DataFrame,
    Z: pd.DataFrame,
    *,
    clusters: Optional[Union[pd.Series, np.ndarray, Sequence, pd.DataFrame]] = None,
    start_params: Optional[np.ndarray] = None,
    maxiter: int = 600,
    gtol: float = 1e-5,
) -> ScaleAdjustedLogitFit:
    """Fit the scale-adjusted logit via MLE.

    Notes
    -----
    - X: utility design (WITH intercept and condition intercept dummies)
    - Z: scale design (NO intercept; reference condition omitted => baseline scale=1)
    """

    y = np.asarray(y, dtype=float).reshape(-1)
    Xv = np.asarray(X, dtype=float)
    Zv = np.asarray(Z, dtype=float)

    n = int(y.size)
    if n == 0:
        raise ValueError("fit_scale_adjusted_logit: empty y")

    p_beta = int(Xv.shape[1])
    p_gamma = int(Zv.shape[1])
    k = p_beta + p_gamma

    # Initialize beta from standard logit; gamma = 0
    if start_params is None:
        try:
            import statsmodels.api as sm
            res0 = sm.GLM(y, Xv, family=sm.families.Binomial()).fit()
            beta0 = np.asarray(res0.params, dtype=float)
            if beta0.size != p_beta:
                beta0 = np.zeros(p_beta, dtype=float)
        except Exception:
            beta0 = np.zeros(p_beta, dtype=float)
        gamma0 = np.zeros(p_gamma, dtype=float)
        start_params = np.concatenate([beta0, gamma0], axis=0)

    start_params = np.asarray(start_params, dtype=float).reshape(-1)
    if start_params.size != k:
        raise ValueError(f"start_params has wrong length: got {start_params.size}, expected {k}")

    def nll(theta: np.ndarray) -> float:
        llf, _, _, _, _, _ = _loglike_grad_hess(theta, y, Xv, Zv)
        return -llf

    def grad(theta: np.ndarray) -> np.ndarray:
        _, score, _, _, _, _ = _loglike_grad_hess(theta, y, Xv, Zv)
        return -score

    def hess(theta: np.ndarray) -> np.ndarray:
        _, _, H_ll, _, _, _ = _loglike_grad_hess(theta, y, Xv, Zv)
        return -H_ll

    # Try second-order first; fallback to BFGS
    opt = minimize(
        nll,
        x0=start_params,
        jac=grad,
        hess=hess,
        method="trust-ncg",
        options={"gtol": float(gtol), "maxiter": int(maxiter), "disp": False},
    )
    if not bool(getattr(opt, "success", False)):
        opt = minimize(
            nll,
            x0=np.asarray(opt.x, dtype=float),
            jac=grad,
            method="BFGS",
            options={"gtol": float(gtol), "maxiter": int(maxiter) * 2, "disp": False},
        )

    theta_hat = np.asarray(opt.x, dtype=float)
    llf, _, H_ll, p, t, a = _loglike_grad_hess(theta_hat, y, Xv, Zv)

    H_nll = -H_ll
    H_inv = np.linalg.pinv(H_nll)

    # score per observation (for sandwich)
    r = y - p
    Xs = Xv * a[:, None]
    score_beta_obs = r[:, None] * Xs
    if p_gamma:
        score_gamma_obs = -(r * t)[:, None] * Zv
        score_obs = np.concatenate([score_beta_obs, score_gamma_obs], axis=1)
    else:
        score_obs = score_beta_obs

    cov_iid = H_inv
    cov_cluster, G_eff = cluster_sandwich_cov(score_obs, H_inv, clusters, use_correction=True)

    util_cols = list(X.columns)
    scale_cols = list(Z.columns)
    names = util_cols + [f"log_scale_{c.replace('cond_', '')}" for c in scale_cols]
    params = pd.Series(theta_hat, index=names, dtype=float)

    return ScaleAdjustedLogitFit(
        params=params,
        llf=float(llf),
        n_obs=int(n),
        n_params=int(k),
        converged=bool(getattr(opt, "success", False)),
        message=str(getattr(opt, "message", "")),
        cov_iid=np.asarray(cov_iid, dtype=float),
        cov_cluster=np.asarray(cov_cluster, dtype=float),
        n_clusters=int(G_eff),
        utility_cols=util_cols,
        scale_cols=scale_cols,
    )


# ----------------------------
# Pipeline-facing "suite" runner
# ----------------------------

@dataclass(frozen=True)
class ScaleAdjustedSuiteRow:
    disc_variant: str
    split: str
    cluster_spec: str
    n_obs: int
    n_clusters: int

    # A2 blocks (Wald; tested in FULL interaction model)
    A2_outcome_disc_wald_chi2: float
    A2_outcome_disc_df: int
    A2_outcome_disc_p: float

    A2_outcome_clarity_wald_chi2: float
    A2_outcome_clarity_df: int
    A2_outcome_clarity_p: float

    A2_outcome_len_wald_chi2: float
    A2_outcome_len_df: int
    A2_outcome_len_p: float

    A2_outcome_full_wald_chi2: float
    A2_outcome_full_df: int
    A2_outcome_full_p: float

    # A5 blocks
    A5_outcome_base_wald_chi2: float
    A5_outcome_base_df: int
    A5_outcome_base_p: float

    A5_outcome_given_A2_outcome_full_wald_chi2: float
    A5_outcome_given_A2_outcome_full_df: int
    A5_outcome_given_A2_outcome_full_p: float

    # Working-i.i.d. LR (base vs full interaction model)
    A2_outcome_full_lr_chi2: float
    A2_outcome_full_lr_df: int
    A2_outcome_full_lr_p: float

    # Scale parameters (log scale for non-reference conditions)
    log_scale_far: float
    log_scale_split: float
    log_scale_far_p: float
    log_scale_split_p: float

    converged_base: bool
    converged_full: bool


def run_scale_adjusted_logit_suite(
    df: pd.DataFrame,
    *,
    disc_variant: str,
    split_label: str,
    condition_categories: Sequence[str],
    cluster_cols: Union[str, Sequence[str]] = "worker",
    disc_col: str = "z_disc",
    clarity_col: str = "z_clarity",
    len_col: str = "z_len_chars",
) -> Tuple[ScaleAdjustedSuiteRow, pd.DataFrame, pd.DataFrame]:
    """Run the A2_outcome/A5_outcome diagnostics under a scale-adjusted logit.

    Two-step implementation (stable):
      1) Fit heteroskedastic (scale) logit WITHOUT slope×condition interactions
         to estimate condition-specific scales (gamma).
      2) Treat estimated scales as fixed; refit standard GLMs on row-scaled
         design matrices (X * exp(-Z@gamma)) and compute cluster-robust Wald tests.

    This avoids the identification/singularity problems that arise when trying
    to estimate free group-specific slopes and free group-specific scales jointly
    in a binary logit.
    """
    import statsmodels.api as sm
    from scipy.stats import chi2, t as student_t

    df = df.copy()
    y = df["y"].to_numpy(dtype=float)

    # Stable condition dummies (baseline dropped)
    dum = condition_dummies(df, categories=condition_categories, prefix="cond")

    # Utility designs (unscaled)
    main = df[[disc_col, clarity_col, len_col]].copy()
    X_base = add_const(pd.concat([main, dum], axis=1))

    ints_disc = interactions_with_series(dum, df[disc_col], disc_col)
    ints_clarity = interactions_with_series(dum, df[clarity_col], clarity_col)
    ints_len = interactions_with_series(dum, df[len_col], len_col)
    ints_full = pd.concat([ints_disc, ints_clarity, ints_len], axis=1)

    X_full = add_const(pd.concat([main, dum, ints_full], axis=1))

    # Scale design: no intercept; baseline condition has scale=1
    Z = dum.copy()

    # Clustering object
    ccols = [c for c in parse_cluster_cols(cluster_cols) if c in df.columns]
    if not ccols:
        ccols = ["worker"]
    clusters_obj = df[ccols] if len(ccols) > 1 else df[ccols[0]]
    cluster_spec = "+".join(ccols)

    # ------------------------------------------------------------------
    # Step 1) Estimate condition-specific scales (gamma) in identified model
    # ------------------------------------------------------------------
    fit_scale = fit_scale_adjusted_logit(y, X_base, Z, clusters=clusters_obj)

    # Extract gamma in Z column order
    gamma_names = [f"log_scale_{c.replace('cond_', '')}" for c in list(Z.columns)]
    gamma_hat = np.asarray([float(fit_scale.params.get(nm, 0.0)) for nm in gamma_names], dtype=float)

    if Z.shape[1] > 0:
        log_scale_i = Z.to_numpy(dtype=float) @ gamma_hat
        a = np.exp(-log_scale_i)  # 1/scale
    else:
        a = np.ones_like(y, dtype=float)

    # ------------------------------------------------------------------
    # Step 2) Fit standard GLMs on row-scaled design matrices
    # ------------------------------------------------------------------
    X_base_scaled = X_base.mul(a, axis=0)
    X_full_scaled = X_full.mul(a, axis=0)

    res_b = sm.GLM(y, X_base_scaled, family=sm.families.Binomial()).fit()
    res_f = sm.GLM(y, X_full_scaled, family=sm.families.Binomial()).fit()

    from .stats import cluster_covariance

    cov_b, G_b = cluster_covariance(res_b, clusters_obj)
    cov_f, G_f = cluster_covariance(res_f, clusters_obj)

    # Wald blocks (FULL scaled GLM)
    a2_disc = wald_joint(
        res_f, cov_f, clusters_n=G_f,
        terms=list(ints_disc.columns), block="A2_outcome_disc_block", use_f=True
    )
    a2_cl = wald_joint(
        res_f, cov_f, clusters_n=G_f,
        terms=list(ints_clarity.columns), block="A2_outcome_clarity_block", use_f=True
    )
    a2_len = wald_joint(
        res_f, cov_f, clusters_n=G_f,
        terms=list(ints_len.columns), block="A2_outcome_len_block", use_f=True
    )
    a2_full = wald_joint(
        res_f, cov_f, clusters_n=G_f,
        terms=list(ints_full.columns), block="A2_outcome_full_block", use_f=True
    )

    # A5 blocks (scaled GLMs)
    a5_base = wald_joint(
        res_b, cov_b, clusters_n=G_b,
        terms=list(dum.columns), block="A5_outcome_base_block", use_f=True
    )
    a5_given = wald_joint(
        res_f, cov_f, clusters_n=G_f,
        terms=list(dum.columns), block="A5_outcome_given_A2_outcome_full_block", use_f=True
    )

    # LR (working-i.i.d.) for adding ALL slope×condition interactions, conditional on scale
    lr = 2.0 * (float(res_f.llf) - float(res_b.llf))
    df_lr = int(X_full.shape[1] - X_base.shape[1])
    p_lr = float(chi2.sf(max(lr, 0.0), df_lr))

    # Scale params and p-values (from Step 1)
    scale_tab = fit_scale.coef_table(use_cluster=True)
    df_t_scale = max(int(fit_scale.n_clusters) - 1, 1)

    def _scale_term(term: str) -> Tuple[float, float]:
        row = scale_tab.loc[scale_tab["term"] == term]
        if row.empty:
            return float("nan"), float("nan")
        tval = float(row["t"].values[0])
        pval = float(2.0 * student_t.sf(abs(tval), df=df_t_scale))
        return float(row["coef"].values[0]), pval

    log_scale_far, p_far = _scale_term("log_scale_far")
    log_scale_split, p_split = _scale_term("log_scale_split")

    row = ScaleAdjustedSuiteRow(
        disc_variant=str(disc_variant),
        split=str(split_label),
        cluster_spec=str(cluster_spec),
        n_obs=int(df.shape[0]),
        n_clusters=int(G_f),

        A2_outcome_disc_wald_chi2=float(a2_disc.stat_chi2),
        A2_outcome_disc_df=int(a2_disc.df),
        A2_outcome_disc_p=float(a2_disc.p_value_chi2),

        A2_outcome_clarity_wald_chi2=float(a2_cl.stat_chi2),
        A2_outcome_clarity_df=int(a2_cl.df),
        A2_outcome_clarity_p=float(a2_cl.p_value_chi2),

        A2_outcome_len_wald_chi2=float(a2_len.stat_chi2),
        A2_outcome_len_df=int(a2_len.df),
        A2_outcome_len_p=float(a2_len.p_value_chi2),

        A2_outcome_full_wald_chi2=float(a2_full.stat_chi2),
        A2_outcome_full_df=int(a2_full.df),
        A2_outcome_full_p=float(a2_full.p_value_chi2),

        A5_outcome_base_wald_chi2=float(a5_base.stat_chi2),
        A5_outcome_base_df=int(a5_base.df),
        A5_outcome_base_p=float(a5_base.p_value_chi2),

        A5_outcome_given_A2_outcome_full_wald_chi2=float(a5_given.stat_chi2),
        A5_outcome_given_A2_outcome_full_df=int(a5_given.df),
        A5_outcome_given_A2_outcome_full_p=float(a5_given.p_value_chi2),

        A2_outcome_full_lr_chi2=float(lr),
        A2_outcome_full_lr_df=int(df_lr),
        A2_outcome_full_lr_p=float(p_lr),

        log_scale_far=float(log_scale_far),
        log_scale_split=float(log_scale_split),
        log_scale_far_p=float(p_far),
        log_scale_split_p=float(p_split),

        converged_base=bool(fit_scale.converged) and bool(getattr(res_b, "converged", True)),
        converged_full=bool(getattr(res_f, "converged", True)),
    )

    # Output coefficient tables:
    #   - base: scaled GLM utility coefficients + (appended) scale parameters
    #   - full: scaled GLM utility + interaction coefficients
    df_t_glm_b = max(int(G_b) - 1, 1)
    df_t_glm_f = max(int(G_f) - 1, 1)

    coefs_base_glm = tidy_coef_table(res_b, cov_b, df_t=df_t_glm_b)
    scale_rows = scale_tab.loc[scale_tab["term"].isin(["log_scale_far", "log_scale_split"])].copy()
    if not scale_rows.empty:
        scale_rows["df_t"] = df_t_scale

    coefs_base = pd.concat([coefs_base_glm, scale_rows], ignore_index=True)
    coefs_full = tidy_coef_table(res_f, cov_f, df_t=df_t_glm_f)

    return row, coefs_base, coefs_full
