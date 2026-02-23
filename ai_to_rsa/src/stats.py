from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm


def parse_cluster_cols(cluster_cols: Union[str, Sequence[str]]) -> List[str]:
    """Parse a clustering specification into a list of column names.

    This helper exists so callers can accept natural reviewer-oriented strings
    such as:

        "worker"            (one-way)
        "worker+game"       (two-way)
        "worker×game"       (two-way, unicode times)
        "worker,game"       (two-way, comma-separated)
        "worker+game+listener" (three-way)
    """

    if isinstance(cluster_cols, str):
        s = cluster_cols.strip()
        if not s:
            return ["worker"]
        # Normalize common separators.
        for sep in ["×", "x", "X", "*", "/", "\\"]:
            s = s.replace(sep, "+")
        s = s.replace(",", "+")
        parts = [p.strip() for p in s.split("+") if p.strip()]
        return parts if parts else ["worker"]

    cols = [str(c).strip() for c in list(cluster_cols) if str(c).strip()]
    return cols if cols else ["worker"]


@dataclass(frozen=True)
class WaldBlockResult:
    block: str
    stat_chi2: float
    df: int
    p_value_chi2: float
    f_stat: float
    df_num: int
    df_denom: int
    p_value_f: float
    n_clusters: int
    terms: List[str]
    missing_terms: List[str]

    # Optional wild cluster bootstrap p-value (computed under the block null).
    #
    # We keep this optional so the default pipeline remains fast and fully
    # backwards compatible with the archived manuscript outputs.
    p_value_wild: Optional[float] = None
    n_boot: Optional[int] = None
    wild_weight: Optional[str] = None
    wild_seed: Optional[int] = None


def _align_params_to_full(
    full_names: Sequence[str],
    null_names: Sequence[str],
    null_params: np.ndarray,
) -> np.ndarray:
    """Embed restricted-model parameters into the full parameter vector.

    Parameters present in the restricted fit are copied; parameters absent
    (i.e., restricted to zero) are set to 0.
    """

    name_to_idx_null = {str(nm): i for i, nm in enumerate(list(null_names))}
    out = np.zeros(len(full_names), dtype=float)
    for j, nm in enumerate(list(full_names)):
        i = name_to_idx_null.get(str(nm))
        if i is not None:
            out[j] = float(null_params[i])
    return out


def _wild_weights(
    clusters: Union[pd.Series, pd.DataFrame],
    rng: np.random.Generator,
    *,
    weight: str = "rademacher",
) -> np.ndarray:
    """Observation-level wild weights for one-way or multiway clustering.

    For one-way clustering, each cluster g gets an i.i.d. weight w_g and each
    observation i inherits w_{g(i)}.

    For multiway clustering (DataFrame with multiple columns), we implement the
    common *product* weighting scheme:

        w_i = Π_d w_{g_d(i)}

    where each dimension d gets its own i.i.d. cluster weights.

    Notes
    -----
    - This is a pragmatic implementation intended for diagnostics and reviewer
      sensitivity checks.
    - For the primary CiC analyses, the default remains the (faster) asymptotic
      multiway cluster Wald; users can enable this bootstrap to assess whether
      small-cluster df heuristics materially change conclusions.
    """

    wkey = str(weight).strip().lower()
    if wkey not in {"rademacher", "webb"}:
        raise ValueError("weight must be 'rademacher' or 'webb'")

    if isinstance(clusters, pd.DataFrame):
        cols = list(clusters.columns)
        if len(cols) == 0:
            raise ValueError("clusters DataFrame has no columns")
        w = np.ones(int(clusters.shape[0]), dtype=float)
        for c in cols:
            g = _coerce_cluster_series(clusters[c], name=str(c)).astype("category")
            codes = g.cat.codes.to_numpy()
            G = int(g.cat.categories.size)
            if wkey == "rademacher":
                wg = rng.choice([-1.0, 1.0], size=G)
            else:
                # Webb's 6-point distribution (balanced, variance=1)
                # Standard choice: { -sqrt(1.5), -sqrt(0.5), 0, 0, sqrt(0.5), sqrt(1.5) } each with prob 1/6.
                vals = np.array(
                    [
                        -np.sqrt(1.5),
                        -np.sqrt(0.5),
                        0.0,
                        0.0,
                        np.sqrt(0.5),
                        np.sqrt(1.5),
                    ],
                    dtype=float,
                )
                wg = rng.choice(vals, size=G, replace=True)
            w *= wg[codes]
        return w

    g = _coerce_cluster_series(clusters, name="cluster").astype("category")
    codes = g.cat.codes.to_numpy()
    G = int(g.cat.categories.size)
    if wkey == "rademacher":
        wg = rng.choice([-1.0, 1.0], size=G)
    else:
        vals = np.array(
            [
                -np.sqrt(1.5),
                -np.sqrt(0.5),
                0.0,
                0.0,
                np.sqrt(0.5),
                np.sqrt(1.5),
            ],
            dtype=float,
        )
        wg = rng.choice(vals, size=G, replace=True)
    return wg[codes]


def wild_cluster_bootstrap_wald_pvalue(
    res_full,
    res_null,
    *,
    cov_full: np.ndarray,
    clusters: Union[pd.Series, pd.DataFrame],
    terms: Sequence[str],
    n_boot: int = 999,
    seed: int = 0,
    weight: str = "rademacher",
) -> float:
    """Wild cluster bootstrap p-value for a joint Wald test.

    This implements a *multiplier / score* wild cluster bootstrap for the joint
    Wald statistic of a block of coefficients.

    - We approximate the distribution under the null by evaluating score and
      Hessian at the *restricted* estimator (fitted by excluding the tested
      block), embedded into the full parameter vector with zeros for the tested
      coefficients.
    - Cluster dependence is handled by assigning random weights at the cluster
      level (one-way) or via product weights across dimensions (multiway).

    The returned p-value is computed as:

        p = (1 + #{ W_b >= W_obs }) / (n_boot + 1)

    where W_obs is the standard cluster-robust Wald chi-square statistic from
    the full model using ``cov_full``.

    Notes
    -----
    This function is intended as a diagnostic supplement. The default pipeline
    remains the asymptotic cluster-robust Wald (chi2 + F approximation).
    """

    from numpy.linalg import pinv

    if n_boot <= 0:
        return float("nan")

    full_names = list(res_full.params.index) if hasattr(res_full.params, "index") else [f"beta{i}" for i in range(len(res_full.params))]
    null_names = list(res_null.params.index) if hasattr(res_null.params, "index") else [f"beta{i}" for i in range(len(res_null.params))]

    # Indices for the tested block in the full parameter vector.
    idx: List[int] = []
    for t in list(terms):
        if t in full_names:
            idx.append(full_names.index(t))
    if len(idx) == 0:
        return float("nan")

    idx_arr = np.array(idx, dtype=int)

    # Observed Wald (chi-square) from the full fit.
    b_hat = np.asarray(res_full.params, dtype=float)[idx_arr]
    V_sub = np.asarray(cov_full, dtype=float)[np.ix_(idx_arr, idx_arr)]
    W_obs = float(b_hat.T @ pinv(V_sub) @ b_hat)
    W_obs = max(W_obs, 0.0)

    # Null-embedded parameter vector.
    beta0 = _align_params_to_full(full_names, null_names, np.asarray(res_null.params, dtype=float))

    # Score observations (n x p) and Hessian (p x p) at beta0.
    try:
        score_obs = np.asarray(res_full.model.score_obs(beta0), dtype=float)
    except Exception:
        # Fallback for models without score_obs.
        mu = np.asarray(res_full.model.predict(beta0), dtype=float)
        y = np.asarray(res_full.model.endog, dtype=float)
        X = np.asarray(res_full.model.exog, dtype=float)
        score_obs = X * (y - mu)[:, None]

    try:
        hess = np.asarray(res_full.model.hessian(beta0), dtype=float)
        hess_inv = pinv(hess)
    except Exception:
        # Last resort: use normalized cov params (already inverse-ish scale).
        hess_inv = np.asarray(getattr(res_full, "normalized_cov_params"), dtype=float)

    rng = np.random.default_rng(int(seed))
    Wb = np.empty(int(n_boot), dtype=float)
    for b in range(int(n_boot)):
        w = _wild_weights(clusters, rng, weight=weight)
        # Weighted score sum (p,)
        s = w @ score_obs
        # One-step approximation to the full estimator under the wild score.
        # Newton step: beta = beta0 - H^{-1} s
        beta_star = beta0 - (hess_inv @ s)

        b_star = beta_star[idx_arr]
        W = float(b_star.T @ pinv(V_sub) @ b_star)
        Wb[b] = max(W, 0.0)

    # Conservative finite-sample correction.
    p_boot = (1.0 + float(np.sum(Wb >= W_obs))) / (float(n_boot) + 1.0)
    return float(min(max(p_boot, 0.0), 1.0))


def fit_binom_glm(y: Iterable, X: pd.DataFrame, *, link: str = "logit") -> sm.GLM:
    """Fit a binomial GLM under the working i.i.d. likelihood.

    Parameters
    ----------
    link:
        Link function for the binomial GLM. Supported values:
        - "logit" (default)
        - "probit"

    Notes
    -----
    This wrapper exists so the pipeline can run the reviewer-requested
    link-function sensitivity analysis (logit vs probit) without
    duplicating model code across modules.
    """
    y = np.asarray(y, dtype=float)
    X_ = X.astype(float)
    if y.size == 0 or X_.shape[0] == 0:
        raise ValueError(
            "fit_binom_glm received an empty dataset (n_obs=0). "
            "This typically means the split/variant filtering removed all rows. "
            "Check your disc definition, missingness filters, and train/test split."
        )
    link_key = str(link).strip().lower()
    if link_key in {"logit", "logistic"}:
        link_obj = sm.families.links.Logit()
    elif link_key in {"probit"}:
        link_obj = sm.families.links.Probit()
    else:
        raise ValueError(f"Unsupported link='{link}'. Use 'logit' or 'probit'.")

    return sm.GLM(y, X_, family=sm.families.Binomial(link=link_obj)).fit()


def _coerce_cluster_series(x: Iterable, *, name: str = "cluster") -> pd.Series:
    """Coerce a 1D cluster label vector into a string Series.

    We intentionally cast to string to avoid pandas treating mixed types as
    `object` and to make intersection-cluster construction stable.
    """

    s = x if isinstance(x, pd.Series) else pd.Series(x)
    if s.name is None:
        s = s.rename(name)
    return s.astype("string").fillna("(missing)")


def _intersection_codes(df_clusters: pd.DataFrame) -> tuple[np.ndarray, int]:
    """Return integer codes for the row-wise intersection clusters.

    Parameters
    ----------
    df_clusters:
        DataFrame with one column per clustering dimension.
    """

    # MultiIndex is a convenient, hashable representation of row-wise tuples.
    mi = pd.MultiIndex.from_frame(df_clusters.astype("string"), names=list(df_clusters.columns))
    codes, uniq = pd.factorize(mi, sort=False)
    return codes.astype(int), int(len(uniq))


def _sandwich_arrays_pinv(res) -> tuple[np.ndarray, np.ndarray]:
    """(score_obs, hessian_inv) arrays for sandwich covariance.

    We use a Moore–Penrose pseudoinverse fallback for the Hessian to keep the
    pipeline running under quasi-separation / near-collinearity.
    """

    if hasattr(res.model, "score_obs"):
        jac = res.model.score_obs(res.params)
    elif hasattr(res.model, "jac"):
        jac = res.model.jac(res.params)
    else:
        jac = res.model.wexog * res.wresid[:, None]

    if hasattr(res.model, "hessian"):
        hess = np.asarray(res.model.hessian(res.params), dtype=float)
        hess_inv = np.linalg.pinv(hess)
    else:
        # As a last resort, use the (already inverted) normalized covariance.
        hess_inv = np.asarray(getattr(res, "normalized_cov_params"), dtype=float)

    return np.asarray(jac, dtype=float), np.asarray(hess_inv, dtype=float)


def _cov_cluster_safe(res, g: np.ndarray, *, jac_hess: Optional[tuple[np.ndarray, np.ndarray]] = None) -> np.ndarray:
    """Statsmodels `cov_cluster` with a pseudo-inverse fallback."""

    from statsmodels.stats.sandwich_covariance import cov_cluster

    # If (scores, hessian_inv) are supplied, prefer the explicit form to avoid
    # repeated Hessian inversions inside statsmodels.
    if jac_hess is not None:
        jac, hess_inv = jac_hess
        return np.asarray(cov_cluster((jac, hess_inv), g), dtype=float)

    try:
        return np.asarray(cov_cluster(res, g), dtype=float)
    except np.linalg.LinAlgError:
        import warnings

        warnings.warn(
            "cluster_covariance: cov_cluster failed with a singular Hessian; "
            "using a Moore–Penrose pseudoinverse fallback. "
            "(This typically indicates quasi-separation or near-collinearity.)",
            RuntimeWarning,
        )

        jac, hess_inv = _sandwich_arrays_pinv(res)
        return np.asarray(cov_cluster((jac, hess_inv), g), dtype=float)


def cluster_covariance(
    res,
    clusters: Union[Iterable, pd.Series, pd.DataFrame, Sequence[Iterable]],
) -> tuple[np.ndarray, int]:
    """Cluster-robust covariance matrix (one-way or multiway clustering).

    Parameters
    ----------
    res:
        A fitted statsmodels result.
    clusters:
        Either:
        - a 1D vector / Series (one-way clustering), or
        - a DataFrame / sequence of 1D vectors (multiway clustering).

        For multiway clustering, we use the Cameron–Gelbach–Miller inclusion–
        exclusion estimator:

            V = \sum_{s \neq \emptyset} (-1)^{|s|+1} V_s

        where V_s is the one-way cluster covariance computed on the intersection
        clusters for subset s of clustering dimensions.

    Returns
    -------
    cov:
        Sandwich covariance matrix.
    n_clusters_eff:
        Effective cluster count used for small-sample df heuristics (e.g., t/F
        approximations). For multiway clustering we return the *minimum* number
        of unique clusters across the provided dimensions (after dropping
        degenerate dimensions with <= 1 unique cluster).

    Notes
    -----
    Statsmodels' ``cov_cluster`` computes the sandwich covariance using an explicit
    inversion of the model Hessian. For some GLM fits (e.g., quasi-separation or
    near-collinearity), the Hessian can be singular even when point estimates are
    available. In that case, we fall back to a Moore–Penrose pseudoinverse so the
    pipeline can still report conservative uncertainty estimates.
    """

    # -------------------------------
    # Normalize clusters input
    # -------------------------------
    if isinstance(clusters, pd.DataFrame):
        cdf = clusters.copy()
    elif isinstance(clusters, (list, tuple)) and len(clusters) > 0 and not isinstance(clusters, (pd.Series, np.ndarray)):
        cdf = pd.DataFrame({f"c{i}": _coerce_cluster_series(c, name=f"c{i}") for i, c in enumerate(clusters)})
    else:
        cdf = None

    # -------------------------------
    # One-way clustering
    # -------------------------------
    if cdf is None or cdf.shape[1] <= 1:
        gser = _coerce_cluster_series(clusters, name="cluster").astype("category")
        g = gser.cat.codes.to_numpy()
        G = int(np.unique(g).size)

        cov = _cov_cluster_safe(res, g)

        if not np.isfinite(cov).all():
            import warnings

            warnings.warn(
                "cluster_covariance: non-finite entries detected in cluster covariance matrix. "
                "Downstream SE/Wald statistics may be undefined for this fit.",
                RuntimeWarning,
            )

        return cov, G

    # -------------------------------
    # Multiway clustering (Cameron–Gelbach–Miller)
    # -------------------------------

    # Drop degenerate dimensions (all one cluster) to avoid pathological df.
    keep_cols: List[str] = []
    g_counts: List[int] = []
    for c in list(cdf.columns):
        nuniq = int(_coerce_cluster_series(cdf[c], name=c).nunique(dropna=False))
        if nuniq > 1:
            keep_cols.append(c)
            g_counts.append(nuniq)

    if len(keep_cols) <= 1:
        # Reduce to one-way if only one informative dimension remains.
        col = keep_cols[0] if keep_cols else cdf.columns[0]
        return cluster_covariance(res, cdf[col])

    cdf = cdf[keep_cols].copy()
    n_dims = int(cdf.shape[1])
    n_clusters_eff = int(min(g_counts)) if g_counts else int(cdf.iloc[:, 0].nunique())

    # Precompute (scores, hessian_inv) once, then use cov_cluster on each grouping.
    jac_hess = _sandwich_arrays_pinv(res)

    cov_total = np.zeros((int(len(res.params)), int(len(res.params))), dtype=float)

    # Inclusion–exclusion over non-empty subsets of dimensions.
    from itertools import combinations

    for r in range(1, n_dims + 1):
        sign = 1.0 if (r % 2 == 1) else -1.0
        for dims in combinations(range(n_dims), r):
            sub = cdf.iloc[:, list(dims)]
            if sub.shape[1] == 1:
                g = _coerce_cluster_series(sub.iloc[:, 0], name=sub.columns[0]).astype("category").cat.codes.to_numpy()
            else:
                g, _ = _intersection_codes(sub)
            cov_s = _cov_cluster_safe(res, g, jac_hess=jac_hess)
            cov_total += sign * cov_s

    if not np.isfinite(cov_total).all():
        import warnings

        warnings.warn(
            "cluster_covariance: non-finite entries detected in multiway cluster covariance matrix. "
            "Downstream SE/Wald statistics may be undefined for this fit.",
            RuntimeWarning,
        )

    return cov_total, n_clusters_eff
def tidy_coef_table(res, cov: np.ndarray, *, df_t: Optional[int] = None) -> pd.DataFrame:
    """Create a tidy coefficient table from a result and a covariance matrix."""
    params = np.asarray(res.params, dtype=float)
    se = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
    if df_t is None:
        df_t = np.nan
    terms = list(res.params.index) if hasattr(res.params, "index") else [f"beta{i}" for i in range(len(params))]
    tvals = params / np.where(se == 0, np.nan, se)
    return pd.DataFrame({"term": terms, "coef": params, "se": se, "t": tvals, "df_t": df_t})


def wald_joint(
    res,
    cov: np.ndarray,
    clusters_n: int,
    terms: Sequence[str],
    block: str,
    *,
    use_f: bool = True,
) -> WaldBlockResult:
    """Cluster-robust joint Wald test for a block of coefficients.

    The chi-square version is the canonical Wald statistic:
        W = b' V^{-1} b

    When `use_f=True` we also report a small-sample F approximation:
        F = W / q
        df_num = q
        df_denom = G - 1
    where q = number of tested coefficients and G = number of clusters.
    """

    from scipy.stats import chi2, f as f_dist

    names = list(res.params.index) if hasattr(res.params, "index") else [f"beta{i}" for i in range(len(res.params))]

    idx: List[int] = []
    missing: List[str] = []
    for t in list(terms):
        if t in names:
            idx.append(names.index(t))
        else:
            missing.append(t)

    if len(idx) == 0:
        raise ValueError(f"No requested terms found in model params for block '{block}'. Missing={missing}")

    b = np.asarray(res.params, dtype=float)[np.array(idx, dtype=int)]
    V = np.asarray(cov, dtype=float)[np.ix_(idx, idx)]

    Vinv = np.linalg.pinv(V)
    stat = float(b.T @ Vinv @ b)
    df = int(len(idx))
    p = float(chi2.sf(max(stat, 0.0), df))

    if use_f:
        df_num = df
        df_denom = max(int(clusters_n) - 1, 1)
        f_stat = float(stat / df_num) if df_num > 0 else float("nan")
        p_f = float(f_dist.sf(max(f_stat, 0.0), df_num, df_denom))
    else:
        df_num, df_denom, f_stat, p_f = df, max(int(clusters_n) - 1, 1), float("nan"), float("nan")

    return WaldBlockResult(
        block=block,
        stat_chi2=stat,
        df=df,
        p_value_chi2=p,
        f_stat=f_stat,
        df_num=df_num,
        df_denom=df_denom,
        p_value_f=p_f,
        n_clusters=int(clusters_n),
        terms=list(terms),
        missing_terms=missing,
    )


def bic_llf(res) -> float:
    """LLF-based BIC (lower is better).

    This matches the original archive's score:
        BIC = -2 * llf + k * log(n)

    where k is the number of free parameters (including intercept).
    """

    llf = float(res.llf)
    n = int(getattr(res, "nobs", len(np.atleast_1d(res.model.endog))))

    k = None
    if hasattr(res, "df_modelwc"):
        try:
            k = int(res.df_modelwc)
        except Exception:
            k = None
    if k is None and hasattr(res, "df_model"):
        try:
            k = int(res.df_model) + 1
        except Exception:
            k = None
    if k is None:
        k = int(np.size(getattr(res, "params", [])))

    return -2.0 * llf + k * math.log(max(n, 1))


def lr_test(res_full, res_restr, *, df_diff: int) -> tuple[float, int, float]:
    """Likelihood-ratio (deviance) test for nested models (working i.i.d.)."""

    from scipy.stats import chi2

    llf_full = float(res_full.llf)
    llf_restr = float(res_restr.llf)

    lr = 2.0 * (llf_full - llf_restr)
    p = float(chi2.sf(max(lr, 0.0), int(df_diff)))
    return float(lr), int(df_diff), p


def log_likelihood_from_probs(
    y: np.ndarray,
    p: np.ndarray,
    *,
    eps: float = 1e-12,
    weights: Optional[np.ndarray] = None,
) -> float:
    """Bernoulli log-likelihood (sum) for predicted probabilities.

    Parameters
    ----------
    weights:
        Optional non-negative weights (same length as y/p). Useful for
        multiway bootstrap schemes that operate via reweighting.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, 1.0 - eps)

    ll_i = y * np.log(p) + (1.0 - y) * np.log(1.0 - p)
    if weights is None:
        return float(np.sum(ll_i))

    w = np.asarray(weights, dtype=float)
    if w.shape[0] != ll_i.shape[0]:
        raise ValueError("weights must have the same length as y and p")
    # Be conservative: treat negative weights as invalid.
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    return float(np.sum(w * ll_i))


def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean((y - p) ** 2))


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson binomial confidence interval."""
    if n <= 0:
        return (0.0, 0.0)
    from scipy.stats import norm

    z = float(norm.ppf(1.0 - alpha / 2.0))
    phat = k / n
    denom = 1.0 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denom
    lo, hi = center - half, center + half
    return (max(min(lo, 1.0), 0.0), max(min(hi, 1.0), 0.0))
