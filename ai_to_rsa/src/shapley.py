from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import math
import numpy as np
import pandas as pd

from .modeling import add_const
from .stats import fit_binom_glm, log_likelihood_from_probs, parse_cluster_cols


@dataclass(frozen=True)
class ShapleyResult:
    disc_variant: str
    split: str  # usually "test" (value computed on held-out)
    metric: str
    ll_base: float
    ll_full: float
    delta_ll_full: float
    phi_offsets: float
    phi_geometry: float
    phi_decoder: float
    ratio_offsets: float
    ratio_geometry: float
    ratio_decoder: float


def _powerset(iterable: Sequence[str]) -> List[Tuple[str, ...]]:
    items = list(iterable)
    out: List[Tuple[str, ...]] = [()]
    for r in range(1, len(items) + 1):
        out.extend(list(combinations(items, r)))
    return out


def _subset_key(s: Iterable[str]) -> str:
    return "+".join(sorted(s)) if s else "(none)"


def fit_models_for_subsets(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    y_col: str,
    group_cols: Dict[str, List[str]],
    link: str = "logit",
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Fit a GLM for each subset of groups and evaluate on `test`.

    Returns a table with log-likelihoods and a dict of test predicted probs.
    """

    groups = list(group_cols.keys())
    subsets = _powerset(groups)

    y_tr = train[y_col].to_numpy(dtype=float)
    y_te = test[y_col].to_numpy(dtype=float)

    prob_cache: Dict[str, np.ndarray] = {}
    rows: List[Dict] = []

    for s in subsets:
        cols: List[str] = []
        for g in s:
            cols.extend(group_cols[g])
        # Keep deterministic column ordering
        cols = [c for c in cols if c in train.columns]
        X_tr = add_const(train[cols]) if cols else add_const(pd.DataFrame(index=train.index))
        X_te = add_const(test[cols]) if cols else add_const(pd.DataFrame(index=test.index))

        res = fit_binom_glm(y_tr, X_tr, link=link)
        p_te = np.asarray(res.predict(X_te), dtype=float)

        ll = log_likelihood_from_probs(y_te, p_te)
        key = _subset_key(s)
        prob_cache[key] = p_te
        rows.append({"subset": key, "groups": ",".join(s), "n_groups": len(s), "ll_test": ll})

    tab = pd.DataFrame(rows).sort_values(["n_groups", "subset"]).reset_index(drop=True)
    return tab, prob_cache


def shapley_from_values(value: Dict[str, float], groups: Sequence[str]) -> Dict[str, float]:
    """Compute Shapley values from a dictionary mapping subset->value.

    `value` keys must be produced by `_subset_key`.
    The empty subset is key "(none)".
    """

    n = len(groups)
    fact = math.factorial
    denom = fact(n)

    out: Dict[str, float] = {}
    for g in groups:
        phi = 0.0
        others = [h for h in groups if h != g]
        for r in range(0, len(others) + 1):
            for S in combinations(others, r):
                S_key = _subset_key(S)
                Sg_key = _subset_key(tuple(sorted(S + (g,))))
                vS = float(value[S_key])
                vSg = float(value[Sg_key])
                w = fact(len(S)) * fact(n - len(S) - 1) / denom
                phi += w * (vSg - vS)
        out[g] = float(phi)

    return out


def compute_shapley_threeway(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    disc_variant: str,
    y_col: str = "y",
    group_cols: Dict[str, List[str]],
    link: str = "logit",
    bootstrap: bool = True,
    n_boot: int = 300,
    seed: int = 0,
    cluster_cols: Union[str, Sequence[str]] = "worker",
    bootstrap_method: str = "cluster",
) -> Tuple[ShapleyResult, pd.DataFrame, Optional[pd.DataFrame]]:
    """Compute a 3-way Shapley attribution on held-out (test) log-likelihood.

    The value function is defined as:
        v(S) = LL_test(model using subset S) - LL_test(intercept-only)
    so v((none)) = 0 and Shapley values sum to v(all).

    Bootstrap options (evaluation uncertainty only; models are fit once on train):

    - bootstrap_method="cluster" (default): one-way cluster resampling using the
      *first* column in `cluster_cols`.
    - bootstrap_method="intersection": resample intersection clusters formed by
      the Cartesian intersection of all `cluster_cols` (e.g., speaker×game).
    - bootstrap_method="pigeonhole": multiway "pigeonhole" bootstrap via
      reweighting based on independently resampled cluster labels in each
      dimension. This is computationally cheap and keeps the expected sample size
      fixed via weight normalization.
    """

    groups = list(group_cols.keys())
    if len(groups) != 3:
        raise ValueError("This function is intended for exactly 3 groups.")

    tab_ll, prob_cache = fit_models_for_subsets(train, test, y_col=y_col, group_cols=group_cols, link=link)

    # Build value function on test
    ll_by_subset = {row["subset"]: float(row["ll_test"]) for _, row in tab_ll.iterrows()}
    ll_base = ll_by_subset["(none)"]

    value = {k: v - ll_base for k, v in ll_by_subset.items()}
    phi = shapley_from_values(value, groups)

    delta_full = value[_subset_key(tuple(groups))]
    total = float(delta_full)
    if total == 0.0 or not np.isfinite(total):
        ratios = {g: float("nan") for g in groups}
    else:
        ratios = {g: float(phi[g] / total) for g in groups}

    res = ShapleyResult(
        disc_variant=disc_variant,
        split="test",
        metric="log_likelihood (nats)",
        ll_base=float(ll_base),
        ll_full=float(ll_by_subset[_subset_key(tuple(groups))]),
        delta_ll_full=float(total),
        phi_offsets=float(phi.get('offsets', phi[groups[0]])),
        phi_geometry=float(phi[groups[1]]),
        phi_decoder=float(phi[groups[2]]),
        ratio_offsets=float(ratios.get('offsets', ratios[groups[0]])),
        ratio_geometry=float(ratios[groups[1]]),
        ratio_decoder=float(ratios[groups[2]]),
    )

    boot_df: Optional[pd.DataFrame] = None
    if bootstrap:
        rng = np.random.default_rng(seed)

        ccols = [c for c in parse_cluster_cols(cluster_cols) if c in test.columns]
        if not ccols:
            ccols = ["worker"]

        y = test[y_col].to_numpy(dtype=float)
        subset_keys = list(prob_cache.keys())

        method = str(bootstrap_method).strip().lower()
        if method not in {"cluster", "intersection", "pigeonhole"}:
            raise ValueError("bootstrap_method must be one of: 'cluster', 'intersection', 'pigeonhole'")

        # ------------------------------------------------------------------
        # Precompute cluster codes / index maps for fast resampling
        # ------------------------------------------------------------------
        idx_by_cluster: Optional[List[np.ndarray]] = None
        cluster_codes: Optional[np.ndarray] = None

        if method in {"cluster", "intersection"}:
            if method == "cluster":
                cl = test[ccols[0]].astype("string").fillna("(missing)")
                cluster_codes, uniq = pd.factorize(cl, sort=False)
            else:
                cl_df = test[ccols].astype("string").fillna("(missing)")
                mi = pd.MultiIndex.from_frame(cl_df, names=ccols)
                cluster_codes, uniq = pd.factorize(mi, sort=False)

            cluster_codes = np.asarray(cluster_codes, dtype=int)
            n_cl = int(len(uniq))
            # idx_by_cluster[k] = row indices for intersection cluster k
            idx_by_cluster = [np.nonzero(cluster_codes == k)[0] for k in range(n_cl)]

        # For pigeonhole: precompute per-dimension codes (0..Gd-1)
        pigeonhole_codes: List[np.ndarray] = []
        pigeonhole_sizes: List[int] = []
        if method == "pigeonhole":
            for c in ccols:
                codes_c, uniq_c = pd.factorize(test[c].astype("string").fillna("(missing)"), sort=False)
                pigeonhole_codes.append(np.asarray(codes_c, dtype=int))
                pigeonhole_sizes.append(int(len(uniq_c)))

        # ------------------------------------------------------------------
        # Bootstrap loop
        # ------------------------------------------------------------------
        boot_rows: List[Dict] = []
        n_obs = int(test.shape[0])

        for b in range(int(n_boot)):
            ll_b: Dict[str, float] = {}

            if method in {"cluster", "intersection"}:
                assert idx_by_cluster is not None
                n_cl = len(idx_by_cluster)
                sampled = rng.integers(0, n_cl, size=n_cl)
                idx = np.concatenate([idx_by_cluster[k] for k in sampled], axis=0)

                for sk in subset_keys:
                    p = prob_cache[sk]
                    ll_b[sk] = log_likelihood_from_probs(y[idx], p[idx])

            else:
                # Pigeonhole / multiway bootstrap via reweighting.
                # Sample cluster labels independently within each dimension, then
                # assign each observation the product of its per-dimension counts.
                w = np.ones(n_obs, dtype=float)
                for codes_c, Gc in zip(pigeonhole_codes, pigeonhole_sizes):
                    if Gc <= 1:
                        continue
                    sampled = rng.integers(0, Gc, size=Gc)
                    counts = np.bincount(sampled, minlength=Gc).astype(float)
                    w *= counts[codes_c]

                s = float(w.sum())
                if s > 0:
                    w *= (float(n_obs) / s)

                for sk in subset_keys:
                    p = prob_cache[sk]
                    ll_b[sk] = log_likelihood_from_probs(y, p, weights=w)

            ll0 = ll_b["(none)"]
            v_b = {k: ll_b[k] - ll0 for k in subset_keys}
            phi_b = shapley_from_values(v_b, groups)
            total_b = v_b[_subset_key(tuple(groups))]
            if total_b == 0.0 or not np.isfinite(total_b):
                ratio_b = {g: float("nan") for g in groups}
            else:
                ratio_b = {g: float(phi_b[g] / total_b) for g in groups}

            boot_rows.append(
                {
                    "boot": b,
                    "bootstrap_method": method,
                    "bootstrap_cluster_spec": "+".join(ccols),
                    "delta_ll_full": total_b,
                    f"phi_{groups[0]}": phi_b[groups[0]],
                    f"phi_{groups[1]}": phi_b[groups[1]],
                    f"phi_{groups[2]}": phi_b[groups[2]],
                    f"ratio_{groups[0]}": ratio_b[groups[0]],
                    f"ratio_{groups[1]}": ratio_b[groups[1]],
                    f"ratio_{groups[2]}": ratio_b[groups[2]],
                }
            )

        boot_df = pd.DataFrame(boot_rows)

    return res, tab_ll, boot_df


def summarize_bootstrap(boot_df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """Percentile summary (2.5%, 50%, 97.5%)."""

    qs = [0.025, 0.5, 0.975]
    rows = []
    for c in cols:
        v = boot_df[c].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            rows.append({"metric": c, "q025": np.nan, "q500": np.nan, "q975": np.nan})
        else:
            rows.append({"metric": c, "q025": float(np.quantile(v, qs[0])), "q500": float(np.quantile(v, qs[1])), "q975": float(np.quantile(v, qs[2]))})
    return pd.DataFrame(rows)
