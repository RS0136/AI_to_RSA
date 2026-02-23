# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from scipy import stats
from scipy.special import logsumexp
import matplotlib
import matplotlib.pyplot as plt


# -------------------------
# Math helpers
# -------------------------


def softmax_rows(logits: np.ndarray) -> np.ndarray:
    """Row-wise stable softmax for 2D arrays."""
    m = np.max(logits, axis=1, keepdims=True)
    ex = np.exp(logits - m)
    z = np.sum(ex, axis=1, keepdims=True)
    z = np.where(z <= 0, 1.0, z)
    return ex / z


def sample_categorical_rows(rng: np.random.Generator, probs: np.ndarray) -> np.ndarray:
    """Vectorized categorical sampling for row-stochastic probs (n,k)."""
    cum = np.cumsum(probs, axis=1)
    cum[:, -1] = 1.0
    r = rng.random(probs.shape[0])[:, None]
    return (cum > r).argmax(axis=1)


def zscore_train_apply(train: pd.Series, test: pd.Series) -> Tuple[pd.Series, pd.Series, float, float]:
    mu = float(train.mean())
    sd = float(train.std(ddof=0))
    if not np.isfinite(sd) or sd <= 0:
        sd = 1.0
    return (train - mu) / sd, (test - mu) / sd, mu, sd


# -------------------------
# Wilson CI for proportions
# -------------------------


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return (np.nan, np.nan)
    z = stats.norm.ppf(1 - alpha / 2)
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    half = (z * np.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def binom_ci_wilson(phat: float, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Wilson interval given phat and n (rounded to counts)."""
    k = int(np.round(phat * n))
    return wilson_ci(k, n, alpha=alpha)


# -------------------------
# Simulation config (A5)
# -------------------------


@dataclass
class SimConfig:
    seed: int = 0

    # Data size / structure
    n_trials: int = 20000
    n_workers: int = 600
    test_frac_workers: float = 0.2
    n_candidates: int = 15

    # Condition mixture
    cond_probs: Dict[str, float] = None  # {"close":...,"far":...,"split":...}

    # Geometry distances (positive)
    close_mu: float = 0.12
    close_sd: float = 0.04
    far_mu: float = 0.35
    far_sd: float = 0.06
    min_dist: float = 0.01

    # A5 knob: context prior P(w|c)
    prior_logit_bias_by_cond: Dict[str, float] = None

    # A2-like knob: speaker precision by condition
    alpha_by_cond: Dict[str, float] = None

    # Production cost proxy
    cost_per_char: float = 0.02

    # Candidate length model
    len_base_by_cond: Dict[str, float] = None
    len_hard_coef: float = 18.0
    len_min: int = 5
    len_mean_for_sem: float = 35.0
    verb_scale: float = 8.0

    # Semantics [[u]](w): unnormalized Gaussian compatibility in a 1D feature space
    sem_base_sigma: float = 0.18
    sem_len_coef: float = 1.4
    sem_sigma_min: float = 0.03

    # Candidate prototype noise around target
    proto_noise_base: float = 0.08
    proto_noise_hard_coef: float = 0.9

    # Worker random effects
    worker_skill_sd: float = 0.6
    worker_verb_sd: float = 0.4
    worker_alpha_sd: float = 0.2

    # ------------------------------------------------------------
    # Extensions for "violation" calibration experiments
    # ------------------------------------------------------------

    # Candidate set differences (size / quality) by condition
    n_candidates_by_cond: Dict[str, int] = None
    candidate_quality_mult_by_cond: Dict[str, float] = None  # multiplies proto noise ( <1 => better candidates )

    # Context-dependent semantics: allow length→semantics coupling to vary by condition
    sem_len_coef_by_cond: Dict[str, float] = None

    # Ceiling compression / nonlinearity in the observed success probability
    # Implemented as: p_obs = 1 - (1 - p_internal) ** gamma_cond
    # gamma = 1 => identity; gamma > 1 pushes probabilities toward 1 (ceiling).
    ceiling_gamma_by_cond: Dict[str, float] = None

    # Unobserved confounding example: non-random condition assignment by worker
    confound_conditions_by_worker: bool = False
    cond_logit_skill_coef: float = 0.0
    cond_logit_verb_coef: float = 0.0
    cond_base_logits: Dict[str, float] = None

    # Check A5 identity
    check_a5_identity: bool = True
    check_n: int = 150

    def __post_init__(self):
        if self.cond_probs is None:
            self.cond_probs = {"close": 0.36, "far": 0.31, "split": 0.33}
        if self.prior_logit_bias_by_cond is None:
            self.prior_logit_bias_by_cond = {"close": 0.0, "far": 0.0, "split": 0.0}
        if self.alpha_by_cond is None:
            self.alpha_by_cond = {"close": 2.0, "far": 2.0, "split": 2.0}
        if self.len_base_by_cond is None:
            self.len_base_by_cond = {"close": 42.0, "far": 28.0, "split": 34.0}

        if self.n_candidates_by_cond is None:
            self.n_candidates_by_cond = {"close": int(self.n_candidates), "far": int(self.n_candidates), "split": int(self.n_candidates)}
        if self.candidate_quality_mult_by_cond is None:
            self.candidate_quality_mult_by_cond = {"close": 1.0, "far": 1.0, "split": 1.0}
        if self.ceiling_gamma_by_cond is None:
            self.ceiling_gamma_by_cond = {"close": 1.0, "far": 1.0, "split": 1.0}
        if self.cond_base_logits is None:
            # Use log probabilities as base logits by default.
            probs = self.cond_probs if self.cond_probs is not None else {"close": 1.0, "far": 1.0, "split": 1.0}
            den = float(sum(float(v) for v in probs.values()))
            den = den if den > 0 else 1.0
            self.cond_base_logits = {k: float(np.log(float(v) / den + 1e-12)) for k, v in probs.items()}


def simulate_dataset_a5(cfg: SimConfig) -> pd.DataFrame:
    """Generate observed outcome-level dataset with A5 holding at L0 level."""
    rng = np.random.default_rng(cfg.seed)
    conds = ["close", "far", "split"]
    p = np.array([cfg.cond_probs[c] for c in conds], dtype=float)
    p = p / p.sum()

    # Workers
    worker_ids = rng.integers(0, cfg.n_workers, size=cfg.n_trials)
    w_skill = rng.normal(0.0, cfg.worker_skill_sd, size=cfg.n_workers)
    w_verb = rng.normal(0.0, cfg.worker_verb_sd, size=cfg.n_workers)
    w_alpha = rng.normal(0.0, cfg.worker_alpha_sd, size=cfg.n_workers)

    # Conditions
    if not bool(cfg.confound_conditions_by_worker):
        cond_ids = rng.choice(3, size=cfg.n_trials, p=p)
    else:
        # Non-random condition assignment by worker (simple confounding toy).
        base = np.array([cfg.cond_base_logits[c] for c in conds], dtype=float)
        skill_term = float(cfg.cond_logit_skill_coef) * w_skill[worker_ids]
        verb_term = float(cfg.cond_logit_verb_coef) * w_verb[worker_ids]
        logits = np.stack(
            [
                np.full(cfg.n_trials, base[0], dtype=float),
                base[1] + skill_term + verb_term,
                np.full(cfg.n_trials, base[2], dtype=float),
            ],
            axis=1,
        )
        probs = softmax_rows(logits)
        cond_ids = sample_categorical_rows(rng, probs)

    cond_arr = np.array([conds[i] for i in cond_ids], dtype=object)

    # Target positions in 1D
    tpos = rng.uniform(0.0, 1.0, size=cfg.n_trials)

    def _sample_pos_normal(mu: float, sd: float, n: int) -> np.ndarray:
        x = rng.normal(mu, sd, size=n)
        return np.clip(x, cfg.min_dist, None)

    d1 = np.empty(cfg.n_trials, dtype=float)
    d2 = np.empty(cfg.n_trials, dtype=float)

    idx_close = np.where(cond_ids == 0)[0]
    idx_far = np.where(cond_ids == 1)[0]
    idx_split = np.where(cond_ids == 2)[0]

    if len(idx_close) > 0:
        tmp = _sample_pos_normal(cfg.close_mu, cfg.close_sd, len(idx_close) * 2).reshape(-1, 2)
        d1[idx_close], d2[idx_close] = tmp[:, 0], tmp[:, 1]

    if len(idx_far) > 0:
        tmp = _sample_pos_normal(cfg.far_mu, cfg.far_sd, len(idx_far) * 2).reshape(-1, 2)
        d1[idx_far], d2[idx_far] = tmp[:, 0], tmp[:, 1]

    if len(idx_split) > 0:
        a = _sample_pos_normal(cfg.close_mu, cfg.close_sd, len(idx_split))
        b = _sample_pos_normal(cfg.far_mu, cfg.far_sd, len(idx_split))
        swap = rng.random(len(idx_split)) < 0.5
        d1[idx_split] = np.where(swap, a, b)
        d2[idx_split] = np.where(swap, b, a)

    # Random left/right placement
    s1 = rng.choice([-1.0, 1.0], size=cfg.n_trials)
    s2 = rng.choice([-1.0, 1.0], size=cfg.n_trials)
    d1pos = tpos + s1 * d1
    d2pos = tpos + s2 * d2

    clarity = 0.5 * (np.abs(tpos - d1pos) + np.abs(tpos - d2pos))
    disc = -np.minimum(np.abs(tpos - d1pos), np.abs(tpos - d2pos))

    # Hardness scaling (unitless)
    hard = 1.0 / (clarity + 1e-3)
    hard_norm = hard / np.median(hard)

    # Candidate lengths
    n_cand_by_cond = np.array([int(cfg.n_candidates_by_cond[c]) for c in conds], dtype=int)
    K_i = n_cand_by_cond[cond_ids]
    K_max = int(np.max(n_cand_by_cond))
    if K_max <= 0:
        raise ValueError("n_candidates_by_cond must be positive")

    len_base_arr = np.array([cfg.len_base_by_cond[c] for c in conds], dtype=float)
    lam_len = len_base_arr[cond_ids] + cfg.len_hard_coef * hard_norm + cfg.verb_scale * w_verb[worker_ids]
    lam_len = np.clip(lam_len, 2.0, None)

    lengths = rng.poisson(lam=lam_len[:, None], size=(cfg.n_trials, K_max)).astype(int)
    lengths = np.maximum(lengths, cfg.len_min)

    active = (np.arange(K_max)[None, :] < K_i[:, None])

    # Utterance specificity
    len_centered = (lengths - cfg.len_mean_for_sem) / max(cfg.len_mean_for_sem, 1.0)
    if cfg.sem_len_coef_by_cond is None:
        sem_len_coef_trial = float(cfg.sem_len_coef) * np.ones(cfg.n_trials, dtype=float)
    else:
        sem_len_arr = np.array([float(cfg.sem_len_coef_by_cond[c]) for c in conds], dtype=float)
        sem_len_coef_trial = sem_len_arr[cond_ids]

    sigma_u = cfg.sem_base_sigma * np.exp(-(sem_len_coef_trial[:, None]) * len_centered)
    sigma_u = np.clip(sigma_u, cfg.sem_sigma_min, None)

    # Utterance prototypes
    qmult_arr = np.array([float(cfg.candidate_quality_mult_by_cond[c]) for c in conds], dtype=float)
    qmult = qmult_arr[cond_ids]
    sigma_proto = cfg.proto_noise_base * (1.0 + cfg.proto_noise_hard_coef * hard_norm) * np.exp(-w_skill[worker_ids])
    sigma_proto = sigma_proto * qmult
    proto = tpos[:, None] + rng.normal(0.0, sigma_proto[:, None], size=(cfg.n_trials, K_max))

    # Referent positions: (target, d1, d2)
    refpos = np.stack([tpos, d1pos, d2pos], axis=1)

    # log semantics: -0.5 ((refpos - proto)/sigma)^2
    diff = proto[:, :, None] - refpos[:, None, :]
    log_sem = -0.5 * (diff / sigma_u[:, :, None]) ** 2

    # A5 prior: weights [exp(bias), 1, 1]
    bias_arr = np.array([cfg.prior_logit_bias_by_cond[c] for c in conds], dtype=float)
    b = bias_arr[cond_ids]
    w0 = np.exp(b)
    den = w0 + 2.0
    logP0 = np.log(w0) - np.log(den)
    logP1 = -np.log(den)
    log_prior = np.stack([logP0, logP1, logP1], axis=1)

    # Literal listener: logw = log_sem + log_prior ; L0 = softmax(logw)
    logw = log_sem + log_prior[:, None, :]
    kappa = logsumexp(logw, axis=2)               # κ(u,c)
    lnL0_target = logw[:, :, 0] - kappa

    # Cost
    cost = cfg.cost_per_char * lengths

    # Speaker utility in A5 rewrite form: log_sem_target - (cost + κ)
    util = log_sem[:, :, 0] - (cost + kappa)

    # Inactive candidates (beyond K_i for each trial) get effectively -inf utility
    util = np.where(active, util, -1e9)

    # Speaker precision (positive)
    alpha_arr = np.array([cfg.alpha_by_cond[c] for c in conds], dtype=float)
    alpha = alpha_arr[cond_ids] * np.exp(w_alpha[worker_ids])
    logits_u = alpha[:, None] * util
    q_u = softmax_rows(logits_u)

    # Optional A5 identity check
    if cfg.check_a5_identity and cfg.check_n > 0:
        ii = rng.choice(cfg.n_trials, size=min(cfg.check_n, cfg.n_trials), replace=False)
        util_direct = lnL0_target[ii, :] - cost[ii, :]
        # If the candidate set size is condition-dependent, apply the same
        # "inactive" mask to the direct-utility computation.
        try:
            util_direct = np.where(active[ii, :], util_direct, -1e9)
        except Exception:
            pass
        q_direct = softmax_rows(alpha[ii, None] * util_direct)
        q_rewrite = q_u[ii, :]
        maxdiff = float(np.max(np.abs(q_direct - q_rewrite)))
        if maxdiff > 1e-10:
            raise RuntimeError(f"A5 identity check failed: max |Δq| = {maxdiff:g}")

    # Sample utterance
    u_star = sample_categorical_rows(rng, q_u)

    # Listener choice under realized utterance
    logw_star = logw[np.arange(cfg.n_trials), u_star, :]
    probs_y = softmax_rows(logw_star)

    # Observed success probability (optionally with ceiling compression)
    p_internal = probs_y[:, 0]
    gamma_arr = np.array([float(cfg.ceiling_gamma_by_cond[c]) for c in conds], dtype=float)
    gamma = gamma_arr[cond_ids]
    # p_obs = 1 - (1-p)^gamma ; identity when gamma==1
    p_obs = 1.0 - np.power(np.clip(1.0 - p_internal, 0.0, 1.0), gamma)
    p_obs = np.clip(p_obs, 0.0, 1.0)
    y = rng.binomial(1, p_obs).astype(int)
    len_obs = lengths[np.arange(cfg.n_trials), u_star].astype(float)

    return pd.DataFrame({
        "y": y,
        "condition": cond_arr,
        "disc": disc,
        "clarity": clarity,
        "len_chars": len_obs,
        "worker": worker_ids.astype(int),
    })


# -------------------------
# Split and standardize
# -------------------------


def worker_disjoint_split(df: pd.DataFrame, test_frac_workers: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    workers = df["worker"].unique()
    rng.shuffle(workers)
    n_test = int(np.round(test_frac_workers * len(workers)))
    test_workers = set(workers[:n_test])
    is_test = df["worker"].isin(test_workers)
    return df.loc[~is_test].copy(), df.loc[is_test].copy()


def standardize_for_glm(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, float]]]:
    train = train.copy()
    test = test.copy()
    train["condition"] = pd.Categorical(train["condition"], categories=["close", "far", "split"])
    test["condition"] = pd.Categorical(test["condition"], categories=["close", "far", "split"])

    zinfo: Dict[str, Dict[str, float]] = {}
    for col, zcol in [("disc", "zdisc"), ("clarity", "zclarity"), ("len_chars", "zlen")]:
        train[zcol], test[zcol], mu, sd = zscore_train_apply(train[col], test[col])
        zinfo[zcol] = {"mean_train": mu, "sd_train": sd}
    return train, test, zinfo


# -------------------------
# GLM + cluster-robust Wald block tests
# -------------------------


@dataclass
class WaldResult:
    stat: float
    df: int
    p_chi2: float
    f_stat: float
    df_denom: int
    p_f: float


def _wald_block(params: np.ndarray, cov: np.ndarray, idx: List[int], n_clusters: int) -> WaldResult:
    b = params[idx]
    V = cov[np.ix_(idx, idx)]
    Vinv = np.linalg.pinv(V)
    W = float(b.T @ Vinv @ b)
    W_pos = max(W, 0.0)  # conservative for indefinite V
    q = len(idx)
    p_chi2 = 1.0 - stats.chi2.cdf(W_pos, df=q)
    df_denom = max(n_clusters - 1, 1)
    f_stat = W_pos / max(q, 1)
    p_f = 1.0 - stats.f.cdf(f_stat, dfn=q, dfd=df_denom)
    return WaldResult(stat=W_pos, df=q, p_chi2=p_chi2, f_stat=f_stat, df_denom=df_denom, p_f=p_f)


def fit_models_and_tests(df: pd.DataFrame) -> Dict[str, WaldResult]:
    df = df.copy()
    cond_term = 'C(condition, Treatment(reference="close"))'
    f_base = f"y ~ zdisc + zclarity + zlen + {cond_term}"
    f_full = f"y ~ (zdisc + zclarity + zlen) * {cond_term}"

    y_b, X_b = patsy.dmatrices(f_base, df, return_type="dataframe")
    y_f, X_f = patsy.dmatrices(f_full, df, return_type="dataframe")

    groups = df["worker"].to_numpy()
    n_clusters = int(pd.Series(groups).nunique())

    res_base = sm.GLM(y_b, X_b, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": groups}
    )
    res_full = sm.GLM(y_f, X_f, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": groups}
    )

    names_base = list(res_base.model.exog_names)
    names_full = list(res_full.model.exog_names)

    def idx_by_pred(names: List[str], pred) -> List[int]:
        return [i for i, nm in enumerate(names) if pred(nm)]

    idx_cond_base = idx_by_pred(
        names_base,
        lambda nm: ("C(condition" in nm) and ("[T." in nm) and (":" not in nm)
    )
    idx_cond_full = idx_by_pred(
        names_full,
        lambda nm: ("C(condition" in nm) and ("[T." in nm) and (":" not in nm)
    )

    def idx_interaction(var: str) -> List[int]:
        return idx_by_pred(
            names_full,
            lambda nm: (f"{var}:" in nm) and ("C(condition" in nm) and ("[T." in nm)
        )

    idx_disc = idx_interaction("zdisc")
    idx_clarity = idx_interaction("zclarity")
    idx_len = idx_interaction("zlen")
    idx_full = sorted(set(idx_disc + idx_clarity + idx_len))

    p_base = res_base.params.to_numpy()
    V_base = res_base.cov_params().to_numpy()
    p_full = res_full.params.to_numpy()
    V_full = res_full.cov_params().to_numpy()

    return {
        "A5_outcome_base": _wald_block(p_base, V_base, idx_cond_base, n_clusters=n_clusters),
        "A2_outcome_disc": _wald_block(p_full, V_full, idx_disc, n_clusters=n_clusters),
        "A2_outcome_clarity": _wald_block(p_full, V_full, idx_clarity, n_clusters=n_clusters),
        "A2_outcome_len": _wald_block(p_full, V_full, idx_len, n_clusters=n_clusters),
        "A2_outcome_full": _wald_block(p_full, V_full, idx_full, n_clusters=n_clusters),
        "A5_outcome|A2_outcome_full": _wald_block(p_full, V_full, idx_cond_full, n_clusters=n_clusters),
    }


# -------------------------
# Study runner
# -------------------------


def run_once(cfg: SimConfig) -> Dict[str, object]:
    df = simulate_dataset_a5(cfg)
    train, test = worker_disjoint_split(df, cfg.test_frac_workers, seed=cfg.seed + 12345)
    train, test, zinfo = standardize_for_glm(train, test)
    out = {
        "df": df,
        "train": train,
        "test": test,
        "zinfo": zinfo,
        "tests_train": fit_models_and_tests(train),
        "tests_test": fit_models_and_tests(test),
    }
    return out


def run_many(base_cfg: SimConfig, n_rep: int, alpha: float = 0.05) -> pd.DataFrame:
    rows = []
    for r in range(n_rep):
        cfg = SimConfig(**{**asdict(base_cfg), "seed": base_cfg.seed + r, "check_a5_identity": (r == 0)})
        res = run_once(cfg)
        for split_name, tests in [("train", res["tests_train"]), ("test", res["tests_test"])]:
            for block, wr in tests.items():
                rows.append({
                    "rep": r,
                    "split": split_name,
                    "block": block,
                    "stat": wr.stat,
                    "df": wr.df,
                    "p_chi2": wr.p_chi2,
                    "reject": (wr.p_chi2 < alpha),
                })
    return pd.DataFrame(rows)


def summarize_rejections(df_res: pd.DataFrame) -> pd.DataFrame:
    return (df_res
            .groupby(["split", "block"], as_index=False)
            .agg(reject_rate=("reject", "mean"),
                 mean_stat=("stat", "mean"),
                 median_p=("p_chi2", "median"),
                 n=("reject", "size")))


def confusion_matrix_from_rejections(
    df_res: pd.DataFrame,
    *,
    truth_positive: bool,
    block: str = "A2_outcome_full",
    split: str = "test",
) -> pd.DataFrame:
    """Compute TP/FP/TN/FN counts and proportions for a given block.

    Parameters
    ----------
    truth_positive:
        Ground-truth label for the regime: True means the assumption is violated
        and we *want* the diagnostic to reject (a "positive" case).
        False means the assumption holds and we *do not* want to reject.
    """

    sub = df_res[(df_res["split"] == split) & (df_res["block"] == block)].copy()
    if sub.empty:
        return pd.DataFrame(
            [
                {
                    "split": split,
                    "block": block,
                    "truth_positive": bool(truth_positive),
                    "n": 0,
                    "TP": 0,
                    "FP": 0,
                    "TN": 0,
                    "FN": 0,
                    "TP_rate": np.nan,
                    "FP_rate": np.nan,
                    "TN_rate": np.nan,
                    "FN_rate": np.nan,
                    "false_rejection_rate": np.nan,
                    "false_nonrejection_rate": np.nan,
                }
            ]
        )

    rej = sub["reject"].astype(bool).to_numpy()
    n = int(rej.size)
    pos = bool(truth_positive)

    if pos:
        # True positive / false negative are defined only on positive (violation) regimes.
        TP = int(np.sum(rej))
        FN = int(np.sum(~rej))
        FP = 0
        TN = 0
    else:
        # False positive / true negative are defined only on negative (assumption-holds) regimes.
        TP = 0
        FN = 0
        FP = int(np.sum(rej))
        TN = int(np.sum(~rej))

    # Rates as proportions of n (so they sum to 1 within this truth class).
    TP_rate = TP / n if n else np.nan
    FP_rate = FP / n if n else np.nan
    TN_rate = TN / n if n else np.nan
    FN_rate = FN / n if n else np.nan

    # Diagnostic-oriented error rates
    # - False rejection: reject when assumption holds (i.e., FP rate when truth_positive=False)
    # - False non-rejection: fail to reject when assumption is violated (i.e., FN rate when truth_positive=True)
    false_rej = FP_rate if not pos else np.nan
    false_nonrej = FN_rate if pos else np.nan

    return pd.DataFrame(
        [
            {
                "split": split,
                "block": block,
                "truth_positive": pos,
                "n": n,
                "TP": TP,
                "FP": FP,
                "TN": TN,
                "FN": FN,
                "TP_rate": TP_rate,
                "FP_rate": FP_rate,
                "TN_rate": TN_rate,
                "FN_rate": FN_rate,
                "false_rejection_rate": false_rej,
                "false_nonrejection_rate": false_nonrej,
            }
        ]
    )


# -------------------------
# Plotting (grayscale, dpi=1200, no titles)
# -------------------------


def _savefig(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=1200)
    plt.close()


def plot_condition_success(df: pd.DataFrame, outpath: str) -> pd.DataFrame:
    """Bar plot of success rate by condition with Wilson 95% CI error bars."""
    cond_order = ["close", "far", "split"]
    rows = []
    for c in cond_order:
        sub = df[df["condition"] == c]
        n = int(len(sub))
        k = int(sub["y"].sum())
        phat = k / n if n else np.nan
        lo, hi = wilson_ci(k, n)
        rows.append({"condition": c, "n": n, "acc": phat, "lo": lo, "hi": hi})
    tab = pd.DataFrame(rows)

    x = np.arange(len(cond_order))
    y = tab["acc"].to_numpy()
    yerr = np.vstack([y - tab["lo"].to_numpy(), tab["hi"].to_numpy() - y])

    plt.figure(figsize=(4.2, 3.2))
    plt.bar(x, y, color="0.25", edgecolor="0.15", linewidth=0.8)
    plt.errorbar(x, y, yerr=yerr, fmt="none", ecolor="0.05", elinewidth=0.9, capsize=3)
    plt.xticks(x, cond_order, rotation=20)
    plt.ylim(0, 1.0)
    plt.ylabel("Accuracy (proportion; unitless)")
    plt.xlabel("Condition")
    _savefig(outpath)
    return tab


def plot_rejection_rates(df_res: pd.DataFrame, outpath: str, blocks: List[str]) -> pd.DataFrame:
    """Bar plot of rejection rates across reps, with Wilson CI over reps (binomial n = #reps)."""
    # rejection is computed per rep; summarize over reps
    summ = (df_res[df_res["block"].isin(blocks)]
            .groupby(["split", "block"], as_index=False)
            .agg(rej=("reject", "mean"), nrep=("reject", "size")))

    # CI over #rep trials (unitless proportion)
    lo_list, hi_list = [], []
    for _, r in summ.iterrows():
        lo, hi = binom_ci_wilson(float(r["rej"]), int(r["nrep"]))
        lo_list.append(lo)
        hi_list.append(hi)
    summ["lo"], summ["hi"] = lo_list, hi_list

    # Plot: split facets vertically
    splits = ["train", "test"]
    fig, axes = plt.subplots(2, 1, figsize=(6.0, 4.8), sharex=True)
    for ax, sp in zip(axes, splits):
        sub = summ[summ["split"] == sp].set_index("block").loc[blocks].reset_index()
        x = np.arange(len(blocks))
        y = sub["rej"].to_numpy()
        yerr = np.vstack([y - sub["lo"].to_numpy(), sub["hi"].to_numpy() - y])
        # Numerical guard: Wilson intervals derived from rounded counts can
        # occasionally yield tiny negative error bars. Clip to 0 for plotting.
        yerr = np.clip(yerr, 0.0, None)
        ax.bar(x, y, color="0.35", edgecolor="0.15", linewidth=0.8)
        ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor="0.05", elinewidth=0.9, capsize=3)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel(f"Reject rate ({sp}; proportion)\n(unitless)")
        ax.grid(axis="y", linestyle="-", linewidth=0.3, color="0.85")
    axes[-1].set_xticks(np.arange(len(blocks)))
    axes[-1].set_xticklabels(blocks, rotation=25, ha="right")
    axes[-1].set_xlabel("Block test")
    plt.tight_layout()
    plt.savefig(outpath, dpi=1200)
    plt.close()
    return summ


def plot_wald_stats_example(test_results: Dict[str, WaldResult], outpath: str, blocks: List[str]) -> pd.DataFrame:
    """Single-run Wald chi2 bar plot with no title, grayscale."""
    rows = []
    for b in blocks:
        wr = test_results[b]
        rows.append({"block": b, "chi2": wr.stat, "df": wr.df, "p": wr.p_chi2})
    tab = pd.DataFrame(rows)
    x = np.arange(len(blocks))
    y = tab["chi2"].to_numpy()
    plt.figure(figsize=(6.0, 3.0))
    plt.bar(x, y, color="0.25", edgecolor="0.15", linewidth=0.8)
    plt.xticks(x, blocks, rotation=25, ha="right")
    plt.ylabel("Wald χ² (unitless)")
    plt.xlabel("Block test")
    plt.grid(axis="y", linestyle="-", linewidth=0.3, color="0.85")
    _savefig(outpath)
    return tab


def plot_sweep_lines(
    summ: pd.DataFrame,
    *,
    outpath: str,
    x_col: str,
    blocks: List[str],
    split: str = "test",
) -> None:
    """Line plot of rejection rates across a sweep of a single knob.

    The plot uses line styles (not colors) to remain legible in grayscale.
    """

    sub = summ[(summ["split"] == split) & (summ["block"].isin(blocks))].copy()
    if sub.empty:
        return

    # Stable ordering
    xs = sorted(sub[x_col].unique().tolist(), key=lambda v: float(v))

    linestyles = ["-", "--", ":", "-."]
    markers = ["o", "s", "^", "D"]

    plt.figure(figsize=(6.2, 3.8))
    for j, blk in enumerate(blocks):
        tmp = sub[sub["block"] == blk].set_index(x_col).reindex(xs)
        y = tmp["reject_rate"].to_numpy(dtype=float)
        plt.plot(
            xs,
            y,
            linestyle=linestyles[j % len(linestyles)],
            marker=markers[j % len(markers)],
            linewidth=1.2,
            markersize=4,
            label=str(blk),
        )

    plt.ylim(0.0, 1.0)
    plt.xlabel(x_col)
    plt.ylabel(f"Reject rate ({split}; proportion)\n(unitless)")
    plt.grid(axis="y", linestyle="-", linewidth=0.3, color="0.85")
    plt.legend(frameon=False, fontsize=8)
    _savefig(outpath)


def run_violation_sweep(args) -> None:
    """Run a calibration sweep to see which violation types trigger A2_outcome rejection.

    This starts from a "strict" baseline where:
      - alpha is equal across conditions
      - P(w|c) is uniform (no prior shift)
      - length affects *only* cost (sem_len_coef = 0)

    and then perturbs one mechanism at a time.
    """

    os.makedirs(args.outdir, exist_ok=True)

    base_cfg = SimConfig(
        seed=args.seed,
        n_trials=args.n_trials,
        n_workers=args.n_workers,
        n_candidates=args.n_candidates,
        test_frac_workers=args.test_frac_workers,
        prior_logit_bias_by_cond={"close": 0.0, "far": 0.0, "split": 0.0},
        alpha_by_cond={"close": 2.0, "far": 2.0, "split": 2.0},
        sem_len_coef=0.0,
    )

    sweeps = [
        # 1) Scale difference (speaker precision) by condition
        {
            "violation": "scale_difference_alpha",
            "x_col": "alpha_log_delta",
            "levels": [0.0, 0.25, 0.5, 0.75, 1.0],
            "make_cfg": lambda d: {
                "alpha_by_cond": {
                    "close": 2.0,
                    "far": float(2.0 * np.exp(-float(d))),
                    "split": float(2.0 * np.exp(float(d))),
                }
            },
        },
        # 2) Ceiling compression (nonlinear measurement of success)
        {
            "violation": "ceiling_compression",
            "x_col": "gamma_far",
            "levels": [1.0, 1.25, 1.5, 2.0, 3.0],
            "make_cfg": lambda g: {
                "ceiling_gamma_by_cond": {"close": 1.0, "far": float(g), "split": 1.0}
            },
        },
        # 3) Candidate set size differences
        {
            "violation": "candidate_set_size",
            "x_col": "K_far",
            "levels": [int(args.n_candidates), max(int(args.n_candidates) - 3, 2), max(int(args.n_candidates) - 6, 2), max(int(args.n_candidates) - 9, 2)],
            "make_cfg": lambda K: {
                "n_candidates_by_cond": {"close": int(args.n_candidates), "far": int(K), "split": int(args.n_candidates)}
            },
        },
        # 4) Candidate set quality differences (via prototype noise multiplier)
        {
            "violation": "candidate_set_quality",
            "x_col": "proto_noise_mult_far",
            "levels": [1.0, 0.8, 0.6, 0.4, 0.2],
            "make_cfg": lambda m: {
                "candidate_quality_mult_by_cond": {"close": 1.0, "far": float(m), "split": 1.0}
            },
        },
        # 5) Context-dependent semantics (length sharpens semantics only in far)
        {
            "violation": "context_dependent_semantics",
            "x_col": "sem_len_far",
            "levels": [0.0, 0.5, 1.0, 1.4, 2.0],
            "make_cfg": lambda v: {
                "sem_len_coef_by_cond": {"close": 0.0, "far": float(v), "split": 0.0}
            },
        },
        # 6) Unobserved confounding: condition assignment depends on worker skill/verbosity
        {
            "violation": "unobserved_confounding_worker",
            "x_col": "skill_logit_coef",
            "levels": [0.0, 0.5, 1.0, 2.0],
            "make_cfg": lambda c: {
                "confound_conditions_by_worker": True,
                "cond_logit_skill_coef": float(c),
                # Make verbose workers *less* likely to be assigned to far.
                "cond_logit_verb_coef": float(-0.4 * float(c)),
            },
        },
    ]

    blocks = ["A2_outcome_disc", "A2_outcome_clarity", "A2_outcome_len", "A2_outcome_full", "A5_outcome_base", "A5_outcome|A2_outcome_full"]
    all_rows: List[pd.DataFrame] = []

    for spec in sweeps:
        vio = str(spec["violation"])
        x_col = str(spec["x_col"])
        levels = list(spec["levels"])
        make_cfg = spec["make_cfg"]

        per_level = []
        for lv in levels:
            overrides = make_cfg(lv) if callable(make_cfg) else {}
            cfg = SimConfig(**{**asdict(base_cfg), **overrides, "seed": int(args.seed)})
            df_rep = run_many(cfg, n_rep=args.n_rep)
            summ = summarize_rejections(df_rep)
            summ.insert(0, "violation", vio)
            summ.insert(1, x_col, lv)
            per_level.append(summ)

        out = pd.concat(per_level, ignore_index=True)
        out_path = os.path.join(args.outdir, f"sweep_{vio}_reject_summary.csv")
        out.to_csv(out_path, index=False)

        # Plot A2_outcome rejection rates on TEST split
        fig_path = os.path.join(args.outdir, f"sweep_{vio}_a2_reject_test.png")
        plot_sweep_lines(out, outpath=fig_path, x_col=x_col, blocks=["A2_outcome_disc", "A2_outcome_clarity", "A2_outcome_len", "A2_outcome_full"], split="test")

        all_rows.append(out)

    all_df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    all_path = os.path.join(args.outdir, "violation_sweep_all_reject_summary.csv")
    all_df.to_csv(all_path, index=False)


# -------------------------
# Reproducibility bundle writers
# -------------------------


def write_pip_freeze(outpath: str) -> None:
    try:
        txt = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    except Exception as e:
        txt = f"# pip freeze failed: {e}\n"
    with open(outpath, "w", encoding="utf-8") as f:
        f.write(txt)


def write_environment_json(outpath: str) -> Dict[str, object]:
    ts = datetime.now(timezone.utc).isoformat()
    info: Dict[str, object] = {
        "timestamp_utc": ts,
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "cpu": {
            "os_cpu_count": os.cpu_count(),
        },
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "statsmodels": sm.__version__,
            "scipy": stats.__version__ if hasattr(stats, "__version__") else None,
            "matplotlib": matplotlib.__version__,
            "patsy": patsy.__version__,
        },
    }
    # scipy's version isn't in scipy.stats; patch from scipy import __version__
    try:
        import scipy

        info["packages"]["scipy"] = scipy.__version__
    except Exception:
        pass
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    return info


def write_run_manifest(outpath: str, manifest: Dict[str, object]) -> None:
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


# -------------------------
# Main
# -------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="outputs")
    ap.add_argument(
        "--mode",
        type=str,
        default="paper_regimes",
        choices=["paper_regimes", "violation_sweep"],
        help="Which simulation suite to run: the paper's 4 regimes or an A2 sensitivity sweep.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_trials", type=int, default=20000)
    ap.add_argument("--n_workers", type=int, default=600)
    ap.add_argument("--n_candidates", type=int, default=15)
    ap.add_argument("--test_frac_workers", type=float, default=0.2)
    ap.add_argument("--n_rep", type=int, default=20)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if str(args.mode).strip().lower() == "violation_sweep":
        run_violation_sweep(args)

        # Reproducibility bundle
        pip_freeze_path = os.path.join(args.outdir, "pip_freeze.txt")
        env_path = os.path.join(args.outdir, "environment.json")
        write_pip_freeze(pip_freeze_path)
        env_info = write_environment_json(env_path)
        manifest_path = os.path.join(args.outdir, "run_manifest.json")
        write_run_manifest(
            manifest_path,
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "script": os.path.basename(__file__),
                "mode": "violation_sweep",
                "args": vars(args),
                "environment": {
                    "python_version": env_info.get("python", {}).get("version"),
                    "platform": env_info.get("platform"),
                },
            },
        )
        print(f"Wrote outputs to: {args.outdir}")
        return

    # Four regimes (all satisfy A5; regime 2-3 introduce δ shifts; regime 3 violates A2 via alpha_c).
    # Added regime4 to isolate a key confound in outcome-level A2_outcome diagnostics:
    #   - In regimes 1–3, length affects BOTH (i) semantics (utterance specificity) and (ii) cost.
    #   - In regime4, length affects ONLY cost (sem_len_coef = 0), so any A2_outcome_len-driven rejections
    #     should weaken if they were driven primarily by semantic-length coupling.
    regimes = [
        (
            "regime1_uniform_prior_alpha_equal",
            {"close": 0.0, "far": 0.0, "split": 0.0},
            {"close": 2.0, "far": 2.0, "split": 2.0},
            1.4,  # sem_len_coef (default: length sharpens semantics)
        ),
        (
            "regime2_prior_shift_alpha_equal",
            {"close": 0.0, "far": 0.9, "split": 0.2},
            {"close": 2.0, "far": 2.0, "split": 2.0},
            1.4,
        ),
        (
            "regime3_prior_shift_alpha_hetero",
            {"close": 0.0, "far": 0.9, "split": 0.2},
            {"close": 1.2, "far": 0.8, "split": 2.6},
            1.4,
        ),
        (
            "regime4_uniform_prior_alpha_equal_len_cost_only",
            {"close": 0.0, "far": 0.0, "split": 0.0},
            {"close": 2.0, "far": 2.0, "split": 2.0},
            0.0,  # sem_len_coef = 0 ⇒ length does NOT affect semantics (pure cost only)
        ),
    ]

    all_summaries = []
    per_regime_outputs = []
    all_confusions: List[pd.DataFrame] = []

    # Run and plot
    for name, prior_bias, alpha_by_cond, sem_len_coef in regimes:
        cfg = SimConfig(
            seed=args.seed,
            n_trials=args.n_trials,
            n_workers=args.n_workers,
            n_candidates=args.n_candidates,
            test_frac_workers=args.test_frac_workers,
            prior_logit_bias_by_cond=prior_bias,
            alpha_by_cond=alpha_by_cond,
            sem_len_coef=float(sem_len_coef),
        )

        # One representative run for plots
        one = run_once(cfg)
        df_all = one["df"]

        # Repetition for rejection-rate figure
        df_rep = run_many(cfg, n_rep=args.n_rep)
        summ = summarize_rejections(df_rep)
        summ["regime"] = name
        all_summaries.append(summ)

        # Save tables
        summ_path = os.path.join(args.outdir, f"{name}_reject_summary.csv")
        df_rep_path = os.path.join(args.outdir, f"{name}_reject_raw.csv")
        summ.to_csv(summ_path, index=False)
        df_rep.to_csv(df_rep_path, index=False)

        # ------------------------------------------------------------
        # Confusion-matrix style summaries for A2_outcome diagnostics
        #   - regimes with heterogeneous alpha_by_cond are treated as
        #     "positive" (A2_theoretical violated; we want rejection)
        #   - regimes with homogeneous alpha_by_cond are treated as
        #     "negative" (A2 holds; rejection is a false rejection)
        # ------------------------------------------------------------
        truth_pos = len(set(alpha_by_cond.values())) > 1
        conf_blocks = ["A2_outcome_disc", "A2_outcome_clarity", "A2_outcome_len", "A2_outcome_full"]
        conf_rows = []
        for sp in ["train", "test"]:
            for blk in conf_blocks:
                conf_rows.append(
                    confusion_matrix_from_rejections(
                        df_rep,
                        truth_positive=truth_pos,
                        block=blk,
                        split=sp,
                    )
                )
        conf_df = pd.concat(conf_rows, ignore_index=True)
        conf_df.insert(0, "regime", name)
        conf_df_path = os.path.join(args.outdir, f"{name}_a2_confusion_matrix.csv")
        conf_df.to_csv(conf_df_path, index=False)
        all_confusions.append(conf_df)

        # Figures
        fig1 = os.path.join(args.outdir, f"{name}_condition_success.png")
        fig2 = os.path.join(args.outdir, f"{name}_rejection_rates.png")
        fig3 = os.path.join(args.outdir, f"{name}_wald_stats_example_test.png")

        tab_cond = plot_condition_success(df_all, fig1)
        # Include A2_outcome_len explicitly (this is the block most directly affected by the
        # semantic-length coupling toggled by `sem_len_coef`).
        tab_rej = plot_rejection_rates(df_rep, fig2, blocks=["A2_outcome_len", "A2_outcome_full", "A5_outcome_base", "A5_outcome|A2_outcome_full"])
        tab_wald = plot_wald_stats_example(one["tests_test"], fig3,
                                           blocks=["A2_outcome_disc", "A2_outcome_clarity", "A2_outcome_len", "A2_outcome_full", "A5_outcome_base", "A5_outcome|A2_outcome_full"])

        tab_cond.to_csv(os.path.join(args.outdir, f"{name}_condition_success_table.csv"), index=False)
        tab_rej.to_csv(os.path.join(args.outdir, f"{name}_rejection_rates_table.csv"), index=False)
        tab_wald.to_csv(os.path.join(args.outdir, f"{name}_wald_stats_example_test_table.csv"), index=False)

        per_regime_outputs.append({
            "name": name,
            "config": asdict(cfg),
            "files": {
                "condition_success_fig": os.path.basename(fig1),
                "rejection_rates_fig": os.path.basename(fig2),
                "wald_stats_example_test_fig": os.path.basename(fig3),
                "reject_summary_csv": os.path.basename(summ_path),
                "reject_raw_csv": os.path.basename(df_rep_path),
                "a2_confusion_matrix_csv": os.path.basename(conf_df_path),
            },
            "one_run": {
                "n_total": int(len(df_all)),
                "n_train": int(len(one["train"])),
                "n_test": int(len(one["test"])),
                "workers_total": int(df_all["worker"].nunique()),
                "workers_train": int(one["train"]["worker"].nunique()),
                "workers_test": int(one["test"]["worker"].nunique()),
            },
        })

    # Combined summary
    combined = pd.concat(all_summaries, ignore_index=True)
    combined_path = os.path.join(args.outdir, "all_regimes_reject_summary.csv")
    combined.to_csv(combined_path, index=False)

    # Combined confusion-matrix summaries
    if all_confusions:
        conf_all = pd.concat(all_confusions, ignore_index=True)
        conf_all_path = os.path.join(args.outdir, "all_regimes_a2_confusion_matrix.csv")
        conf_all.to_csv(conf_all_path, index=False)

        # Aggregate across regimes into "positive" vs "negative" truth classes.
        agg_rows = []
        for sp in ["train", "test"]:
            for blk in ["A2_outcome_disc", "A2_outcome_clarity", "A2_outcome_len", "A2_outcome_full"]:
                sub = conf_all[(conf_all["split"] == sp) & (conf_all["block"] == blk)]
                if sub.empty:
                    continue
                pos = sub[sub["truth_positive"] == True]
                neg = sub[sub["truth_positive"] == False]
                TP = int(pos["TP"].sum())
                FN = int(pos["FN"].sum())
                FP = int(neg["FP"].sum())
                TN = int(neg["TN"].sum())
                n_pos = int(pos["n"].sum())
                n_neg = int(neg["n"].sum())
                # Rates conditional on truth class
                power = TP / n_pos if n_pos else np.nan
                false_rej = FP / n_neg if n_neg else np.nan
                agg_rows.append(
                    {
                        "split": sp,
                        "block": blk,
                        "n_pos": n_pos,
                        "n_neg": n_neg,
                        "TP": TP,
                        "FN": FN,
                        "FP": FP,
                        "TN": TN,
                        "power_reject_if_violation": power,
                        "false_rejection_rate_if_holds": false_rej,
                    }
                )
        agg_df = pd.DataFrame(agg_rows)
        agg_path = os.path.join(args.outdir, "a2_confusion_aggregate.csv")
        agg_df.to_csv(agg_path, index=False)

    # Reproducibility bundle
    pip_freeze_path = os.path.join(args.outdir, "pip_freeze.txt")
    env_path = os.path.join(args.outdir, "environment.json")
    manifest_path = os.path.join(args.outdir, "run_manifest.json")
    write_pip_freeze(pip_freeze_path)
    env_info = write_environment_json(env_path)

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script": os.path.basename(__file__),
        "args": vars(args),
        "environment": {
            "python_version": env_info.get("python", {}).get("version"),
            "platform": env_info.get("platform"),
        },
        "regimes": per_regime_outputs,
        "combined_summary": os.path.basename(combined_path),
        "reproducibility_files": {
            "pip_freeze": os.path.basename(pip_freeze_path),
            "environment": os.path.basename(env_path),
            "run_manifest": os.path.basename(manifest_path),
        },
        "notes": {
            "figures": {
                "grayscale": True,
                "dpi": 1200,
                "titles": "none",
                "error_bars": "Wilson 95% intervals where applicable",
            }
        }
    }
    write_run_manifest(manifest_path, manifest)

    print(f"Wrote outputs to: {args.outdir}")
    print(f"- {os.path.basename(pip_freeze_path)}")
    print(f"- {os.path.basename(env_path)}")
    print(f"- {os.path.basename(manifest_path)}")


if __name__ == "__main__":
    main()
