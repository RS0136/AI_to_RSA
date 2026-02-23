from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .modeling import add_const, condition_dummies, interactions_with_disc, interactions_with_series
from .stats import (
    WaldBlockResult,
    cluster_covariance,
    parse_cluster_cols,
    lr_test,
    fit_binom_glm,
    tidy_coef_table,
    wald_joint,
    wild_cluster_bootstrap_wald_pvalue,
)


@dataclass(frozen=True)
class AssumptionDiagnostics:
    """Results for the A2_outcome–A5_outcome diagnostic suite on one split and one disc variant.
    """

    split: str
    disc_variant: str
    n_obs: int
    n_clusters: int

    # A2 (interaction block)
    a2_block: WaldBlockResult
    a2_lr_chi2: float
    a2_lr_df: int
    a2_lr_p_value: float

    # A2 (clarity interaction block)
    a2_clarity_block: WaldBlockResult
    a2_clarity_lr_chi2: float
    a2_clarity_lr_df: int
    a2_clarity_lr_p_value: float

    # A2 (length interaction block)
    a2_len_block: WaldBlockResult
    a2_len_lr_chi2: float
    a2_len_lr_df: int
    a2_len_lr_p_value: float

    # A2 (full slope interaction block: disc×condition + clarity×condition + len×condition)
    a2_full_block: WaldBlockResult
    a2_full_lr_chi2: float
    a2_full_lr_df: int
    a2_full_lr_p_value: float

    # A3/A4 (cluster-robust t)
    a3_t_clarity: float
    a4_t_disc: float

    # A5 (condition-intercept block)
    a5_block: WaldBlockResult
    a5_lr_chi2: float
    a5_lr_df: int
    a5_lr_p_value: float

    # A5_outcome|A2_outcome_full (conditional A5): condition intercept block tested within the A2_outcome_full model
    # (i.e., after allowing all slope×condition interactions).
    a5_conditional_block: WaldBlockResult
    a5_conditional_lr_chi2: float
    a5_conditional_lr_df: int
    a5_conditional_lr_p_value: float
def run_assumption_suite(
    df: pd.DataFrame,
    *,
    split_label: str,
    disc_variant: str,
    disc_z_col: str,
    condition_categories: Sequence[str],
    cluster_cols: Union[str, Sequence[str]] = "worker",
    link: str = "logit",
    wald_bootstrap: str = "none",
    wald_n_boot: int = 999,
    wald_bootstrap_seed: int = 0,
    wald_bootstrap_weight: str = "rademacher",
) -> Tuple[AssumptionDiagnostics, pd.DataFrame, pd.DataFrame]:
    """Compute the A2_outcome–A5_outcome diagnostic suite.

    Parameters
    ----------
    df:
        DataFrame already containing standardized columns (z_clarity, z_len_chars,
        and a standardized disc column `disc_z_col`).
    disc_z_col:
        Standardized disc column (used for GLMs and interactions).

    Returns
    -------
    diag:
        A structured result object.
    coef_a34:
        Cluster-robust coefficient table for the A3/A4 model.
    coef_a5_full:
        Cluster-robust coefficient table for the A5 full model.
    """

    wboot = str(wald_bootstrap).strip().lower()
    do_wild = wboot in {"wild", "wild_cluster", "wild-cluster"}

    y = df["y"].to_numpy(dtype=float)
    n_obs = int(df.shape[0])

    # Dummy coding (stable columns across splits)
    dum = condition_dummies(df, categories=condition_categories)
    controls = df[["z_clarity", "z_len_chars"]]

    # --- A2: interaction block Wald test (disc × condition) ---
    ints = interactions_with_disc(dum, df[disc_z_col], disc_z_col)
    X_a2_full = add_const(pd.concat([df[[disc_z_col]], dum, controls, ints], axis=1))
    res_a2_full = fit_binom_glm(y, X_a2_full, link=link)

    # Original archive: also report working-i.i.d. LR chi^2 for the interaction block
    X_a2_restr = add_const(pd.concat([df[[disc_z_col]], dum, controls], axis=1))
    res_a2_restr = fit_binom_glm(y, X_a2_restr, link=link)
    a2_lr, a2_df, a2_p = lr_test(res_a2_full, res_a2_restr, df_diff=(X_a2_full.shape[1] - X_a2_restr.shape[1]))
    # Multiway clustering: pass a 1D Series (one-way) or a multi-column frame.
    _cl_cols = parse_cluster_cols(cluster_cols)
    _cl_cols = [c for c in _cl_cols if c in df.columns]
    if not _cl_cols:
        _cl_cols = ["worker"]
    clusters_obj = df[_cl_cols] if len(_cl_cols) > 1 else df[_cl_cols[0]]

    cov_a2, G_a2 = cluster_covariance(res_a2_full, clusters_obj)
    a2_block = wald_joint(
        res_a2_full,
        cov_a2,
        clusters_n=G_a2,
        terms=list(ints.columns),
        block="A2_outcome_interaction_block",
        use_f=True,
    )

    if do_wild:
        p_wild = wild_cluster_bootstrap_wald_pvalue(
            res_a2_full,
            res_a2_restr,
            cov_full=cov_a2,
            clusters=clusters_obj,
            terms=list(ints.columns),
            n_boot=int(wald_n_boot),
            seed=int(wald_bootstrap_seed),
            weight=str(wald_bootstrap_weight),
        )
        a2_block = WaldBlockResult(**{**a2_block.__dict__, "p_value_wild": p_wild, "n_boot": int(wald_n_boot), "wild_weight": str(wald_bootstrap_weight), "wild_seed": int(wald_bootstrap_seed)})

    # --- A2: clarity × condition interaction block ---
    ints_clarity = interactions_with_series(dum, df["z_clarity"], "z_clarity")
    X_a2_clarity_full = add_const(pd.concat([df[[disc_z_col]], dum, controls, ints_clarity], axis=1))
    res_a2_clarity_full = fit_binom_glm(y, X_a2_clarity_full, link=link)

    a2_cl_lr, a2_cl_df, a2_cl_p = lr_test(
        res_a2_clarity_full,
        res_a2_restr,
        df_diff=(X_a2_clarity_full.shape[1] - X_a2_restr.shape[1]),
    )
    cov_a2_cl, G_a2_cl = cluster_covariance(res_a2_clarity_full, clusters_obj)
    a2_clarity_block = wald_joint(
        res_a2_clarity_full,
        cov_a2_cl,
        clusters_n=G_a2_cl,
        terms=list(ints_clarity.columns),
        block="A2_outcome_clarity_interaction_block",
        use_f=True,
    )

    if do_wild:
        p_wild = wild_cluster_bootstrap_wald_pvalue(
            res_a2_clarity_full,
            res_a2_restr,
            cov_full=cov_a2_cl,
            clusters=clusters_obj,
            terms=list(ints_clarity.columns),
            n_boot=int(wald_n_boot),
            seed=int(wald_bootstrap_seed),
            weight=str(wald_bootstrap_weight),
        )
        a2_clarity_block = WaldBlockResult(**{**a2_clarity_block.__dict__, "p_value_wild": p_wild, "n_boot": int(wald_n_boot), "wild_weight": str(wald_bootstrap_weight), "wild_seed": int(wald_bootstrap_seed)})

    # --- A2: length × condition interaction block ---
    ints_len = interactions_with_series(dum, df["z_len_chars"], "z_len_chars")
    X_a2_len_full = add_const(pd.concat([df[[disc_z_col]], dum, controls, ints_len], axis=1))
    res_a2_len_full = fit_binom_glm(y, X_a2_len_full, link=link)

    a2_len_lr, a2_len_df, a2_len_p = lr_test(
        res_a2_len_full,
        res_a2_restr,
        df_diff=(X_a2_len_full.shape[1] - X_a2_restr.shape[1]),
    )
    cov_a2_len, G_a2_len = cluster_covariance(res_a2_len_full, clusters_obj)
    a2_len_block = wald_joint(
        res_a2_len_full,
        cov_a2_len,
        clusters_n=G_a2_len,
        terms=list(ints_len.columns),
        block="A2_outcome_len_interaction_block",
        use_f=True,
    )

    if do_wild:
        p_wild = wild_cluster_bootstrap_wald_pvalue(
            res_a2_len_full,
            res_a2_restr,
            cov_full=cov_a2_len,
            clusters=clusters_obj,
            terms=list(ints_len.columns),
            n_boot=int(wald_n_boot),
            seed=int(wald_bootstrap_seed),
            weight=str(wald_bootstrap_weight),
        )
        a2_len_block = WaldBlockResult(**{**a2_len_block.__dict__, "p_value_wild": p_wild, "n_boot": int(wald_n_boot), "wild_weight": str(wald_bootstrap_weight), "wild_seed": int(wald_bootstrap_seed)})

    # --- A2 (full slope interaction block): (disc, clarity, len) × condition ---
    # This corresponds to the "all-slope interactions" diagnostic requested in review,
    # extending Eq. (33) to allow condition-specific slopes for *all* continuous predictors.
    ints_full = pd.concat([ints, ints_clarity, ints_len], axis=1)

    X_a2_full_slopes = add_const(pd.concat([df[[disc_z_col]], dum, controls, ints_full], axis=1))
    res_a2_full_slopes = fit_binom_glm(y, X_a2_full_slopes, link=link)

    # Working-i.i.d. LR chi^2 for adding the full interaction block (legacy-style diagnostic).
    a2_full_lr, a2_full_df, a2_full_p = lr_test(
        res_a2_full_slopes,
        res_a2_restr,
        df_diff=(X_a2_full_slopes.shape[1] - X_a2_restr.shape[1]),
    )

    cov_a2_full, G_a2_full = cluster_covariance(res_a2_full_slopes, clusters_obj)
    a2_full_block = wald_joint(
        res_a2_full_slopes,
        cov_a2_full,
        clusters_n=G_a2_full,
        terms=list(ints_full.columns),
        block="A2_outcome_full_slope_interaction_block",
        use_f=True,
    )

    if do_wild:
        p_wild = wild_cluster_bootstrap_wald_pvalue(
            res_a2_full_slopes,
            res_a2_restr,
            cov_full=cov_a2_full,
            clusters=clusters_obj,
            terms=list(ints_full.columns),
            n_boot=int(wald_n_boot),
            seed=int(wald_bootstrap_seed),
            weight=str(wald_bootstrap_weight),
        )
        a2_full_block = WaldBlockResult(**{**a2_full_block.__dict__, "p_value_wild": p_wild, "n_boot": int(wald_n_boot), "wild_weight": str(wald_bootstrap_weight), "wild_seed": int(wald_bootstrap_seed)})


    # --- A5_outcome|A2_outcome_full (conditional A5): test condition intercept block *within* the A2_outcome_full model ---
    # This answers: do condition baselines remain necessary after allowing all slope×condition interactions?
    a5_conditional_block = wald_joint(
        res_a2_full_slopes,
        cov_a2_full,
        clusters_n=G_a2_full,
        terms=list(dum.columns),
        block="A5_outcome_condition_intercepts_given_A2_outcome_full",
        use_f=True,
    )

    # Working-i.i.d. LR chi^2 for the same conditional A5 block (legacy-style diagnostic).
    # Nested comparison: full A2_outcome_full_slopes model vs. the same model with condition intercepts removed.
    X_a5_cond_restr = add_const(pd.concat([df[[disc_z_col]], controls, ints_full], axis=1))
    res_a5_cond_restr = fit_binom_glm(y, X_a5_cond_restr, link=link)
    a5c_lr, a5c_df, a5c_p = lr_test(
        res_a2_full_slopes,
        res_a5_cond_restr,
        df_diff=(X_a2_full_slopes.shape[1] - X_a5_cond_restr.shape[1]),
    )

    if do_wild:
        p_wild = wild_cluster_bootstrap_wald_pvalue(
            res_a2_full_slopes,
            res_a5_cond_restr,
            cov_full=cov_a2_full,
            clusters=clusters_obj,
            terms=list(dum.columns),
            n_boot=int(wald_n_boot),
            seed=int(wald_bootstrap_seed),
            weight=str(wald_bootstrap_weight),
        )
        a5_conditional_block = WaldBlockResult(**{**a5_conditional_block.__dict__, "p_value_wild": p_wild, "n_boot": int(wald_n_boot), "wild_weight": str(wald_bootstrap_weight), "wild_seed": int(wald_bootstrap_seed)})

    # --- A3/A4: disc + clarity + length (cluster-robust t) ---
    X_a34 = add_const(df[[disc_z_col, "z_clarity", "z_len_chars"]])
    res_a34 = fit_binom_glm(y, X_a34, link=link)
    cov_a34, G_a34 = cluster_covariance(res_a34, clusters_obj)
    coef_a34 = tidy_coef_table(res_a34, cov_a34, df_t=max(G_a34 - 1, 1))

    def _t(term: str) -> float:
        row = coef_a34.loc[coef_a34["term"] == term]
        if row.empty:
            return float("nan")
        return float(row["t"].values[0])

    a3_t = _t("z_clarity")
    a4_t = _t(disc_z_col)

    # --- A5: condition intercept block Wald test ---
    X_a5_full = add_const(pd.concat([df[[disc_z_col]], controls, dum], axis=1))
    res_a5_full = fit_binom_glm(y, X_a5_full, link=link)

    # Original archive: working-i.i.d. LR chi^2 for adding condition intercepts
    X_a5_restr = add_const(pd.concat([df[[disc_z_col]], controls], axis=1))
    res_a5_restr = fit_binom_glm(y, X_a5_restr, link=link)
    a5_lr, a5_df, a5_p = lr_test(res_a5_full, res_a5_restr, df_diff=(X_a5_full.shape[1] - X_a5_restr.shape[1]))
    cov_a5, G_a5 = cluster_covariance(res_a5_full, clusters_obj)
    a5_block = wald_joint(
        res_a5_full,
        cov_a5,
        clusters_n=G_a5,
        terms=list(dum.columns),
        block="A5_outcome_condition_intercepts",
        use_f=True,
    )

    if do_wild:
        p_wild = wild_cluster_bootstrap_wald_pvalue(
            res_a5_full,
            res_a5_restr,
            cov_full=cov_a5,
            clusters=clusters_obj,
            terms=list(dum.columns),
            n_boot=int(wald_n_boot),
            seed=int(wald_bootstrap_seed),
            weight=str(wald_bootstrap_weight),
        )
        a5_block = WaldBlockResult(**{**a5_block.__dict__, "p_value_wild": p_wild, "n_boot": int(wald_n_boot), "wild_weight": str(wald_bootstrap_weight), "wild_seed": int(wald_bootstrap_seed)})
    coef_a5 = tidy_coef_table(res_a5_full, cov_a5, df_t=max(G_a5 - 1, 1))

    diag = AssumptionDiagnostics(
        split=split_label,
        disc_variant=disc_variant,
        n_obs=n_obs,
        n_clusters=max(G_a5, G_a2, G_a2_cl, G_a2_len, G_a2_full, G_a34),
        a2_block=a2_block,
        a2_lr_chi2=a2_lr,
        a2_lr_df=a2_df,
        a2_lr_p_value=a2_p,
        a2_clarity_block=a2_clarity_block,
        a2_clarity_lr_chi2=a2_cl_lr,
        a2_clarity_lr_df=a2_cl_df,
        a2_clarity_lr_p_value=a2_cl_p,
        a2_len_block=a2_len_block,
        a2_len_lr_chi2=a2_len_lr,
        a2_len_lr_df=a2_len_df,
        a2_len_lr_p_value=a2_len_p,
        a2_full_block=a2_full_block,
        a2_full_lr_chi2=a2_full_lr,
        a2_full_lr_df=a2_full_df,
        a2_full_lr_p_value=a2_full_p,
        a3_t_clarity=a3_t,
        a4_t_disc=a4_t,
        a5_block=a5_block,
        a5_lr_chi2=a5_lr,
        a5_lr_df=a5_df,
        a5_lr_p_value=a5_p,
        a5_conditional_block=a5_conditional_block,
        a5_conditional_lr_chi2=a5c_lr,
        a5_conditional_lr_df=a5c_df,
        a5_conditional_lr_p_value=a5c_p,
    )

    return diag, coef_a34, coef_a5


def diagnostics_to_rows(diag: AssumptionDiagnostics) -> List[Dict]:
    """Flatten diagnostics for CSV-friendly output."""

    base = {
        "split": diag.split,
        "disc_variant": diag.disc_variant,
        "n_obs": diag.n_obs,
        "n_clusters": diag.n_clusters,
    }

    rows: List[Dict] = []

    # Original archive LR diagnostics (working i.i.d. likelihood)
    rows.append(
        {
            **base,
            "test": "A2_outcome_interaction_block_lr_chi2",
            "stat": diag.a2_lr_chi2,
            "df": diag.a2_lr_df,
            "p_value": diag.a2_lr_p_value,
            "note": "LR chi^2 for adding disc×condition interactions (i.i.d.).",
        }
    )

    rows.append(
        {
            **base,
            "test": "A2_outcome_clarity_interaction_block_lr_chi2",
            "stat": diag.a2_clarity_lr_chi2,
            "df": diag.a2_clarity_lr_df,
            "p_value": diag.a2_clarity_lr_p_value,
            "note": "LR chi^2 for adding clarity×condition interactions (i.i.d.).",
        }
    )

    rows.append(
        {
            **base,
            "test": "A2_outcome_len_interaction_block_lr_chi2",
            "stat": diag.a2_len_lr_chi2,
            "df": diag.a2_len_lr_df,
            "p_value": diag.a2_len_lr_p_value,
            "note": "LR chi^2 for adding len_chars×condition interactions (i.i.d.).",
        }
    )

    rows.append(
        {
            **base,
            "test": "A2_outcome_full_slope_interaction_block_lr_chi2",
            "stat": diag.a2_full_lr_chi2,
            "df": diag.a2_full_lr_df,
            "p_value": diag.a2_full_lr_p_value,
            "note": "LR chi^2 for adding (disc, clarity, len)×condition interactions (i.i.d.).",
        }
    )
    rows.append(
        {
            **base,
            "test": "A5_outcome_condition_intercepts_lr_chi2",
            "stat": diag.a5_lr_chi2,
            "df": diag.a5_lr_df,
            "p_value": diag.a5_lr_p_value,
            "note": "LR chi^2 for adding condition intercepts (i.i.d.).",
        }
    )

    rows.append(
        {
            **base,
            "test": "A5_outcome_condition_intercepts_given_A2_outcome_full_lr_chi2",
            "stat": diag.a5_conditional_lr_chi2,
            "df": diag.a5_conditional_lr_df,
            "p_value": diag.a5_conditional_lr_p_value,
            "note": "LR chi^2 for adding condition intercepts within the full-slope-interaction model (i.i.d.).",
        }
    )

    for blk in [diag.a2_block, diag.a2_clarity_block, diag.a2_len_block, diag.a2_full_block, diag.a5_block, diag.a5_conditional_block]:
        rows.append(
            {
                **base,
                "test": f"{blk.block}_wald_chi2",
                "stat": blk.stat_chi2,
                "df": blk.df,
                "p_value": blk.p_value_chi2,
                "note": "Cluster-robust Wald (chi-square).",
            }
        )
        rows.append(
            {
                **base,
                "test": f"{blk.block}_wald_f",
                "stat": blk.f_stat,
                "df": f"{blk.df_num},{blk.df_denom}",
                "p_value": blk.p_value_f,
                "note": "Cluster-robust Wald (F approx).",
            }
        )

        if blk.p_value_wild is not None and np.isfinite(float(blk.p_value_wild)):
            rows.append(
                {
                    **base,
                    "test": f"{blk.block}_wald_wild_cluster_bootstrap",
                    "stat": blk.stat_chi2,
                    "df": blk.df,
                    "p_value": float(blk.p_value_wild),
                    "note": f"Wild cluster bootstrap p-value (multiplier; weight={blk.wild_weight}, n_boot={blk.n_boot}).",
                }
            )

    rows.append({**base, "test": "A3_clarity_t", "stat": diag.a3_t_clarity, "df": diag.n_clusters - 1, "p_value": np.nan, "note": "Cluster-robust t for z_clarity."})
    rows.append({**base, "test": "A4_disc_t", "stat": diag.a4_t_disc, "df": diag.n_clusters - 1, "p_value": np.nan, "note": "Cluster-robust t for disc."})

    return rows
