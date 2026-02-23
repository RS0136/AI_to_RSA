from __future__ import annotations

import os
from dataclasses import asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import TRI_GROUPS, DiscDefinition, disc_definitions
from .data import load_cic_csv
from .diagnostics import run_assumption_suite
from .io_utils import ensure_dir, save_df, save_json, save_text, collect_environment_info, pip_freeze
from .modeling import add_const, condition_dummies, interactions_with_series
from .plotting import bar_with_error, bar_with_error_allow_na, scatter
from .legacy import (
    feature_histograms,
    plot_condition_intercepts_from_a5,
    split_half_stability,
    vif_table_and_plot,
)
from .shapley import compute_shapley_threeway, summarize_bootstrap
from .split import make_split
from .standardize import fit_standardizer
from .stats import bic_llf, brier_score, log_likelihood_from_probs, wilson_ci
from .sensitivity import (
    add_condition_interactions,
    calibration_by_group,
    fit_cluster_robust_glm,
    fit_mixed_effect_logit_random_intercepts,
    plot_calibration_curves,
)
from .scale_adjusted_logit import run_scale_adjusted_logit_suite


def _safe_mean(x: pd.Series) -> float:
    v = x.to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else float("nan")


def _build_disc_column(df: pd.DataFrame, disc_def: DiscDefinition) -> pd.Series:
    return disc_def.func(df["t1"], df["t2"], df.get("d12", pd.Series(np.nan, index=df.index)))


def _filter_variant(df: pd.DataFrame, disc: pd.Series) -> pd.DataFrame:
    out = df.copy()
    out["disc"] = pd.to_numeric(disc, errors="coerce")
    mask = np.isfinite(out["disc"].to_numpy(dtype=float))
    return out.loc[mask].reset_index(drop=True)


def _degeneracy_status(
    df: pd.DataFrame,
    *,
    disc_col: str = "z_disc",
    clarity_col: str = "z_clarity",
    corr_abs_threshold: float = 0.999999,
) -> tuple[str, str, Dict[str, float]]:
    """Detect degenerate (rank-deficient) variants where disc is (near-)collinear with clarity.

    We primarily guard against disc variants that are effectively a (signed) re-expression of clarity.
    In such cases, models that include both disc and clarity (and their condition interactions)
    become rank-deficient, and downstream Wald tests/attributions are not meaningful.

    Returns
    -------
    status:
        "OK" or "NA"
    reason:
        Short machine-readable string
    meta:
        Diagnostic scalars (e.g., correlation, rank, condition number)
    """

    x = pd.to_numeric(df[disc_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[clarity_col], errors="coerce").to_numpy(dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    n_finite = int(mask.sum())
    if n_finite < 3:
        return "NA", "too_few_finite_rows", {"n_finite": float(n_finite)}

    mat = np.column_stack([x[mask], y[mask]])

    # Rank deficiency catches exact collinearity (e.g., corr == ±1).
    rank = int(np.linalg.matrix_rank(mat))

    # Correlation (as a secondary guard; very high |corr| can still cause unstable inference).
    sx = float(np.std(mat[:, 0]))
    sy = float(np.std(mat[:, 1]))
    corr = float(np.corrcoef(mat[:, 0], mat[:, 1])[0, 1]) if (sx > 0 and sy > 0) else float("nan")

    # Condition number (helps interpret near-collinearity).
    try:
        cond = float(np.linalg.cond(mat))
    except Exception:
        cond = float("nan")

    if rank < 2:
        return "NA", "disc_clarity_rank_deficient", {"corr": corr, "rank": float(rank), "cond": cond}

    if np.isfinite(corr) and abs(corr) >= float(corr_abs_threshold):
        return "NA", "disc_clarity_corr_near_1", {"corr": corr, "rank": float(rank), "cond": cond}

    return "OK", "", {"corr": corr, "rank": float(rank), "cond": cond}


def _interaction_terms(dum_cols: Sequence[str], var_name: str) -> List[str]:
    """Helper to mirror `modeling.interactions_with_series` column naming."""
    return [f"{var_name}_x_{c.replace('cond_', '')}" for c in list(dum_cols)]

def condition_overview(
    df: pd.DataFrame,
    out_tables: str,
    out_figures: str,
    *,
    split_label: str,
    dpi: int,
) -> None:
    """Condition-level accuracy table and plot with Wilson CIs."""

    # Condition-level means of geometry/length fields.
    #
    # NOTE (rev2): the manuscript's *primary* disc proxy is now the nearest-competitor
    # difficulty index (more directly interpretable). For transparency we also report
    # the prior manuscript's Eq.(9) coding as a legacy column.
    disc_primary = -pd.concat([df["t1"].abs(), df["t2"].abs()], axis=1).min(axis=1)
    disc_legacy = df["d12"].where(df["d12"].notna(), df["t1"] - df["t2"])

    g = (
        df.assign(disc_primary=disc_primary, disc_legacy_eq9=disc_legacy)
        .groupby("condition", as_index=False)
        .agg(
            n=("y", "size"),
            acc=("y", "mean"),
            disc_primary=("disc_primary", "mean"),
            disc_legacy_eq9=("disc_legacy_eq9", "mean"),
            clarity=("clarity", "mean"),
            len_tokens=("len_tokens", "mean"),
            len_chars=("len_chars", "mean"),
        )
        .sort_values("condition")
        .reset_index(drop=True)
    )

    ci_lo: List[float] = []
    ci_hi: List[float] = []
    for _, row in g.iterrows():
        n = int(row["n"])
        k = int(round(float(row["acc"]) * n))
        lo, hi = wilson_ci(k, n, alpha=0.05)
        ci_lo.append(max(float(row["acc"]) - lo, 0.0))
        ci_hi.append(max(hi - float(row["acc"]), 0.0))

    g["err_lo"] = ci_lo
    g["err_hi"] = ci_hi

    save_df(g, os.path.join(out_tables, f"condition_overview_{split_label}.csv"))

    yerr = np.vstack([np.array(ci_lo, dtype=float), np.array(ci_hi, dtype=float)])
    bar_with_error(
        labels=g["condition"].tolist(),
        values=g["acc"].to_numpy(dtype=float),
        yerr=yerr,
        xlabel="Condition",
        ylabel="Accuracy (proportion)",
        base_path=os.path.join(out_figures, f"condition_overview_accuracy_{split_label}"),
        rotation=15,
        dpi=dpi,
    )


def fit_and_plot_coefficients(
    df: pd.DataFrame,
    out_tables: str,
    out_figures: str,
    *,
    disc_variant: str,
    split_label: str,
    cluster_cols: str = "worker",
    dpi: int,
) -> None:
    """Plot disc/clarity/length coefficients with cluster-robust SE.
    """

    import statsmodels.api as sm

    y = df["y"].to_numpy(dtype=float)
    X = sm.add_constant(df[["z_disc", "z_clarity", "z_len_chars"]].astype(float), has_constant="add")
    res = sm.GLM(y, X, family=sm.families.Binomial()).fit()

    # Use the shared robust helper (with pseudo-inverse fallback for singular Hessians).
    from .stats import cluster_covariance, parse_cluster_cols

    ccols = [c for c in parse_cluster_cols(cluster_cols) if c in df.columns]
    if not ccols:
        ccols = ["worker"]
    clusters_obj = df[ccols] if len(ccols) > 1 else df[ccols[0]]

    cov, G = cluster_covariance(res, clusters_obj)

    coef = pd.DataFrame(
        {
            "term": list(res.params.index),
            "coef": np.asarray(res.params, dtype=float),
            "se": np.sqrt(np.clip(np.diag(cov), 0.0, np.inf)),
            "df_t": max(int(G) - 1, 1),
        }
    )

    save_df(
        coef,
        os.path.join(out_tables, f"glm_coefficients_cluster_robust_{split_label}_{disc_variant}.csv"),
    )

    # 95% t-interval using df = G-1 (common small-sample cluster proxy).
    from scipy.stats import t as student_t

    tcrit = float(student_t.ppf(0.975, df=max(int(G) - 1, 1)))
    terms = ["z_disc", "z_clarity", "z_len_chars"]
    vals = [float(coef.loc[coef["term"] == t, "coef"].values[0]) for t in terms]
    se = [float(coef.loc[coef["term"] == t, "se"].values[0]) for t in terms]
    yerr = np.vstack([tcrit * np.array(se, dtype=float), tcrit * np.array(se, dtype=float)])

    bar_with_error(
        labels=["disc", "clarity", "len_chars"],
        values=vals,
        yerr=yerr,
        xlabel="Term",
        ylabel="Coefficient (log-odds per 1 SD)",
        base_path=os.path.join(out_figures, f"glm_coefficients_{split_label}_{disc_variant}"),
        rotation=0,
        figsize=(5.0, 4.0),
        dpi=dpi,
    )

def stepwise_bic(
    df: pd.DataFrame,
    out_tables: str,
    out_figures: str,
    *,
    disc_variant: str,
    split_label: str,
    dpi: int,
) -> None:
    """Reproduce the original archive's stepwise BIC table/plot.
    """

    import statsmodels.api as sm

    y = df["y"].to_numpy(dtype=float)

    models: List[Tuple[str, List[str]]] = [
        ("disc", ["z_disc"]),
        ("disc+clarity", ["z_disc", "z_clarity"]),
        ("disc+clarity+len_chars", ["z_disc", "z_clarity", "z_len_chars"]),
    ]

    rows: List[Dict] = []
    bics: List[float] = []
    labels: List[str] = []

    for name, cols in models:
        if not set(cols).issubset(set(df.columns)):
            continue
        X = sm.add_constant(df[cols].astype(float), has_constant="add")
        res = sm.GLM(y, X, family=sm.families.Binomial()).fit()
        bic = float(bic_llf(res))
        rows.append(
            {
                "split": split_label,
                "disc_variant": disc_variant,
                "model": name,
                "n_obs": int(res.nobs),
                "n_params": int(len(res.params)),
                "llf": float(res.llf),
                "bic": bic,
            }
        )
        bics.append(bic)
        labels.append(name)

    if not rows:
        return

    tab = pd.DataFrame(rows)
    save_df(tab, os.path.join(out_tables, f"stepwise_bic_{split_label}_{disc_variant}.csv"))

    bar_with_error(
        labels=labels,
        values=bics,
        yerr=None,
        xlabel="Model",
        ylabel="BIC (LLF-based; unitless)",
        base_path=os.path.join(out_figures, f"stepwise_bic_{split_label}_{disc_variant}"),
        rotation=20,
        dpi=dpi,
        figsize=(6.2, 4.0),
    )


def assumption_summary_plots(
    diag,
    *,
    out_figures: str,
    dpi: int,
) -> None:
    """Minimal assumption-test plots used in the paper.
    """

    split_label = str(getattr(diag, "split", "")).strip().title() or "Split"

    # --- A2 quad (Wald chi2) ---
    vals_w_a2 = [
        float(diag.a2_block.stat_chi2),
        float(diag.a2_clarity_block.stat_chi2),
        float(diag.a2_len_block.stat_chi2),
        float(diag.a2_full_block.stat_chi2),
    ]
    labels_w_a2 = [
        "A2_outcome_disc (Wald chi2)",
        "A2_outcome_clarity (Wald chi2)",
        "A2_outcome_len (Wald chi2)",
        "A2_outcome_full (Wald chi2)",
    ]
    bar_with_error_allow_na(
        labels=labels_w_a2,
        values=vals_w_a2,
        yerr=None,
        xlabel=split_label,
        ylabel="Wald chi2 (unitless)",
        base_path=os.path.join(out_figures, f"assumption_suite_wald_a2quad_{diag.split}_{diag.disc_variant}"),
        rotation=25,
        dpi=dpi,
        figsize=(7.2, 4.0),
    )

    # --- A3/A4 only (cluster-robust t) ---
    vals_a34 = [
        float(diag.a3_t_clarity),
        float(diag.a4_t_disc),
    ]
    labels_a34 = [
        "A3 (t: clarity)",
        "A4 (t: disc)",
    ]
    bar_with_error_allow_na(
        labels=labels_a34,
        values=vals_a34,
        yerr=None,
        xlabel=split_label,
        ylabel="t statistic (unitless)",
        base_path=os.path.join(out_figures, f"assumption_suite_wald_a34only_{diag.split}_{diag.disc_variant}"),
        rotation=0,
        dpi=dpi,
        figsize=(4.8, 4.0),
    )

    # --- A5 base vs conditional on A2_outcome_full (Wald chi2) ---
    vals_a5pair = [
        float(diag.a5_block.stat_chi2),
        float(diag.a5_conditional_block.stat_chi2),
    ]
    labels_a5pair = [
        "A5_outcome_base (Wald chi2)",
        "A5_outcome|A2_outcome_full (Wald chi2)",
    ]
    bar_with_error_allow_na(
        labels=labels_a5pair,
        values=vals_a5pair,
        yerr=None,
        xlabel=split_label,
        ylabel="Wald chi2 (unitless)",
        base_path=os.path.join(out_figures, f"assumption_suite_wald_a5pair_{diag.split}_{diag.disc_variant}"),
        rotation=15,
        dpi=dpi,
        figsize=(6.2, 4.0),
    )


def assumption_summary_plots_lr(
    diag,
    *,
    out_figures: str,
    dpi: int,
) -> None:
    """LR (working-i.i.d.) versions of the minimal assumption-test plots.
    """

    split_label = str(getattr(diag, "split", "")).strip().title() or "Split"

    # --- A2 quad (LR chi2) ---
    vals_lr_a2 = [
        float(diag.a2_lr_chi2),
        float(diag.a2_clarity_lr_chi2),
        float(diag.a2_len_lr_chi2),
        float(diag.a2_full_lr_chi2),
    ]
    labels_lr_a2 = [
        "A2_outcome_disc (LR chi2)",
        "A2_outcome_clarity (LR chi2)",
        "A2_outcome_len (LR chi2)",
        "A2_outcome_full (LR chi2)",
    ]
    bar_with_error_allow_na(
        labels=labels_lr_a2,
        values=vals_lr_a2,
        yerr=None,
        xlabel=split_label,
        ylabel="LR chi2 (unitless; i.i.d.)",
        base_path=os.path.join(out_figures, f"assumption_suite_lr_a2quad_{diag.split}_{diag.disc_variant}"),
        rotation=25,
        dpi=dpi,
        figsize=(7.2, 4.0),
    )

    # --- A5 base vs conditional on A2_outcome_full (LR chi2) ---
    vals_lr_a5 = [
        float(diag.a5_lr_chi2),
        float(getattr(diag, "a5_conditional_lr_chi2", np.nan)),
    ]
    labels_lr_a5 = [
        "A5_outcome_base (LR chi2)",
        "A5_outcome|A2_outcome_full (LR chi2)",
    ]
    bar_with_error_allow_na(
        labels=labels_lr_a5,
        values=vals_lr_a5,
        yerr=None,
        xlabel=split_label,
        ylabel="LR chi2 (unitless; i.i.d.)",
        base_path=os.path.join(out_figures, f"assumption_suite_lr_a5pair_{diag.split}_{diag.disc_variant}"),
        rotation=15,
        dpi=dpi,
        figsize=(6.2, 4.0),
    )



def run_pipeline(
    *,
    input_csv: str,
    output_dir: str,
    seed: int = 0,
    test_size: float = 0.20,
    disc_variants: Optional[Sequence[str]] = None,
    shapley_bootstrap: bool = True,
    shapley_n_boot: int = 300,
    cluster_spec: str = "worker",
    shapley_bootstrap_method: str = "cluster",
    wald_bootstrap: str = "none",
    wald_n_boot: int = 999,
    wald_bootstrap_weight: str = "rademacher",
    dpi: int = 800,
) -> None:
    ensure_dir(output_dir)
    out_tables = ensure_dir(os.path.join(output_dir, "tables"))
    out_figures = ensure_dir(os.path.join(output_dir, "figures"))
    # Reproducibility record (system + library versions)
    save_json(collect_environment_info(), os.path.join(output_dir, "environment.json"))
    save_text(pip_freeze(), os.path.join(output_dir, "pip_freeze.txt"))

    data = load_cic_csv(input_csv)
    split = make_split(data.df, group_col="worker", test_size=test_size, seed=seed)

    # Save split manifest (worker IDs) for reproducibility.
    manifest = {
        "seed": seed,
        "test_size": test_size,
        "cluster_spec": str(cluster_spec),
        "wald_bootstrap": str(wald_bootstrap),
        "wald_n_boot": int(wald_n_boot),
        "wald_bootstrap_weight": str(wald_bootstrap_weight),
        "shapley_bootstrap_method": str(shapley_bootstrap_method),
        "n_train": int(split.train.shape[0]),
        "n_test": int(split.test.shape[0]),
        "n_train_workers": int(split.train["worker"].nunique()),
        "n_test_workers": int(split.test["worker"].nunique()),
    }
    save_json(manifest, os.path.join(output_dir, "run_manifest.json"))

    # Condition overview on both splits
    condition_overview(split.train, out_tables, out_figures, split_label="train", dpi=dpi)
    condition_overview(split.test, out_tables, out_figures, split_label="test", dpi=dpi)

    # Condition category set (fixed from TRAIN for stable dummy coding)
    cond_categories = sorted(split.train["condition"].astype(str).unique().tolist())

    defs = disc_definitions()
    if disc_variants is None:
        selected = list(defs.keys())
        # 'original' is now defined as the nearest-competitor proxy. We keep
        # the explicit 'nearest_competitor' key as a backwards-compatible alias,
        # but exclude it from the default 'all' list to avoid duplicate variants.
        if "original" in selected and "nearest_competitor" in selected:
            selected.remove("nearest_competitor")
    else:
        selected = []
        for k in disc_variants:
            if k not in defs:
                raise ValueError(f"Unknown disc variant: {k}. Available={list(defs.keys())}")
            selected.append(k)

    # Save disc definition catalog
    disc_catalog = [{"name": defs[k].name, "description": defs[k].description} for k in selected]
    save_df(pd.DataFrame(disc_catalog), os.path.join(out_tables, "disc_definitions.csv"))

    # Aggregate tables
    assumption_rows: List[Dict] = []
    assumption_lr_rows: List[Dict] = []
    shapley_rows: List[Dict] = []
    warnings_rows: List[Dict] = []

    # "Primary" disc variant used for the manuscript-aligned *main* outputs
    # and for the additional review-oriented sensitivity analyses (link choice,
    # alternative clustering, mixed-effects logistic, calibration, etc.).
    #
    # The default expects that "original" is present. If the user runs a
    # restricted subset, we fall back to a reasonable candidate.
    if "original" in selected:
        primary_variant = "original"
    elif "nearest_competitor" in selected:
        primary_variant = "nearest_competitor"
    else:
        primary_variant = selected[0] if selected else "original"

    primary_cache: Optional[Dict[str, pd.DataFrame]] = None

    for dv in selected:
        disc_def = defs[dv]

        # Build variant-specific datasets (filter rows where disc is computable)
        train_v = _filter_variant(split.train, _build_disc_column(split.train, disc_def))
        test_v = _filter_variant(split.test, _build_disc_column(split.test, disc_def))

        # Safety: avoid cryptic statsmodels errors if a variant removes all rows.
        if train_v.shape[0] == 0 or test_v.shape[0] == 0:
            status = "NA"
            status_reason = "no_rows_after_disc_filter"

            warnings_rows.append(
                {
                    "disc_variant": dv,
                    "issue": "no_rows_after_disc_filter",
                    "n_train": int(train_v.shape[0]),
                    "n_test": int(test_v.shape[0]),
                    "status": status,
                    "status_reason": status_reason,
                }
            )

            for df_v, split_label in [(train_v, "train"), (test_v, "test")]:
                n_clusters = int(df_v["worker"].nunique()) if df_v.shape[0] else 0
                n_obs_split = int(df_v.shape[0])

                assumption_rows.append(
                    {
                        "split": split_label,
                        "disc_variant": dv,
                        "status": status,
                        "status_reason": status_reason,
                        "n_obs": n_obs_split,
                        "n_clusters": n_clusters,
                        "A2_outcome_disc_wald_chi2": np.nan,
                        "A2_outcome_disc_df": np.nan,
                        "A2_outcome_disc_p": np.nan,
                        "A2_outcome_clarity_wald_chi2": np.nan,
                        "A2_outcome_clarity_df": np.nan,
                        "A2_outcome_clarity_p": np.nan,
                        "A2_outcome_len_wald_chi2": np.nan,
                        "A2_outcome_len_df": np.nan,
                        "A2_outcome_len_p": np.nan,
                        "A2_outcome_full_wald_chi2": np.nan,
                        "A2_outcome_full_df": np.nan,
                        "A2_outcome_full_p": np.nan,
                        "A3_clarity_t": np.nan,
                        "A4_disc_t": np.nan,
                        "A5_outcome_base_wald_chi2": np.nan,
                        "A5_outcome_base_df": np.nan,
                        "A5_outcome_base_p": np.nan,
                        "A5_outcome_given_A2_outcome_full_wald_chi2": np.nan,
                        "A5_outcome_given_A2_outcome_full_df": np.nan,
                        "A5_outcome_given_A2_outcome_full_p": np.nan,
                    }
                )


                assumption_lr_rows.append(
                    {
                        "split": split_label,
                        "disc_variant": dv,
                        "status": status,
                        "status_reason": status_reason,
                        "n_obs": n_obs_split,
                        "n_clusters": n_clusters,
                        "A2_outcome_disc_lr_chi2": np.nan,
                        "A2_outcome_disc_lr_df": np.nan,
                        "A2_outcome_disc_lr_p": np.nan,
                        "A2_outcome_clarity_lr_chi2": np.nan,
                        "A2_outcome_clarity_lr_df": np.nan,
                        "A2_outcome_clarity_lr_p": np.nan,
                        "A2_outcome_len_lr_chi2": np.nan,
                        "A2_outcome_len_lr_df": np.nan,
                        "A2_outcome_len_lr_p": np.nan,
                        "A2_outcome_full_lr_chi2": np.nan,
                        "A2_outcome_full_lr_df": np.nan,
                        "A2_outcome_full_lr_p": np.nan,
                        "A5_outcome_base_lr_chi2": np.nan,
                        "A5_outcome_base_lr_df": np.nan,
                        "A5_outcome_base_lr_p": np.nan,
                        "A5_outcome_given_A2_outcome_full_lr_chi2": np.nan,
                        "A5_outcome_given_A2_outcome_full_lr_df": np.nan,
                        "A5_outcome_given_A2_outcome_full_lr_p": np.nan,
                    }
                )

                split_title = str(split_label).strip().title()
                bar_with_error_allow_na(
                    labels=["A2_outcome_disc", "A2_outcome_clarity", "A2_outcome_len", "A2_outcome_full"],
                    values=[np.nan, np.nan, np.nan, np.nan],
                    yerr=None,
                    xlabel=split_title,
                    ylabel="Wald chi2 (unitless)",
                    base_path=os.path.join(out_figures, f"assumption_suite_wald_a2quad_{split_label}_{dv}"),
                    rotation=25,
                    dpi=dpi,
                    figsize=(7.2, 4.0),
                )
                bar_with_error_allow_na(
                    labels=["A3 (t: clarity)", "A4 (t: disc)"],
                    values=[np.nan, np.nan],
                    yerr=None,
                    xlabel=split_title,
                    ylabel="t statistic (unitless)",
                    base_path=os.path.join(out_figures, f"assumption_suite_wald_a34only_{split_label}_{dv}"),
                    rotation=0,
                    dpi=dpi,
                    figsize=(4.8, 4.0),
                )
                bar_with_error_allow_na(
                    labels=["A5_outcome_base (Wald chi2)", "A5_outcome|A2_outcome_full (Wald chi2)"],
                    values=[np.nan, np.nan],
                    yerr=None,
                    xlabel=split_title,
                    ylabel="Wald chi2 (unitless)",
                    base_path=os.path.join(out_figures, f"assumption_suite_wald_a5pair_{split_label}_{dv}"),
                    rotation=15,
                    dpi=dpi,
                    figsize=(6.2, 4.0),
                )

                bar_with_error_allow_na(
                    labels=["A2_outcome_disc (LR chi2)", "A2_outcome_clarity (LR chi2)", "A2_outcome_len (LR chi2)", "A2_outcome_full (LR chi2)"],
                    values=[np.nan, np.nan, np.nan, np.nan],
                    yerr=None,
                    xlabel=split_title,
                    ylabel="LR chi2 (unitless; i.i.d.)",
                    base_path=os.path.join(out_figures, f"assumption_suite_lr_a2quad_{split_label}_{dv}"),
                    rotation=25,
                    dpi=dpi,
                    figsize=(7.2, 4.0),
                )
                bar_with_error_allow_na(
                    labels=["A5_outcome_base (LR chi2)", "A5_outcome|A2_outcome_full (LR chi2)"],
                    values=[np.nan, np.nan],
                    yerr=None,
                    xlabel=split_title,
                    ylabel="LR chi2 (unitless; i.i.d.)",
                    base_path=os.path.join(out_figures, f"assumption_suite_lr_a5pair_{split_label}_{dv}"),
                    rotation=15,
                    dpi=dpi,
                    figsize=(6.2, 4.0),
                )

            shapley_rows.append(
                {
                    "disc_variant": dv,
                    "split": "test",
                    "metric": "loglik_test",
                    "ll_base": np.nan,
                    "ll_full": np.nan,
                    "delta_ll_full": np.nan,
                    "phi_offsets": np.nan,
                    "phi_geometry": np.nan,
                    "phi_decoder": np.nan,
                    "ratio_offsets": np.nan,
                    "ratio_geometry": np.nan,
                    "ratio_decoder": np.nan,
                    "status": status,
                    "status_reason": status_reason,
                }
            )
            continue

        # ---------------------------------------------------------------------
        # Derived geometry features (review-ready sensitivities)
        # ---------------------------------------------------------------------
        # Many reviewers ask to report / control the raw geometry summaries in
        # addition to any collapsed "disc" index. We therefore carry |t1|, |t2|,
        # d12 (and |d12|) forward so we can run:
        #   - expanded-geometry GLMs (|t1|, |t2|, d12 simultaneously)
        #   - interaction sensitivities (|t1|×|t2|)
        #   - calibration-by-condition diagnostics
        for _df in (train_v, test_v):
            _df["abs_t1"] = _df["t1"].abs()
            _df["abs_t2"] = _df["t2"].abs()
            _df["abs_d12"] = _df["d12"].abs()

        # Standardize (train-only) on the variant-specific train subset.
        # Include both token and character length proxies to reproduce the original archive's appendix outputs.
        # We also standardize the additional geometry columns for reviewer-requested sensitivity analyses.
        std = fit_standardizer(
            train_v,
            cols=["disc", "clarity", "len_chars", "len_tokens", "abs_t1", "abs_t2", "d12", "abs_d12"],
        )
        train_z = std.transform(train_v, prefix="z_")
        test_z = std.transform(test_v, prefix="z_")

        # Rename standardized disc to a common column name used downstream
        train_z = train_z.rename(columns={"z_disc": "z_disc"})
        test_z = test_z.rename(columns={"z_disc": "z_disc"})

        # Degenerate (rank-deficient) variant check: disc vs clarity collinearity.
        status, status_reason, _deg_meta = _degeneracy_status(train_z, disc_col="z_disc", clarity_col="z_clarity")

        if status != "OK":
            warnings_rows.append(
                {
                    "disc_variant": dv,
                    "issue": "degenerate_variant",
                    "status": status,
                    "status_reason": status_reason,
                    **_deg_meta,
                }
            )

            # Record NA rows (so the downstream CSV is deterministic).
            for df_z, split_label in [(train_z, "train"), (test_z, "test")]:
                n_clusters = int(df_z["worker"].nunique())
                n_obs_split = int(df_z.shape[0])

                assumption_rows.append(
                    {
                        "split": split_label,
                        "disc_variant": dv,
                        "status": status,
                        "status_reason": status_reason,
                        "n_obs": n_obs_split,
                        "n_clusters": n_clusters,
                        "A2_outcome_disc_wald_chi2": np.nan,
                        "A2_outcome_disc_df": np.nan,
                        "A2_outcome_disc_p": np.nan,
                        "A2_outcome_clarity_wald_chi2": np.nan,
                        "A2_outcome_clarity_df": np.nan,
                        "A2_outcome_clarity_p": np.nan,
                        "A2_outcome_len_wald_chi2": np.nan,
                        "A2_outcome_len_df": np.nan,
                        "A2_outcome_len_p": np.nan,
                        "A2_outcome_full_wald_chi2": np.nan,
                        "A2_outcome_full_df": np.nan,
                        "A2_outcome_full_p": np.nan,
                        "A3_clarity_t": np.nan,
                        "A4_disc_t": np.nan,
                        "A5_outcome_base_wald_chi2": np.nan,
                        "A5_outcome_base_df": np.nan,
                        "A5_outcome_base_p": np.nan,
                        "A5_outcome_given_A2_outcome_full_wald_chi2": np.nan,
                        "A5_outcome_given_A2_outcome_full_df": np.nan,
                        "A5_outcome_given_A2_outcome_full_p": np.nan,
                    }
                )


                assumption_lr_rows.append(
                    {
                        "split": split_label,
                        "disc_variant": dv,
                        "status": status,
                        "status_reason": status_reason,
                        "n_obs": n_obs_split,
                        "n_clusters": n_clusters,
                        "A2_outcome_disc_lr_chi2": np.nan,
                        "A2_outcome_disc_lr_df": np.nan,
                        "A2_outcome_disc_lr_p": np.nan,
                        "A2_outcome_clarity_lr_chi2": np.nan,
                        "A2_outcome_clarity_lr_df": np.nan,
                        "A2_outcome_clarity_lr_p": np.nan,
                        "A2_outcome_len_lr_chi2": np.nan,
                        "A2_outcome_len_lr_df": np.nan,
                        "A2_outcome_len_lr_p": np.nan,
                        "A2_outcome_full_lr_chi2": np.nan,
                        "A2_outcome_full_lr_df": np.nan,
                        "A2_outcome_full_lr_p": np.nan,
                        "A5_outcome_base_lr_chi2": np.nan,
                        "A5_outcome_base_lr_df": np.nan,
                        "A5_outcome_base_lr_p": np.nan,
                        "A5_outcome_given_A2_outcome_full_lr_chi2": np.nan,
                        "A5_outcome_given_A2_outcome_full_lr_df": np.nan,
                        "A5_outcome_given_A2_outcome_full_lr_p": np.nan,
                    }
                )

                # Placeholder figures so NA is visible at the artifact level.
                split_title = str(split_label).strip().title()
                bar_with_error_allow_na(
                    labels=["A2_outcome_disc", "A2_outcome_clarity", "A2_outcome_len", "A2_outcome_full"],
                    values=[np.nan, np.nan, np.nan, np.nan],
                    yerr=None,
                    xlabel=split_title,
                    ylabel="Wald chi2 (unitless)",
                    base_path=os.path.join(out_figures, f"assumption_suite_wald_a2quad_{split_label}_{dv}"),
                    rotation=25,
                    dpi=dpi,
                    figsize=(7.2, 4.0),
                )
                bar_with_error_allow_na(
                    labels=["A3 (t: clarity)", "A4 (t: disc)"],
                    values=[np.nan, np.nan],
                    yerr=None,
                    xlabel=split_title,
                    ylabel="t statistic (unitless)",
                    base_path=os.path.join(out_figures, f"assumption_suite_wald_a34only_{split_label}_{dv}"),
                    rotation=0,
                    dpi=dpi,
                    figsize=(4.8, 4.0),
                )
                bar_with_error_allow_na(
                    labels=["A5_outcome_base (Wald chi2)", "A5_outcome|A2_outcome_full (Wald chi2)"],
                    values=[np.nan, np.nan],
                    yerr=None,
                    xlabel=split_title,
                    ylabel="Wald chi2 (unitless)",
                    base_path=os.path.join(out_figures, f"assumption_suite_wald_a5pair_{split_label}_{dv}"),
                    rotation=15,
                    dpi=dpi,
                    figsize=(6.2, 4.0),
                )

                bar_with_error_allow_na(
                    labels=["A2_outcome_disc (LR chi2)", "A2_outcome_clarity (LR chi2)", "A2_outcome_len (LR chi2)", "A2_outcome_full (LR chi2)"],
                    values=[np.nan, np.nan, np.nan, np.nan],
                    yerr=None,
                    xlabel=split_title,
                    ylabel="LR chi2 (unitless; i.i.d.)",
                    base_path=os.path.join(out_figures, f"assumption_suite_lr_a2quad_{split_label}_{dv}"),
                    rotation=25,
                    dpi=dpi,
                    figsize=(7.2, 4.0),
                )
                bar_with_error_allow_na(
                    labels=["A5_outcome_base (LR chi2)", "A5_outcome|A2_outcome_full (LR chi2)"],
                    values=[np.nan, np.nan],
                    yerr=None,
                    xlabel=split_title,
                    ylabel="LR chi2 (unitless; i.i.d.)",
                    base_path=os.path.join(out_figures, f"assumption_suite_lr_a5pair_{split_label}_{dv}"),
                    rotation=15,
                    dpi=dpi,
                    figsize=(6.2, 4.0),
                )

            # Shapley attribution is undefined when the geometry block is rank-deficient.
            shapley_rows.append(
                {
                    "disc_variant": dv,
                    "split": "test",
                    "metric": "loglik_test",
                    "ll_base": np.nan,
                    "ll_full": np.nan,
                    "delta_ll_full": np.nan,
                    "phi_offsets": np.nan,
                    "phi_geometry": np.nan,
                    "phi_decoder": np.nan,
                    "ratio_offsets": np.nan,
                    "ratio_geometry": np.nan,
                    "ratio_decoder": np.nan,
                    "status": status,
                    "status_reason": status_reason,
                }
            )
            continue

        # Cache the fully-prepared (standardized, split-specific) data for the
        # primary disc variant so we can run additional review-oriented
        # sensitivities (link function, alternative clustering, calibration,
        # mixed effects, Shapley-with-interactions) after the main multiverse loop.
        if dv == primary_variant:
            primary_cache = {
                "disc_variant": dv,
                "train_z": train_z.copy(),
                "test_z": test_z.copy(),
            }

        # Stepwise BIC (legacy analysis; computed on TRAIN only)
        stepwise_bic(
            train_z,
            out_tables,
            out_figures,
            disc_variant=dv,
            split_label="train",
            dpi=dpi,
        )

        # Legacy appendix diagnostics (feature histograms, VIF, split-half stability)
        # Run on the primary disc definition only, to keep output volume manageable.
        if dv == primary_variant:
            feature_histograms(
                train_z,
                out_tables=out_tables,
                out_figures=out_figures,
                disc_variant=dv,
                split_label="train",
                dpi=dpi,
            )
            vif_table_and_plot(
                train_z,
                out_tables=out_tables,
                out_figures=out_figures,
                disc_variant=dv,
                split_label="train",
                dpi=dpi,
            )
            split_half_stability(
                train_z,
                out_tables=out_tables,
                out_figures=out_figures,
                disc_variant=dv,
                split_label="train",
                seed=int(seed),
                dpi=dpi,
            )

        # Assumption suites on TRAIN and TEST (re-run on test = minimal reproducibility check)
        for df_z, split_label in [(train_z, "train"), (test_z, "test")]:
            diag, coef_a34, coef_a5 = run_assumption_suite(
                df_z,
                split_label=split_label,
                disc_variant=dv,
                disc_z_col="z_disc",
                condition_categories=cond_categories,
                cluster_cols=str(cluster_spec),
                wald_bootstrap=str(wald_bootstrap),
                wald_n_boot=int(wald_n_boot),
                wald_bootstrap_seed=int(seed),
                wald_bootstrap_weight=str(wald_bootstrap_weight),
            )

            # Export A5 coefficient table (cluster-robust) as CSV (used in the manuscript text/tables).
            coef_a5_out = coef_a5.copy()
            coef_a5_out.insert(0, "split", split_label)
            coef_a5_out.insert(1, "disc_variant", dv)
            save_df(coef_a5_out, os.path.join(out_tables, f"a5_coefficients_cluster_robust_{split_label}_{dv}.csv"))

            # Record LR (working-i.i.d.) diagnostics in a separate table (legacy-style nested tests).
            assumption_lr_rows.append(
                {
                    "split": split_label,
                    "disc_variant": dv,
                    "cluster_spec": str(cluster_spec),
                    "status": status,
                    "status_reason": status_reason,
                    "n_obs": int(diag.n_obs),
                    "n_clusters": int(diag.n_clusters),

                    "A2_outcome_disc_lr_chi2": float(diag.a2_lr_chi2),
                    "A2_outcome_disc_lr_df": int(diag.a2_lr_df),
                    "A2_outcome_disc_lr_p": float(diag.a2_lr_p_value),

                    "A2_outcome_clarity_lr_chi2": float(diag.a2_clarity_lr_chi2),
                    "A2_outcome_clarity_lr_df": int(diag.a2_clarity_lr_df),
                    "A2_outcome_clarity_lr_p": float(diag.a2_clarity_lr_p_value),

                    "A2_outcome_len_lr_chi2": float(diag.a2_len_lr_chi2),
                    "A2_outcome_len_lr_df": int(diag.a2_len_lr_df),
                    "A2_outcome_len_lr_p": float(diag.a2_len_lr_p_value),

                    "A2_outcome_full_lr_chi2": float(diag.a2_full_lr_chi2),
                    "A2_outcome_full_lr_df": int(diag.a2_full_lr_df),
                    "A2_outcome_full_lr_p": float(diag.a2_full_lr_p_value),

                    "A5_outcome_base_lr_chi2": float(diag.a5_lr_chi2),
                    "A5_outcome_base_lr_df": int(diag.a5_lr_df),
                    "A5_outcome_base_lr_p": float(diag.a5_lr_p_value),

                    "A5_outcome_given_A2_outcome_full_lr_chi2": float(diag.a5_conditional_lr_chi2),
                    "A5_outcome_given_A2_outcome_full_lr_df": int(diag.a5_conditional_lr_df),
                    "A5_outcome_given_A2_outcome_full_lr_p": float(diag.a5_conditional_lr_p_value),
                }
            )

            # Record minimal assumption-test summary (manuscript-aligned outputs only)
            assumption_rows.append(
                {
                    "split": split_label,
                    "disc_variant": dv,
                    "cluster_spec": str(cluster_spec),
                    "status": status,
                    "status_reason": status_reason,
                    "n_obs": int(diag.n_obs),
                    "n_clusters": int(diag.n_clusters),

                    "A2_outcome_disc_wald_chi2": float(diag.a2_block.stat_chi2),
                    "A2_outcome_disc_df": int(diag.a2_block.df),
                    "A2_outcome_disc_p": float(diag.a2_block.p_value_chi2),
                    "A2_outcome_disc_p_wild": float(diag.a2_block.p_value_wild) if diag.a2_block.p_value_wild is not None else float("nan"),

                    "A2_outcome_clarity_wald_chi2": float(diag.a2_clarity_block.stat_chi2),
                    "A2_outcome_clarity_df": int(diag.a2_clarity_block.df),
                    "A2_outcome_clarity_p": float(diag.a2_clarity_block.p_value_chi2),
                    "A2_outcome_clarity_p_wild": float(diag.a2_clarity_block.p_value_wild) if diag.a2_clarity_block.p_value_wild is not None else float("nan"),

                    "A2_outcome_len_wald_chi2": float(diag.a2_len_block.stat_chi2),
                    "A2_outcome_len_df": int(diag.a2_len_block.df),
                    "A2_outcome_len_p": float(diag.a2_len_block.p_value_chi2),
                    "A2_outcome_len_p_wild": float(diag.a2_len_block.p_value_wild) if diag.a2_len_block.p_value_wild is not None else float("nan"),

                    "A2_outcome_full_wald_chi2": float(diag.a2_full_block.stat_chi2),
                    "A2_outcome_full_df": int(diag.a2_full_block.df),
                    "A2_outcome_full_p": float(diag.a2_full_block.p_value_chi2),
                    "A2_outcome_full_p_wild": float(diag.a2_full_block.p_value_wild) if diag.a2_full_block.p_value_wild is not None else float("nan"),

                    "A3_clarity_t": float(diag.a3_t_clarity),
                    "A4_disc_t": float(diag.a4_t_disc),

                    "A5_outcome_base_wald_chi2": float(diag.a5_block.stat_chi2),
                    "A5_outcome_base_df": int(diag.a5_block.df),
                    "A5_outcome_base_p": float(diag.a5_block.p_value_chi2),
                    "A5_outcome_base_p_wild": float(diag.a5_block.p_value_wild) if diag.a5_block.p_value_wild is not None else float("nan"),

                    "A5_outcome_given_A2_outcome_full_wald_chi2": float(diag.a5_conditional_block.stat_chi2),
                    "A5_outcome_given_A2_outcome_full_df": int(diag.a5_conditional_block.df),
                    "A5_outcome_given_A2_outcome_full_p": float(diag.a5_conditional_block.p_value_chi2),
                    "A5_outcome_given_A2_outcome_full_p_wild": float(diag.a5_conditional_block.p_value_wild) if diag.a5_conditional_block.p_value_wild is not None else float("nan"),
                }
            )

            # Assumption-suite bar plots (Wald; manuscript) + LR (legacy-style nested tests)
            assumption_summary_plots(diag, out_figures=out_figures, dpi=dpi)
            assumption_summary_plots_lr(diag, out_figures=out_figures, dpi=dpi)

            # Appendix-style condition intercept visualization (from the A5 full model)
            plot_condition_intercepts_from_a5(
                coef_a5,
                out_figures=out_figures,
                disc_variant=dv,
                split_label=split_label,
                dpi=dpi,
            )

            # Optional: coefficient plot (disc/clarity/length) with error bars
            fit_and_plot_coefficients(
                df_z,
                out_tables,
                out_figures,
                disc_variant=dv,
                split_label=split_label,
                cluster_cols=str(cluster_spec),
                dpi=dpi,
            )

        # ------------------------ Shapley three-way decomposition ------------------------
        # Fit on TRAIN, evaluate on TEST (strictly out-of-sample)

        # Build stable condition dummies on BOTH splits (using train categories)
        dum_tr = condition_dummies(train_z, categories=cond_categories)
        dum_te = condition_dummies(test_z, categories=cond_categories)

        train_sh = pd.concat([train_z, dum_tr], axis=1)
        test_sh = pd.concat([test_z, dum_te], axis=1)

        group_cols = {
            "offsets": list(dum_tr.columns),
            "geometry": ["z_disc", "z_clarity"],
            "decoder": ["z_len_chars"],
        }

        shap_res, ll_table, boot_df = compute_shapley_threeway(
            train_sh,
            test_sh,
            disc_variant=dv,
            group_cols=group_cols,
            bootstrap=shapley_bootstrap,
            n_boot=int(shapley_n_boot),
            seed=int(seed),
            cluster_cols=str(cluster_spec),
            bootstrap_method=str(shapley_bootstrap_method),
        )

        shap_row = asdict(shap_res)

        # Record the bootstrap clustering specification used for uncertainty.
        shap_row["cluster_spec"] = str(cluster_spec)
        shap_row["shapley_bootstrap_method"] = str(shapley_bootstrap_method)

        # Track degenerate exclusions at the aggregated CSV stage.
        shap_row["status"] = status
        shap_row["status_reason"] = status_reason
        shapley_rows.append(shap_row)

        save_df(ll_table, os.path.join(out_tables, f"shapley_subset_loglik_{dv}.csv"))

        if boot_df is not None:
            save_df(boot_df, os.path.join(out_tables, f"shapley_bootstrap_draws_{dv}.csv"))
            summ = summarize_bootstrap(
                boot_df,
                cols=[
                    "delta_ll_full",
                    "phi_offsets",
                    "phi_geometry",
                    "phi_decoder",
                    "ratio_offsets",
                    "ratio_geometry",
                    "ratio_decoder",
                ],
            )
            save_df(summ, os.path.join(out_tables, f"shapley_bootstrap_summary_{dv}.csv"))

            # Plot ratios with bootstrap CI (error bars) — matches the "error bar" requirement.
            # Use the median and percentile band.
            def _get(metric: str):
                row = summ.loc[summ["metric"] == metric]
                return float(row["q500"].values[0]), float(row["q025"].values[0]), float(row["q975"].values[0])

            med_p, lo_p, hi_p = _get("ratio_offsets")
            med_g, lo_g, hi_g = _get("ratio_geometry")
            med_d, lo_d, hi_d = _get("ratio_decoder")

            yvals = [med_p, med_g, med_d]
            yerr = np.vstack([
                [med_p - lo_p, med_g - lo_g, med_d - lo_d],
                [hi_p - med_p, hi_g - med_g, hi_d - med_d],
            ])

            # Manuscript-facing label: "offsets" corresponds to condition offsets/intercepts.
            bar_with_error(
                labels=["offsets", "geometry", "decoder"],
                values=yvals,
                yerr=yerr,
                xlabel="Component",
                ylabel="Share of Δ log-likelihood (proportion)",
                base_path=os.path.join(out_figures, f"shapley_ratio_{dv}"),
                rotation=0,
                figsize=(5.0, 4.0),
                dpi=dpi,
            )

    # Save aggregated tables (assumption tests are manuscript-aligned and intentionally minimal)
    if assumption_rows:
        assump_df = pd.DataFrame(assumption_rows)
        save_df(assump_df, os.path.join(out_tables, "assumption_tests_minimal.csv"))

        # A5 across disc definitions (base A5 only): one plot per split
        for split_label in ["train", "test"]:
            sub = assump_df.loc[assump_df["split"] == split_label].copy()
            labels = [str(dv) for dv in selected]
            vals = []
            for dv in selected:
                r = sub.loc[sub["disc_variant"] == dv]
                if r.empty:
                    vals.append(np.nan)
                else:
                    vals.append(float(r["A5_outcome_base_wald_chi2"].values[0]))

            # Wider figure for multiverse-style bars
            fig_w = float(max(7.0, 0.45 * len(labels) + 2.0))
            bar_with_error_allow_na(
                labels=labels,
                values=vals,
                yerr=None,
                xlabel=str(split_label).strip().title(),
                ylabel="A5_outcome_base Wald chi2 (unitless)",
                base_path=os.path.join(out_figures, f"assumption_suite_wald_a5_across_disc_variants_{split_label}"),
                rotation=45,
                dpi=dpi,
                figsize=(fig_w, 4.0),
            )
    else:
        save_df(pd.DataFrame([]), os.path.join(out_tables, "assumption_tests_minimal.csv"))


    # LR (working-i.i.d.) nested-test diagnostics (separate wide table)
    if assumption_lr_rows:
        save_df(pd.DataFrame(assumption_lr_rows), os.path.join(out_tables, "assumption_tests_lr.csv"))
    else:
        save_df(pd.DataFrame([]), os.path.join(out_tables, "assumption_tests_lr.csv"))

    save_df(pd.DataFrame(shapley_rows), os.path.join(out_tables, "shapley_threeway_all.csv"))
    if warnings_rows:
        save_df(pd.DataFrame(warnings_rows), os.path.join(out_tables, "pipeline_warnings.csv"))

    # ---------------------------------------------------------------------
    # Review-oriented sensitivities (primary disc variant only)
    # ---------------------------------------------------------------------
    if primary_cache is not None:
        _run_review_sensitivities(
            primary_cache,
            out_tables=out_tables,
            out_figures=out_figures,
            condition_categories=cond_categories,
            seed=int(seed),
            shapley_bootstrap=bool(shapley_bootstrap),
            shapley_n_boot=int(shapley_n_boot),
            dpi=int(dpi),
        )


def _run_review_sensitivities(
    cache: Dict[str, pd.DataFrame],
    *,
    out_tables: str,
    out_figures: str,
    condition_categories: Sequence[str],
    seed: int,
    shapley_bootstrap: bool,
    shapley_n_boot: int,
    dpi: int,
) -> None:
    """Run reviewer-requested sensitivity analyses on the primary disc variant.

    This function is intentionally conservative: it does not change the core
    multiverse outputs, but writes additional tables/figures under
        {tables,figures}/sensitivity/
    so authors can selectively include them as supplementary material.
    """

    # Output subfolders
    out_t = os.path.join(out_tables, "sensitivity")
    out_f = os.path.join(out_figures, "sensitivity")
    ensure_dir(out_t)
    ensure_dir(out_f)

    disc_variant = str(cache.get("disc_variant", "primary"))
    train_z = cache["train_z"].copy()
    test_z = cache["test_z"].copy()

    # ------------------------------------------------------------------
    # 1) Link-function sensitivity (logit vs probit) + alternative clustering
    # ------------------------------------------------------------------
    sens_rows: List[Dict] = []
    for link in ["logit", "probit"]:
        for cluster_col in ["worker", "game"]:
            # Skip game clustering if the column is missing or degenerate
            if cluster_col not in train_z.columns:
                continue

            for df_z, split_label in [(train_z, "train"), (test_z, "test")]:
                if cluster_col not in df_z.columns:
                    continue
                try:
                    diag, coef_a34, coef_a5 = run_assumption_suite(
                        df_z,
                        split_label=split_label,
                        disc_variant=disc_variant,
                        disc_z_col="z_disc",
                        condition_categories=condition_categories,
                        cluster_cols=cluster_col,
                        link=link,
                    )

                    # Save coefficient tables (A3/A4 and A5-full model)
                    save_df(
                        coef_a34,
                        os.path.join(out_t, f"assumption_a34_coefs_{split_label}_{link}_cluster_{cluster_col}.csv"),
                    )
                    save_df(
                        coef_a5,
                        os.path.join(out_t, f"assumption_a5_coefs_{split_label}_{link}_cluster_{cluster_col}.csv"),
                    )

                    sens_rows.append(
                        {
                            "disc_variant": disc_variant,
                            "split": split_label,
                            "link": link,
                            "cluster_col": cluster_col,
                            "n_obs": int(diag.n_obs),
                            "n_clusters": int(diag.n_clusters),
                            "A2_outcome_disc_block_wald_chi2": float(diag.a2_block.stat_chi2),
                            "A2_outcome_disc_block_p": float(diag.a2_block.p_value_chi2),
                            "A2_outcome_full_block_wald_chi2": float(diag.a2_full_block.stat_chi2),
                            "A2_outcome_full_block_p": float(diag.a2_full_block.p_value_chi2),
                            "A5_outcome_base_block_wald_chi2": float(diag.a5_block.stat_chi2),
                            "A5_outcome_base_block_p": float(diag.a5_block.p_value_chi2),
                            "A5_outcome_given_A2_outcome_full_block_wald_chi2": float(diag.a5_conditional_block.stat_chi2),
                            "A5_outcome_given_A2_outcome_full_block_p": float(diag.a5_conditional_block.p_value_chi2),
                            "status": "OK",
                        }
                    )
                except Exception as e:
                    sens_rows.append(
                        {
                            "disc_variant": disc_variant,
                            "split": split_label,
                            "link": link,
                            "cluster_col": cluster_col,
                            "status": "ERROR",
                            "error": repr(e),
                        }
                    )

    save_df(pd.DataFrame(sens_rows), os.path.join(out_t, "assumption_suite_link_and_cluster_sensitivity.csv"))

    # ------------------------------------------------------------------
    # 1b) Scale-adjusted (heteroskedastic) logit sensitivity
    #     (adds A2disc/A2clarity/A2len/A2full under a scale model by condition)
    # ------------------------------------------------------------------
    try:
        from .scale_adjusted_logit import run_scale_adjusted_logit_suite

        sal_rows: List[Dict] = []
        for df_z, split_label in [(train_z, "train"), (test_z, "test")]:
            try:
                row, coefs_base, coefs_full = run_scale_adjusted_logit_suite(
                    df_z,
                    disc_variant=disc_variant,
                    split_label=split_label,
                    condition_categories=condition_categories,
                    cluster_cols="worker",
                    disc_col="z_disc",
                    clarity_col="z_clarity",
                    len_col="z_len_chars",
                )
                sal_rows.append(asdict(row))

                save_df(coefs_base, os.path.join(out_t, f"scale_adjusted_logit_coefs_base_{split_label}.csv"))
                save_df(coefs_full, os.path.join(out_t, f"scale_adjusted_logit_coefs_full_{split_label}.csv"))

            except Exception as e:
                sal_rows.append(
                    {
                        "disc_variant": disc_variant,
                        "split": split_label,
                        "status": "ERROR",
                        "error": repr(e),
                    }
                )

        save_df(pd.DataFrame(sal_rows), os.path.join(out_t, "assumption_suite_scale_adjusted_logit.csv"))

    except Exception as e:
        # If the module is missing / fails to import, keep the pipeline running.
        save_df(
            pd.DataFrame([{"status": "ERROR", "error": f"scale_adjusted_logit import failed: {repr(e)}"}]),
            os.path.join(out_t, "assumption_suite_scale_adjusted_logit.csv"),
        )


    # ------------------------------------------------------------------
    # 2) Expanded-geometry GLMs: |t1|, |t2|, d12 simultaneously (+ interaction)
    # ------------------------------------------------------------------
    # Build stable condition dummies (train categories fixed)
    dum_tr = condition_dummies(train_z, categories=condition_categories, prefix="cond")
    dum_te = condition_dummies(test_z, categories=condition_categories, prefix="cond")

    train_aug = pd.concat([train_z, dum_tr], axis=1)
    test_aug = pd.concat([test_z, dum_te], axis=1)

    # Create interaction term |t1|×|t2| (standardized-product version)
    for _df in (train_aug, test_aug):
        if "z_abs_t1" in _df.columns and "z_abs_t2" in _df.columns:
            _df["z_abs_t1_x_abs_t2"] = _df["z_abs_t1"].to_numpy(dtype=float) * _df["z_abs_t2"].to_numpy(dtype=float)

    offsets_cols = list(dum_tr.columns)
    base_cols = offsets_cols + ["z_disc", "z_clarity", "z_len_chars"]

    # Expanded geometry requires d12 (complete cases only)
    train_d12 = train_aug.loc[np.isfinite(train_aug["z_d12"].to_numpy(dtype=float))].copy() if "z_d12" in train_aug.columns else train_aug.iloc[0:0].copy()
    test_d12 = test_aug.loc[np.isfinite(test_aug["z_d12"].to_numpy(dtype=float))].copy() if "z_d12" in test_aug.columns else test_aug.iloc[0:0].copy()

    # Expanded-geometry model specs
    #
    # NOTE:
    # - "With disc" keeps the headline geometry proxy (z_disc) alongside the more
    #   directly interpretable components (|t1|, |t2|, d12).
    # - "No disc" drops z_disc to quantify whether the expanded components explain
    #   performance without relying on the composite score.
    geom_cols_add = offsets_cols + ["z_disc", "z_abs_t1", "z_abs_t2", "z_d12", "z_len_chars"]
    geom_cols_int = geom_cols_add + ["z_abs_t1_x_abs_t2"]

    geom_cols_add_no_disc = offsets_cols + ["z_abs_t1", "z_abs_t2", "z_d12", "z_len_chars"]
    geom_cols_int_no_disc = geom_cols_add_no_disc + ["z_abs_t1_x_abs_t2"]

    glm_rows: List[Dict] = []
    for link in ["logit", "probit"]:
        for cluster_col in ["worker", "game"]:
            if cluster_col not in train_aug.columns:
                continue

            # --- Baseline full model (matches Shapley full model) ---
            meta, res, coef = fit_cluster_robust_glm(
                train_aug,
                x_cols=base_cols,
                cluster_cols=cluster_col,
                link=link,
            )

            # Evaluate on test
            X_te = add_const(test_aug[base_cols])
            p_te = np.asarray(res.predict(X_te), dtype=float)
            ll_te = log_likelihood_from_probs(test_aug["y"].to_numpy(dtype=float), p_te)
            bs_te = brier_score(test_aug["y"].to_numpy(dtype=float), p_te)

            save_df(
                coef,
                os.path.join(out_t, f"glm_base_full_coefs_train_{link}_cluster_{cluster_col}.csv"),
            )

            glm_rows.append(
                {
                    "model": "base_full",
                    "disc_variant": disc_variant,
                    "link": link,
                    "cluster_col": cluster_col,
                    "n_train": int(train_aug.shape[0]),
                    "n_test": int(test_aug.shape[0]),
                    "ll_test": float(ll_te),
                    "brier_test": float(bs_te),
                    "n_clusters": int(meta.n_clusters),
                }
            )

            # --- Expanded geometry (complete-case d12) ---
            if train_d12.shape[0] > 0 and test_d12.shape[0] > 0:
                meta_g, res_g, coef_g = fit_cluster_robust_glm(
                    train_d12,
                    x_cols=geom_cols_add,
                    cluster_cols=cluster_col,
                    link=link,
                )
                Xg_te = add_const(test_d12[geom_cols_add])
                pg_te = np.asarray(res_g.predict(Xg_te), dtype=float)
                llg_te = log_likelihood_from_probs(test_d12["y"].to_numpy(dtype=float), pg_te)
                bsg_te = brier_score(test_d12["y"].to_numpy(dtype=float), pg_te)

                save_df(
                    coef_g,
                    os.path.join(out_t, f"glm_geom_full_add_coefs_train_{link}_cluster_{cluster_col}.csv"),
                )
                glm_rows.append(
                    {
                        "model": "geom_full_add",
                        "disc_variant": disc_variant,
                        "link": link,
                        "cluster_col": cluster_col,
                        "n_train": int(train_d12.shape[0]),
                        "n_test": int(test_d12.shape[0]),
                        "ll_test": float(llg_te),
                        "brier_test": float(bsg_te),
                        "n_clusters": int(meta_g.n_clusters),
                    }
                )

                # --- Expanded geometry WITHOUT z_disc (complete-case d12) ---
                meta_g0, res_g0, coef_g0 = fit_cluster_robust_glm(
                    train_d12,
                    x_cols=geom_cols_add_no_disc,
                        cluster_cols=cluster_col,
                    link=link,
                )
                Xg0_te = add_const(test_d12[geom_cols_add_no_disc])
                pg0_te = np.asarray(res_g0.predict(Xg0_te), dtype=float)
                llg0_te = log_likelihood_from_probs(test_d12["y"].to_numpy(dtype=float), pg0_te)
                bsg0_te = brier_score(test_d12["y"].to_numpy(dtype=float), pg0_te)

                save_df(
                    coef_g0,
                    os.path.join(out_t, f"glm_geom_full_add_no_disc_coefs_train_{link}_cluster_{cluster_col}.csv"),
                )
                glm_rows.append(
                    {
                        "model": "geom_full_add_no_disc",
                        "disc_variant": disc_variant,
                        "link": link,
                        "cluster_col": cluster_col,
                        "n_train": int(train_d12.shape[0]),
                        "n_test": int(test_d12.shape[0]),
                        "ll_test": float(llg0_te),
                        "brier_test": float(bsg0_te),
                        "n_clusters": int(meta_g0.n_clusters),
                    }
                )

                if "z_abs_t1_x_abs_t2" in train_d12.columns and "z_abs_t1_x_abs_t2" in test_d12.columns:
                    meta_gi, res_gi, coef_gi = fit_cluster_robust_glm(
                        train_d12,
                        x_cols=geom_cols_int,
                        cluster_cols=cluster_col,
                        link=link,
                    )
                    Xgi_te = add_const(test_d12[geom_cols_int])
                    pgi_te = np.asarray(res_gi.predict(Xgi_te), dtype=float)
                    llgi_te = log_likelihood_from_probs(test_d12["y"].to_numpy(dtype=float), pgi_te)
                    bsgi_te = brier_score(test_d12["y"].to_numpy(dtype=float), pgi_te)

                    save_df(
                        coef_gi,
                        os.path.join(out_t, f"glm_geom_full_int_coefs_train_{link}_cluster_{cluster_col}.csv"),
                    )
                    glm_rows.append(
                        {
                            "model": "geom_full_int_abs_t1_x_abs_t2",
                            "disc_variant": disc_variant,
                            "link": link,
                            "cluster_col": cluster_col,
                            "n_train": int(train_d12.shape[0]),
                            "n_test": int(test_d12.shape[0]),
                            "ll_test": float(llgi_te),
                            "brier_test": float(bsgi_te),
                            "n_clusters": int(meta_gi.n_clusters),
                        }
                    )

                    # --- Expanded geometry interaction WITHOUT z_disc ---
                    meta_gi0, res_gi0, coef_gi0 = fit_cluster_robust_glm(
                        train_d12,
                        x_cols=geom_cols_int_no_disc,
                        cluster_cols=cluster_col,
                        link=link,
                    )
                    Xgi0_te = add_const(test_d12[geom_cols_int_no_disc])
                    pgi0_te = np.asarray(res_gi0.predict(Xgi0_te), dtype=float)
                    llgi0_te = log_likelihood_from_probs(test_d12["y"].to_numpy(dtype=float), pgi0_te)
                    bsgi0_te = brier_score(test_d12["y"].to_numpy(dtype=float), pgi0_te)

                    save_df(
                        coef_gi0,
                        os.path.join(out_t, f"glm_geom_full_int_no_disc_coefs_train_{link}_cluster_{cluster_col}.csv"),
                    )
                    glm_rows.append(
                        {
                            "model": "geom_full_int_abs_t1_x_abs_t2_no_disc",
                            "disc_variant": disc_variant,
                            "link": link,
                            "cluster_col": cluster_col,
                            "n_train": int(train_d12.shape[0]),
                            "n_test": int(test_d12.shape[0]),
                            "ll_test": float(llgi0_te),
                            "brier_test": float(bsgi0_te),
                            "n_clusters": int(meta_gi0.n_clusters),
                        }
                    )

    save_df(pd.DataFrame(glm_rows), os.path.join(out_t, "glm_link_cluster_and_geometry_sensitivity_summary.csv"))

    # ------------------------------------------------------------------
    # 3) Condition-wise calibration (predicted probability fit)
    # ------------------------------------------------------------------
    for link in ["logit", "probit"]:
        # Fit baseline full model on train (cluster choice doesn't change point estimates)
        _meta, res, _coef = fit_cluster_robust_glm(
            train_aug,
            x_cols=base_cols,
            cluster_cols="worker" if "worker" in train_aug.columns else "game",
            link=link,
        )
        p_te = np.asarray(res.predict(add_const(test_aug[base_cols])), dtype=float)
        tmp = test_aug[["y", "condition"]].copy()
        tmp["p_hat"] = p_te

        curve_df, sum_df = calibration_by_group(tmp, y_col="y", p_col="p_hat", group_col="condition", n_bins=10)
        save_df(curve_df, os.path.join(out_t, f"calibration_curve_by_condition_test_{link}.csv"))
        save_df(sum_df, os.path.join(out_t, f"calibration_summary_by_condition_test_{link}.csv"))
        plot_calibration_curves(
            curve_df.rename(columns={"group": "group"}),
            base_path=os.path.join(out_f, f"calibration_by_condition_test_{link}"),
            title=f"Calibration by condition (test) — {link}",
            dpi=dpi,
        )

    # ------------------------------------------------------------------
    # 4) Shapley sensitivity: include slope×condition interactions in Geometry
    # ------------------------------------------------------------------
    # Add interaction columns to train/test for Shapley model subsets.
    # We keep the group partition at three factors but expand the geometry group.
    train_sh, _ = add_condition_interactions(train_z, condition_categories=condition_categories, cols=["z_disc", "z_clarity", "z_len_chars"], dummy_prefix="cond")
    test_sh, _ = add_condition_interactions(test_z, condition_categories=condition_categories, cols=["z_disc", "z_clarity", "z_len_chars"], dummy_prefix="cond")

    # Identify dummy columns and interaction columns by prefix
    offsets_cols = [c for c in train_sh.columns if c.startswith("cond_")]
    disc_int_cols = [c for c in train_sh.columns if c.startswith("z_disc_x_")]
    clarity_int_cols = [c for c in train_sh.columns if c.startswith("z_clarity_x_")]
    len_int_cols = [c for c in train_sh.columns if c.startswith("z_len_chars_x_")]

    shap_rows: List[Dict] = []

    # Bootstrap clustering options for Shapley CIs.
    cluster_opts: List[str] = []
    if "worker" in test_sh.columns:
        cluster_opts.append("worker")
    if "game" in test_sh.columns and test_sh["game"].astype(str).nunique() > 1:
        cluster_opts.append("game")
    if not cluster_opts:
        cluster_opts = ["worker"]

    for link in ["logit", "probit"]:
        # Baseline grouping (no interactions)
        group_cols_base = {
            "offsets": offsets_cols,
            "geometry": ["z_disc", "z_clarity"],
            "decoder": ["z_len_chars"],
        }
        for cl in cluster_opts:
            res_base, tab_base, boot_base = compute_shapley_threeway(
                train_sh,
                test_sh,
                disc_variant=f"{disc_variant}__{link}__base",
                group_cols=group_cols_base,
                link=link,
                bootstrap=shapley_bootstrap,
                n_boot=shapley_n_boot,
                seed=seed,
                cluster_cols=cl,
            )
            r = asdict(res_base)
            r["bootstrap_cluster_col"] = cl
            shap_rows.append(r)
            save_df(tab_base, os.path.join(out_t, f"shapley_subsets_{link}_base_cluster_{cl}.csv"))
            if boot_base is not None:
                save_df(boot_base, os.path.join(out_t, f"shapley_bootstrap_{link}_base_cluster_{cl}.csv"))
                boot_sum = summarize_bootstrap(boot_base, cols=["ratio_offsets", "ratio_geometry", "ratio_decoder"])
                save_df(boot_sum, os.path.join(out_t, f"shapley_bootstrap_summary_{link}_base_cluster_{cl}.csv"))

        # Geometry-interaction sensitivity: put disc×cond and clarity×cond into the Geometry group.
        group_cols_geom_int = {
            "offsets": offsets_cols,
            "geometry": ["z_disc", "z_clarity"] + disc_int_cols + clarity_int_cols,
            "decoder": ["z_len_chars"],
        }
        for cl in cluster_opts:
            res_gi, tab_gi, boot_gi = compute_shapley_threeway(
                train_sh,
                test_sh,
                disc_variant=f"{disc_variant}__{link}__geom_int",
                group_cols=group_cols_geom_int,
                link=link,
                bootstrap=shapley_bootstrap,
                n_boot=shapley_n_boot,
                seed=seed,
                cluster_cols=cl,
            )
            r = asdict(res_gi)
            r["bootstrap_cluster_col"] = cl
            shap_rows.append(r)
            save_df(tab_gi, os.path.join(out_t, f"shapley_subsets_{link}_geom_int_cluster_{cl}.csv"))
            if boot_gi is not None:
                save_df(boot_gi, os.path.join(out_t, f"shapley_bootstrap_{link}_geom_int_cluster_{cl}.csv"))
                boot_sum = summarize_bootstrap(boot_gi, cols=["ratio_offsets", "ratio_geometry", "ratio_decoder"])
                save_df(boot_sum, os.path.join(out_t, f"shapley_bootstrap_summary_{link}_geom_int_cluster_{cl}.csv"))

        # Full-interaction sensitivity: also allow len×cond interactions (assigned to Decoder).
        group_cols_full_int = {
            "offsets": offsets_cols,
            "geometry": ["z_disc", "z_clarity"] + disc_int_cols + clarity_int_cols,
            "decoder": ["z_len_chars"] + len_int_cols,
        }
        for cl in cluster_opts:
            res_fi, tab_fi, boot_fi = compute_shapley_threeway(
                train_sh,
                test_sh,
                disc_variant=f"{disc_variant}__{link}__full_int",
                group_cols=group_cols_full_int,
                link=link,
                bootstrap=shapley_bootstrap,
                n_boot=shapley_n_boot,
                seed=seed,
                cluster_cols=cl,
            )
            r = asdict(res_fi)
            r["bootstrap_cluster_col"] = cl
            shap_rows.append(r)
            save_df(tab_fi, os.path.join(out_t, f"shapley_subsets_{link}_full_int_cluster_{cl}.csv"))
            if boot_fi is not None:
                save_df(boot_fi, os.path.join(out_t, f"shapley_bootstrap_{link}_full_int_cluster_{cl}.csv"))
                boot_sum = summarize_bootstrap(boot_fi, cols=["ratio_offsets", "ratio_geometry", "ratio_decoder"])
                save_df(boot_sum, os.path.join(out_t, f"shapley_bootstrap_summary_{link}_full_int_cluster_{cl}.csv"))

    save_df(pd.DataFrame(shap_rows), os.path.join(out_t, "shapley_link_and_interaction_sensitivity.csv"))

    # ------------------------------------------------------------------
    # 5) Mixed-effects logistic (random intercepts)
    # ------------------------------------------------------------------
    # We fit random-intercept models as an additional, complementary robustness check.
    # These are Bayesian approximations in statsmodels and are not used for selection.
    #
    # Reviewer-oriented stability: fit each model independently so a failure in the
    # crossed model does not erase the worker-only or game-only outputs.
    df_me = train_z[["y", "z_disc", "z_clarity", "z_len_chars", "condition", "worker", "game"]].copy()
    formula = "y ~ z_disc + z_clarity + z_len_chars + C(condition)"

    def _fit_and_save(tag: str, vc: Dict[str, str]) -> None:
        try:
            fe, vcp, _ = fit_mixed_effect_logit_random_intercepts(
                df_me,
                formula=formula,
                vc_formulas=vc,
                fit_method="vb",
            )
            save_df(fe, os.path.join(out_t, f"mixed_effects_{tag}_fixed_effects.csv"))
            save_df(vcp, os.path.join(out_t, f"mixed_effects_{tag}_variance_components.csv"))
        except Exception as e:
            save_text(repr(e), os.path.join(out_t, f"mixed_effects_{tag}_error.txt"))

    # Worker random intercept
    _fit_and_save("logit_worker", {"worker": "0 + C(worker)"})

    # Game random intercept (only if meaningful)
    # Use non-missing games to avoid treating NaN as a "real" category.
    games_non_missing = df_me.loc[df_me["game"].notna(), "game"]
    if games_non_missing.astype(str).nunique() > 1:
        _fit_and_save("logit_game", {"game": "0 + C(game)"})
        _fit_and_save("logit_worker_game", {"worker": "0 + C(worker)", "game": "0 + C(game)"})

