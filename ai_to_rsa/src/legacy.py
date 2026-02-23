from __future__ import annotations

"""Legacy/appendix diagnostics retained from the original archive.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import os

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .plotting import bar_with_error, histogram, scatter


def _available(df: pd.DataFrame, cols: Sequence[str]) -> List[str]:
    return [c for c in cols if c in df.columns]


def feature_histograms(
    df: pd.DataFrame,
    *,
    out_tables: str,
    out_figures: str,
    disc_variant: str,
    split_label: str,
    dpi: int,
    bins: int = 30,
    cols: Optional[Sequence[str]] = None,
) -> None:
    """Histograms of standardized predictors with a matching bin-count table."""

    if cols is None:
        cols = ["z_disc", "z_clarity", "z_len_tokens", "z_len_chars"]

    for c in _available(df, cols):
        v = df[c].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue

        # Figure
        base = os.path.join(out_figures, f"hist_{c}_{split_label}_{disc_variant}")
        histogram(
            v,
            bins=int(bins),
            xlabel=c,
            ylabel="Count (samples)",
            base_path=base,
            dpi=dpi,
        )

        # Table (bin counts)
        counts, edges = np.histogram(v, bins=int(bins))
        tab = pd.DataFrame({"bin_lo": edges[:-1], "bin_hi": edges[1:], "count": counts})
        tab.to_csv(os.path.join(out_tables, f"hist_{c}_{split_label}_{disc_variant}.csv"), index=False)


def vif_table_and_plot(
    df: pd.DataFrame,
    *,
    out_tables: str,
    out_figures: str,
    disc_variant: str,
    split_label: str,
    dpi: int,
) -> None:
    """Variance inflation factors (VIF) on standardized predictors."""

    feats = _available(df, ["z_disc", "z_clarity", "z_len_tokens", "z_len_chars"])
    if len(feats) == 0:
        return

    # Add intercept column for VIF implementation compatibility.
    X = df[feats].astype(float).to_numpy()
    Xw = np.column_stack([np.ones(X.shape[0]), X])

    from statsmodels.stats.outliers_influence import variance_inflation_factor

    vif_vals: List[float] = []
    for j in range(1, Xw.shape[1]):
        vif_vals.append(float(variance_inflation_factor(Xw, j)))

    tab = pd.DataFrame({"feature": feats, "vif": vif_vals})
    tab.to_csv(os.path.join(out_tables, f"vif_{split_label}_{disc_variant}.csv"), index=False)

    bar_with_error(
        labels=tab["feature"].tolist(),
        values=tab["vif"].to_numpy(dtype=float),
        yerr=None,
        xlabel="Feature",
        ylabel="VIF (unitless)",
        base_path=os.path.join(out_figures, f"vif_{split_label}_{disc_variant}"),
        rotation=45,
        dpi=dpi,
    )


def split_half_stability(
    df: pd.DataFrame,
    *,
    out_tables: str,
    out_figures: str,
    disc_variant: str,
    split_label: str,
    seed: int,
    dpi: int,
    n_repeats: int = 10,
    features: Optional[Sequence[str]] = None,
) -> None:
    """Split-half coefficient stability (descriptive).

    This reproduces the original archive's appendix check: repeatedly split the
    *rows* at random into two halves, fit the same GLM on each half under the
    working i.i.d. likelihood, and plot the disc coefficient (bA vs bB).
    """

    if features is None:
        features = ["z_disc", "z_clarity", "z_len_tokens", "z_len_chars"]
    feats = _available(df, features)
    if "z_disc" not in feats:
        return

    y = df["y"].to_numpy(dtype=float)
    Xfull = sm.add_constant(df[feats].astype(float), has_constant="add")
    term = "z_disc"

    n = int(df.shape[0])
    if n < 10:
        return

    rng = np.random.default_rng(int(seed))

    bA: List[float] = []
    bB: List[float] = []
    for _ in range(int(n_repeats)):
        idx = rng.permutation(n)
        A = idx[: n // 2]
        B = idx[n // 2 :]
        resA = sm.GLM(y[A], Xfull.iloc[A], family=sm.families.Binomial()).fit()
        resB = sm.GLM(y[B], Xfull.iloc[B], family=sm.families.Binomial()).fit()
        bA.append(float(resA.params.get(term, np.nan)))
        bB.append(float(resB.params.get(term, np.nan)))

    tab = pd.DataFrame({"bA": bA, "bB": bB})
    tab.to_csv(os.path.join(out_tables, f"split_half_disc_coeff_{split_label}_{disc_variant}.csv"), index=False)

    scatter(
        bA,
        bB,
        xlabel="bA (disc coefficient)",
        ylabel="bB (disc coefficient)",
        base_path=os.path.join(out_figures, f"split_half_disc_coeff_{split_label}_{disc_variant}"),
        dpi=dpi,
    )


def plot_condition_intercepts_from_a5(
    coef_table: pd.DataFrame,
    *,
    out_figures: str,
    disc_variant: str,
    split_label: str,
    dpi: int,
    include_disc: bool = True,
) -> None:
    """Plot the condition-intercept terms (A5 proxy visualization).

    Uses the already-computed cluster-robust coefficient table from the A5 full model.
    """

    if coef_table.empty:
        return

    terms = coef_table["term"].astype(str)
    mask = terms.str.startswith("cond_")
    if include_disc:
        mask |= terms.eq("z_disc")

    sub = coef_table.loc[mask].copy()
    if sub.empty:
        return

    # 95% t-interval with df from coef_table (already set to G-1)
    from scipy.stats import t as student_t

    df_t = float(sub["df_t"].iloc[0]) if "df_t" in sub.columns else np.nan
    tcrit = float(student_t.ppf(0.975, df=max(int(df_t), 1))) if np.isfinite(df_t) else 1.96

    vals = sub["coef"].to_numpy(dtype=float)
    se = sub["se"].to_numpy(dtype=float)
    yerr = np.vstack([tcrit * se, tcrit * se])

    labels = [t.replace("cond_", "condition_") for t in sub["term"].astype(str).tolist()]

    bar_with_error(
        labels=labels,
        values=vals,
        yerr=yerr,
        xlabel="Term",
        ylabel="Coefficient (log-odds)",
        base_path=os.path.join(out_figures, f"condition_intercepts_{split_label}_{disc_variant}"),
        rotation=30,
        dpi=dpi,
        figsize=(6.2, 4.0),
    )
