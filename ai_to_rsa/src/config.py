from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DiscDefinition:
    """A discriminability (disc) definition.
    """

    name: str
    func: Callable[[pd.Series, pd.Series, pd.Series], pd.Series]
    description: str


def _safe_abs(x: pd.Series) -> pd.Series:
    return x.abs() if isinstance(x, pd.Series) else pd.Series(x).abs()


def _eps_like(x: pd.Series, eps: float = 1e-9) -> pd.Series:
    # Use eps scaled to typical magnitude to reduce the chance of division by zero.
    # We keep it deterministic and very small.
    m = float(np.nanmedian(_safe_abs(x).to_numpy(dtype=float)))
    return pd.Series(np.full(len(x), eps * (1.0 + m), dtype=float), index=x.index)


def disc_definitions() -> Dict[str, DiscDefinition]:
    """Return a dictionary of disc variants.

    The list below intentionally includes more than one plausible operationalization.
    The goal is not to claim any one is "the" correct definition, but to verify that
    the A2_outcome/A5_outcome diagnostics and related conclusions are not an artifact of a single
    coding choice.

    The definitions are designed to be computable from the CiC fields.
    """

    # NOTE (rev2 / journal-ready):
    # The manuscript's *main analysis* discriminability proxy is now the
    # "nearest competitor" difficulty index, because it is the most directly
    # interpretable geometry summary in a 3A reference game:
    #   larger (i.e., less negative) values imply a closer nearest distractor.
    #
    # The prior manuscript's Eq.(9) coding ("D1D2Diff if available else t1-t2")
    # is retained as a *legacy* robustness variant under a new name.

    def legacy_eq9_signed(t1: pd.Series, t2: pd.Series, d12: pd.Series) -> pd.Series:
        # Legacy Eq. (9): prefer D1D2Diff when present; otherwise fall back.
        out = d12.copy()
        out = out.where(out.notna(), t1 - t2)
        return out

    def tdiff_raw(t1: pd.Series, t2: pd.Series, d12: pd.Series) -> pd.Series:
        # Direct target difference (signed).
        return t1 - t2

    def tdiff_abs(t1: pd.Series, t2: pd.Series, d12: pd.Series) -> pd.Series:
        # Absolute-distance version of the target difference.
        return _safe_abs(t1) - _safe_abs(t2)

    def d12_only(t1: pd.Series, t2: pd.Series, d12: pd.Series) -> pd.Series:
        # Use D1D2Diff only (drops rows where missing).
        return d12

    def d12_abs(t1: pd.Series, t2: pd.Series, d12: pd.Series) -> pd.Series:
        # Absolute D1D2Diff (drops rows where missing).
        return _safe_abs(d12)

    def nearest_competitor(t1: pd.Series, t2: pd.Series, d12: pd.Series) -> pd.Series:
        # Nearest competitor distance: min(|t1|, |t2|).
        # Smaller means closer competitor (harder). We return a *difficulty* proxy:
        #   difficulty = -distance, so larger means harder.
        d1 = _safe_abs(t1)
        d2 = _safe_abs(t2)
        nearest = pd.concat([d1, d2], axis=1).min(axis=1)
        return -nearest

    def mean_competitor(t1: pd.Series, t2: pd.Series, d12: pd.Series) -> pd.Series:
        # Mean target–distractor distance (clarity-like). Return as *difficulty* by negation.
        d1 = _safe_abs(t1)
        d2 = _safe_abs(t2)
        return -(0.5 * (d1 + d2))

    def inv_mean_competitor(t1: pd.Series, t2: pd.Series, d12: pd.Series) -> pd.Series:
        # Inverse-mean distance: 1/(mean(|t1|,|t2|)+eps).
        d1 = _safe_abs(t1)
        d2 = _safe_abs(t2)
        mean = 0.5 * (d1 + d2)
        eps = _eps_like(mean)
        return 1.0 / (mean + eps)

    def nearest_competitor_signed(t1: pd.Series, t2: pd.Series, d12: pd.Series) -> pd.Series:
        # Keep sign information while focusing on the nearest competitor magnitude.
        # sign = sign(t1 - t2). magnitude = min(|t1|,|t2|)
        # The sign is arbitrary w.r.t. label permutation but can reveal coding artifacts.
        d1 = _safe_abs(t1)
        d2 = _safe_abs(t2)
        nearest = pd.concat([d1, d2], axis=1).min(axis=1)
        sign = np.sign((t1 - t2).to_numpy(dtype=float))
        return pd.Series(sign, index=t1.index) * nearest

    def inv_nearest_competitor(t1: pd.Series, t2: pd.Series, d12: pd.Series) -> pd.Series:
        # Inverse-nearest distance: 1/(min(|t1|,|t2|)+eps).
        # Larger means closer nearest competitor (harder).
        d1 = _safe_abs(t1)
        d2 = _safe_abs(t2)
        nearest = pd.concat([d1, d2], axis=1).min(axis=1)
        eps = _eps_like(nearest)
        return 1.0 / (nearest + eps)

    def neglog_nearest_competitor(t1: pd.Series, t2: pd.Series, d12: pd.Series) -> pd.Series:
        # -log(min(|t1|,|t2|)+eps). Larger means closer nearest competitor (harder).
        d1 = _safe_abs(t1)
        d2 = _safe_abs(t2)
        nearest = pd.concat([d1, d2], axis=1).min(axis=1)
        eps = _eps_like(nearest)
        return -np.log(nearest + eps)

    def farthest_distractor(t1: pd.Series, t2: pd.Series, d12: pd.Series) -> pd.Series:
        # Farthest distractor distance: max(|t1|, |t2|). Larger means easier.
        # Return as *ease* proxy (no sign flip).
        d1 = _safe_abs(t1)
        d2 = _safe_abs(t2)
        return pd.concat([d1, d2], axis=1).max(axis=1)

    def gap_near_far(t1: pd.Series, t2: pd.Series, d12: pd.Series) -> pd.Series:
        # Asymmetry between distractors: max(|t1|, |t2|) - min(|t1|, |t2|).
        d1 = _safe_abs(t1)
        d2 = _safe_abs(t2)
        return pd.concat([d1, d2], axis=1).max(axis=1) - pd.concat([d1, d2], axis=1).min(axis=1)

    def gap_norm(t1: pd.Series, t2: pd.Series, d12: pd.Series) -> pd.Series:
        # Normalized asymmetry: (max-min) / (mean+eps)
        d1 = _safe_abs(t1)
        d2 = _safe_abs(t2)
        mx = pd.concat([d1, d2], axis=1).max(axis=1)
        mn = pd.concat([d1, d2], axis=1).min(axis=1)
        mean = 0.5 * (d1 + d2)
        eps = _eps_like(mean)
        return (mx - mn) / (mean + eps)

    def ratio_near_far(t1: pd.Series, t2: pd.Series, d12: pd.Series) -> pd.Series:
        # Relative closeness: min/(max+eps) in [0,1]. Smaller implies stronger asymmetry.
        d1 = _safe_abs(t1)
        d2 = _safe_abs(t2)
        mx = pd.concat([d1, d2], axis=1).max(axis=1)
        mn = pd.concat([d1, d2], axis=1).min(axis=1)
        eps = _eps_like(mx)
        return mn / (mx + eps)

    def log_ratio_near_far(t1: pd.Series, t2: pd.Series, d12: pd.Series) -> pd.Series:
        # log((min+eps)/(max+eps)) <= 0. Values closer to 0 indicate more symmetric.
        d1 = _safe_abs(t1)
        d2 = _safe_abs(t2)
        mx = pd.concat([d1, d2], axis=1).max(axis=1)
        mn = pd.concat([d1, d2], axis=1).min(axis=1)
        eps = _eps_like(mx)
        return np.log((mn + eps) / (mx + eps))

    def softmin_dist(t1: pd.Series, t2: pd.Series, d12: pd.Series) -> pd.Series:
        # Smooth minimum of distances: -(1/k) log(exp(-k d1) + exp(-k d2)).
        # We return a difficulty proxy by negating the softmin (closer -> larger).
        d1 = _safe_abs(t1).astype(float)
        d2 = _safe_abs(t2).astype(float)
        k = 1.0
        m = np.minimum(d1, d2)
        # stable logsumexp: -1/k * ( -k*m + log(exp(-k(d1-m))+exp(-k(d2-m))) )
        # => softmin = m - (1/k) log(exp(-k(d1-m))+exp(-k(d2-m)))
        z = np.exp(-k * (d1 - m)) + np.exp(-k * (d2 - m))
        soft = m - (1.0 / k) * np.log(z)
        return -pd.Series(soft, index=t1.index)

    def d12_over_nearest(t1: pd.Series, t2: pd.Series, d12: pd.Series) -> pd.Series:
        # How separated the distractors are relative to the nearest competitor distance.
        # Large values indicate distractors are far apart even when one is near.
        d1 = _safe_abs(t1)
        d2 = _safe_abs(t2)
        nearest = pd.concat([d1, d2], axis=1).min(axis=1)
        eps = _eps_like(nearest)
        return _safe_abs(d12) / (nearest + eps)

    defs = {
        # "original" is the *main* disc definition used for the primary analysis.
        # (See the NOTE above.)
        "original": DiscDefinition(
            name="original",
            func=nearest_competitor,
            description="Main (nearest competitor) difficulty proxy: -min(|t1|,|t2|).",
        ),
        "legacy_eq9_signed": DiscDefinition(
            name="legacy_eq9_signed",
            func=legacy_eq9_signed,
            description="Legacy (prior manuscript 'original'): D1D2Diff if available else targetD1Diff-targetD2Diff (signed).",
        ),
        "tdiff_raw": DiscDefinition(
            name="tdiff_raw",
            func=tdiff_raw,
            description="Signed target difference: targetD1Diff-targetD2Diff.",
        ),
        "tdiff_abs": DiscDefinition(
            name="tdiff_abs",
            func=tdiff_abs,
            description="Absolute-distance target difference: |targetD1Diff|-|targetD2Diff|.",
        ),
        "d12_only": DiscDefinition(
            name="d12_only",
            func=d12_only,
            description="D1D2Diff only (drops rows where missing).",
        ),
        "d12_abs": DiscDefinition(
            name="d12_abs",
            func=d12_abs,
            description="Absolute D1D2Diff only (drops rows where missing).",
        ),
        "nearest_competitor": DiscDefinition(
            name="nearest_competitor",
            func=nearest_competitor,
            description="Difficulty proxy from nearest competitor: -min(|t1|,|t2|).",
        ),
        "mean_competitor": DiscDefinition(
            name="mean_competitor",
            func=mean_competitor,
            description="Difficulty proxy from mean competitor distance: -0.5(|t1|+|t2|).",
        ),
        "inv_mean_competitor": DiscDefinition(
            name="inv_mean_competitor",
            func=inv_mean_competitor,
            description="Difficulty proxy: 1/(0.5(|t1|+|t2|)+eps).",
        ),
        "nearest_competitor_signed": DiscDefinition(
            name="nearest_competitor_signed",
            func=nearest_competitor_signed,
            description="Sign(t1-t2) * min(|t1|,|t2|) (signed nearest-competitor magnitude).",
        ),
        "inv_nearest_competitor": DiscDefinition(
            name="inv_nearest_competitor",
            func=inv_nearest_competitor,
            description="Difficulty proxy: 1/(min(|t1|,|t2|)+eps).",
        ),
        "neglog_nearest_competitor": DiscDefinition(
            name="neglog_nearest_competitor",
            func=neglog_nearest_competitor,
            description="Difficulty proxy: -log(min(|t1|,|t2|)+eps).",
        ),
        "farthest_distractor": DiscDefinition(
            name="farthest_distractor",
            func=farthest_distractor,
            description="Ease proxy from farthest distractor: max(|t1|,|t2|).",
        ),
        "gap_near_far": DiscDefinition(
            name="gap_near_far",
            func=gap_near_far,
            description="Asymmetry: max(|t1|,|t2|)-min(|t1|,|t2|).",
        ),
        "gap_norm": DiscDefinition(
            name="gap_norm",
            func=gap_norm,
            description="Normalized asymmetry: (max-min)/(mean+eps).",
        ),
        "ratio_near_far": DiscDefinition(
            name="ratio_near_far",
            func=ratio_near_far,
            description="Relative closeness: min/(max+eps) in [0,1].",
        ),
        "log_ratio_near_far": DiscDefinition(
            name="log_ratio_near_far",
            func=log_ratio_near_far,
            description="Log ratio: log((min+eps)/(max+eps)).",
        ),
        "softmin_dist": DiscDefinition(
            name="softmin_dist",
            func=softmin_dist,
            description="Smooth-min difficulty proxy: -softmin(|t1|,|t2|).",
        ),
        "d12_over_nearest": DiscDefinition(
            name="d12_over_nearest",
            func=d12_over_nearest,
            description="Relative distractor separation: |d12|/(min(|t1|,|t2|)+eps).",
        ),
    }

    return defs


# Default groups for the three-way decomposition.
TRI_GROUPS: Tuple[str, str, str] = ("offsets", "geometry", "decoder")
