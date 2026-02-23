from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class CiCData:
    """Container for the parsed CiC table.

    The pipeline keeps raw geometry fields and derives analysis features later
    (disc variants, standardization, dummy coding) to avoid accidental leakage
    across splits/variants.
    """

    df: pd.DataFrame


def coerce_numeric(x) -> pd.Series:
    if isinstance(x, pd.Series):
        return pd.to_numeric(x, errors="coerce")
    return pd.to_numeric(pd.Series(x), errors="coerce")


def build_binary_outcome(raw: pd.DataFrame) -> pd.Series:
    """Construct binary success outcome y.

    The CiC filteredCorpus.csv has historically used fields like numOutcome or clickStatus.
    We support a small set of common variants.
    """

    if "numOutcome" in raw.columns:
        return (coerce_numeric(raw["numOutcome"]).fillna(0) > 0).astype(int)

    if "outcome" in raw.columns:
        v = raw["outcome"]
        if np.issubdtype(v.dtype, np.number):
            return (coerce_numeric(v).fillna(0) > 0).astype(int)
        # String conventions (be conservative; treat unknown as 0)
        return (
            v.astype(str)
            .str.strip()
            .str.lower()
            .isin(["1", "true", "t", "y", "yes", "correct", "win", "won", "success", "target"])
            .astype(int)
        )

    if "clickStatus" in raw.columns:
        return (
            raw["clickStatus"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(["1", "true", "t", "y", "yes", "target", "correct"])
            .astype(int)
        )

    raise ValueError("Cannot construct binary outcome y: expected numOutcome/outcome/clickStatus.")


def load_cic_csv(path: str) -> CiCData:
    """Load filteredCorpus.csv and extract the fields needed for analysis."""

    raw = pd.read_csv(path)

    y = build_binary_outcome(raw)

    # Geometry fields
    t1 = coerce_numeric(raw.get("targetD1Diff", np.nan))
    t2 = coerce_numeric(raw.get("targetD2Diff", np.nan))
    d12 = coerce_numeric(raw.get("D1D2Diff", np.nan))

    # Clarity (Eq. 10): mean absolute target–distractor separation
    clarity = 0.5 * (t1.abs() + t2.abs())

    # Length proxies
    len_tokens = coerce_numeric(raw.get("numCleanWords", raw.get("numRawWords", np.nan)))
    len_chars = coerce_numeric(raw.get("numCleanChars", raw.get("numRawChars", np.nan)))

    condition = raw.get("condition", "unknown").astype(str)

    # Speaker / producer worker id ("speaker" in the paper text)
    worker = raw.get("workerid_uniq", raw.get("workerid", "worker-unknown")).astype(str)

    # Optional higher-level clustering units.
    #
    # Review-oriented sensitivity analyses commonly request alternative
    # clustering (e.g., by "game" / context ID rather than worker ID).
    # The CiC releases have not always used the same field name, so we try a
    # small set of common conventions and fall back to a deterministic
    # placeholder if none are present.
    game = (
        raw.get("gameid_uniq", None)
        if "gameid_uniq" in raw.columns
        else raw.get("gameid", None)
        if "gameid" in raw.columns
        else raw.get("game_id", None)
        if "game_id" in raw.columns
        else raw.get("game", None)
        if "game" in raw.columns
        else None
    )
    if game is None:
        game = pd.Series(["game-unknown"] * len(raw))
    game = game.astype(str)

    # Optional listener / receiver id.
    #
    # Different CiC releases / preprocessed tables have used different naming
    # conventions; we try a small set of common candidates. When absent, we
    # fall back to a deterministic placeholder so downstream code can still
    # accept a 3-way spec (it will effectively reduce to lower-way clustering).
    listener = (
        raw.get("listenerid_uniq", None)
        if "listenerid_uniq" in raw.columns
        else raw.get("listenerid", None)
        if "listenerid" in raw.columns
        else raw.get("listener_id", None)
        if "listener_id" in raw.columns
        else raw.get("listener", None)
        if "listener" in raw.columns
        else raw.get("receiverid_uniq", None)
        if "receiverid_uniq" in raw.columns
        else raw.get("receiverid", None)
        if "receiverid" in raw.columns
        else raw.get("receiver_id", None)
        if "receiver_id" in raw.columns
        else None
    )
    if listener is None:
        listener = pd.Series(["listener-unknown"] * len(raw))
    listener = listener.astype(str)

    # NOTE: keep `d12` in the returned frame.
    # Some robustness checks (e.g., the `d12_only` disc variant) require it.
    df = pd.DataFrame(
        {
            "y": y.astype(int),
            "t1": t1,
            "t2": t2,
            "d12": d12,
            "clarity": clarity,
            "len_tokens": len_tokens,
            "len_chars": len_chars,
            "condition": condition,
            "worker": worker,
            "game": game,
            "listener": listener,
        }
    )

    # Basic missingness filter: ensure y and base fields exist.
    # We do NOT filter on disc variants here (variant-specific later).
    # We do *not* require d12 to be finite here; disc variants will filter as needed.
    base_cols = [
        "y",
        "t1",
        "t2",
        "d12",
        "clarity",
        "len_tokens",
        "len_chars",
        "condition",
        "worker",
        "game",
        "listener",
    ]
    mask = np.isfinite(df["y"].to_numpy(dtype=float))
    for c in ["t1", "t2", "clarity", "len_tokens", "len_chars"]:
        mask &= np.isfinite(df[c].to_numpy(dtype=float))
    df = df.loc[mask, base_cols].reset_index(drop=True)

    return CiCData(df=df)
