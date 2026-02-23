from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm


def condition_dummies(
    df: pd.DataFrame,
    *,
    col: str = "condition",
    categories: Optional[Sequence[str]] = None,
    prefix: str = "cond",
) -> pd.DataFrame:
    """Create condition intercept dummies with stable columns.

    We enforce a shared set of categories so that train/test dummies align.
    """

    if categories is None:
        categories = sorted(df[col].astype(str).unique().tolist())

    cat = pd.Categorical(df[col].astype(str), categories=list(categories), ordered=False)
    dum = pd.get_dummies(cat, prefix=prefix, drop_first=True, dtype=float)

    # Ensure stable ordering
    dum = dum.reindex(sorted(dum.columns), axis=1)
    return dum


def interactions_with_series(
    dum: pd.DataFrame,
    z: pd.Series,
    name: str,
    *,
    dummy_prefix: str = "cond_",
) -> pd.DataFrame:
    """Create an interaction block: z × condition_dummies.

    Column naming follows the existing convention used in the manuscript code:
        {name}_x_{condition}

    where `condition` is the dummy column name with the `dummy_prefix` stripped.
    """
    out = dum.mul(z.to_numpy(dtype=float), axis=0)

    def _strip(c: str) -> str:
        return c.replace(dummy_prefix, "") if c.startswith(dummy_prefix) else c

    out.columns = [f"{name}_x_{_strip(c)}" for c in out.columns]
    return out


def interactions_with_disc(dum: pd.DataFrame, disc_z: pd.Series, disc_name: str) -> pd.DataFrame:
    """Interaction block: disc_z × condition_dummies (backwards compatible wrapper)."""
    return interactions_with_series(dum, disc_z, disc_name)


def add_const(X: pd.DataFrame) -> pd.DataFrame:
    return sm.add_constant(X, has_constant="add").astype(float)


@dataclass(frozen=True)
class Design:
    X: pd.DataFrame
    condition_categories: List[str]
