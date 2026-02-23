from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Standardizer:
    """Train-only standardizer for a fixed set of columns."""

    cols: List[str]
    mean_: pd.Series
    std_: pd.Series

    def transform(self, df: pd.DataFrame, prefix: str = "z_") -> pd.DataFrame:
        out = df.copy()
        for c in self.cols:
            mu = float(self.mean_[c])
            sd = float(self.std_[c])
            if not np.isfinite(sd) or sd == 0.0:
                sd = 1.0
            out[f"{prefix}{c}"] = (df[c] - mu) / sd
        return out


def fit_standardizer(train: pd.DataFrame, cols: Iterable[str]) -> Standardizer:
    cols = list(cols)
    mean_ = train[cols].mean(axis=0)
    std_ = train[cols].std(axis=0, ddof=0)
    std_ = std_.replace(0.0, 1.0)
    return Standardizer(cols=cols, mean_=mean_, std_=std_)
