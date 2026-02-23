from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Split:
    """Train/test split container."""

    train: pd.DataFrame
    test: pd.DataFrame


def group_shuffle_split(
    df: pd.DataFrame,
    group_col: str,
    test_size: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Worker-disjoint shuffle split.

    Parameters
    ----------
    df:
        Full dataset.
    group_col:
        Column containing group IDs (e.g., worker IDs). All rows from a group are
        assigned to the same split.
    test_size:
        Fraction of groups assigned to the test set.
    seed:
        RNG seed.

    Returns
    -------
    train_idx, test_idx:
        Row indices for train/test.
    """

    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be in (0,1).")

    groups = df[group_col].astype(str).to_numpy()
    uniq = np.unique(groups)  # sorted, stable

    rng = np.random.RandomState(seed)
    perm = rng.permutation(uniq)

    n_test = int(np.ceil(test_size * len(perm)))
    test_groups = set(perm[:n_test].tolist())

    is_test = pd.Series(groups).isin(test_groups).to_numpy(bool)
    test_idx = np.nonzero(is_test)[0]
    train_idx = np.nonzero(~is_test)[0]

    if len(train_idx) == 0 or len(test_idx) == 0:
        raise RuntimeError("Split produced empty train or test set. Adjust test_size or seed.")

    return train_idx, test_idx


def make_split(df: pd.DataFrame, *, group_col: str = "worker", test_size: float = 0.20, seed: int = 0) -> Split:
    train_idx, test_idx = group_shuffle_split(df, group_col=group_col, test_size=test_size, seed=seed)
    train = df.iloc[train_idx].reset_index(drop=True)
    test = df.iloc[test_idx].reset_index(drop=True)
    return Split(train=train, test=test)
