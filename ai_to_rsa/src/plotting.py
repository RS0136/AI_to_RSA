from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Headless plotting (CI-friendly)
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


def set_grayscale_style() -> None:
    """Apply a simple grayscale, title-free plotting style."""

    # Grayscale defaults (explicitly set to avoid environment-dependent themes)
    matplotlib.rcParams["image.cmap"] = "Greys"
    matplotlib.rcParams["text.color"] = "0.0"
    matplotlib.rcParams["axes.labelcolor"] = "0.0"
    matplotlib.rcParams["xtick.color"] = "0.0"
    matplotlib.rcParams["ytick.color"] = "0.0"
    matplotlib.rcParams["grid.color"] = "0.5"
    matplotlib.rcParams["axes.edgecolor"] = "0.2"


def save_figure(fig: plt.Figure, base_path: str, *, dpi: int = 600) -> None:
    """Save figure as high-res PNG + vector PDF.

    Parameters
    ----------
    base_path:
        File path without extension.
    dpi:
        PNG resolution.
    """

    Path(base_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base_path + ".png", dpi=dpi, bbox_inches="tight")
    fig.savefig(base_path + ".pdf", bbox_inches="tight")
    plt.close(fig)


def bar_with_error(
    labels: Sequence[str],
    values: Sequence[float],
    yerr: Optional[np.ndarray],
    *,
    xlabel: str,
    ylabel: str,
    base_path: str,
    rotation: float = 0.0,
    figsize: Tuple[float, float] = (5.5, 4.0),
    dpi: int = 800,
) -> None:
    """Single bar chart (grayscale), optional symmetric/asymmetric error bars."""

    set_grayscale_style()

    vals = np.asarray(values, dtype=float)
    idx = np.arange(len(vals))

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(idx, vals, color="0.25", edgecolor="0.25")

    if yerr is not None:
        ax.errorbar(
            idx,
            vals,
            yerr=yerr,
            fmt="none",
            ecolor="0.0",
            elinewidth=1.0,
            capsize=3,
            capthick=1.0,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(idx)
    ax.set_xticklabels(list(labels), rotation=rotation, ha="right" if rotation else "center")

    ax.grid(axis="y", alpha=0.3)

    save_figure(fig, base_path, dpi=dpi)

def bar_with_error_allow_na(
    labels: Sequence[str],
    values: Sequence[float],
    yerr: Optional[np.ndarray],
    *,
    xlabel: str,
    ylabel: str,
    base_path: str,
    rotation: float = 0.0,
    figsize: Tuple[float, float] = (5.5, 4.0),
    dpi: int = 800,
    na_label: str = "NA",
) -> None:
    """Single bar chart that gracefully handles NaN values.

    Any non-finite value is plotted as 0.0 and annotated with `na_label` above the bar.
    This is useful for robustness-multiverse plots where some variants are marked not testable.
    """

    set_grayscale_style()

    vals = np.asarray(values, dtype=float)
    idx = np.arange(len(vals))
    is_na = ~np.isfinite(vals)

    plot_vals = vals.copy()
    plot_vals[is_na] = 0.0

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(idx, plot_vals, color="0.25", edgecolor="0.25")

    if yerr is not None:
        ax.errorbar(
            idx,
            plot_vals,
            yerr=yerr,
            fmt="none",
            ecolor="0.0",
            elinewidth=1.0,
            capsize=3,
            capthick=1.0,
        )

    # Annotate NA bars (keep tiny offset so text is visible even when plotted value is 0)
    for i, na in enumerate(is_na.tolist()):
        if na:
            ax.text(i, 0.0, na_label, ha="center", va="bottom", fontsize=8, rotation=90)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(idx)
    ax.set_xticklabels(list(labels), rotation=rotation, ha="right" if rotation else "center")

    ax.grid(axis="y", alpha=0.3)

    save_figure(fig, base_path, dpi=dpi)



def scatter(
    x: Sequence[float],
    y: Sequence[float],
    *,
    xlabel: str,
    ylabel: str,
    base_path: str,
    figsize: Tuple[float, float] = (5.0, 4.0),
    dpi: int = 800,
) -> None:
    set_grayscale_style()
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(np.asarray(x, dtype=float), np.asarray(y, dtype=float), s=18, color="0.25")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    save_figure(fig, base_path, dpi=dpi)


def histogram(
    values: Sequence[float],
    *,
    bins: int,
    xlabel: str,
    ylabel: str,
    base_path: str,
    figsize: Tuple[float, float] = (5.0, 4.0),
    dpi: int = 800,
) -> None:
    """Histogram helper (grayscale, no title)."""

    set_grayscale_style()
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(v, bins=bins, color="0.25", edgecolor="0.25")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    save_figure(fig, base_path, dpi=dpi)
