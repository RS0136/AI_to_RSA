from __future__ import annotations

import argparse

from .io_utils import resolve_input_csv
from .pipeline import run_pipeline


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cic_diagnostics",
        description=(
            "CiC diagnostic pipeline (train/test, cluster-robust block Wald, disc robustness, "
            "and three-way Shapley decomposition)."
        ),
    )

    p.add_argument("--input", type=str, default=None, help="Path to filteredCorpus.csv (auto-search if omitted)")
    p.add_argument("--output", type=str, default="./output", help="Output directory")

    p.add_argument("--seed", type=int, default=0, help="Random seed (split + bootstrap)")
    p.add_argument("--test_size", type=float, default=0.20, help="Fraction of workers assigned to test")

    p.add_argument(
        "--cluster",
        type=str,
        default="worker",
        help=(
            "Clustering specification for robust inference (one-way or multiway). "
            "Examples: 'worker', 'worker+game', 'worker+game+listener'."
        ),
    )

    p.add_argument(
        "--shapley_bootstrap_method",
        type=str,
        default="cluster",
        choices=["cluster", "intersection", "pigeonhole"],
        help=(
            "Bootstrap method for Shapley uncertainty. "
            "'cluster' resamples the first cluster in --cluster; "
            "'intersection' resamples intersection clusters (e.g., worker×game); "
            "'pigeonhole' uses multiway reweighting." 
        ),
    )

    p.add_argument(
        "--disc_variants",
        type=str,
        default="all",
        help=(
            "Comma-separated disc variants to run, or 'all'. "
            "See tables/disc_definitions.csv after a run for the catalog."
        ),
    )

    p.add_argument(
        "--no_shapley_bootstrap",
        action="store_true",
        help="Disable cluster bootstrap for Shapley ratios (faster, but no error bars).",
    )
    p.add_argument("--shapley_n_boot", type=int, default=300, help="Number of bootstrap draws (clusters) for Shapley")

    p.add_argument("--dpi", type=int, default=800, help="PNG output resolution")

    # Optional: wild cluster bootstrap for block Wald tests (A2_outcome/A5_outcome).
    # Default is 'none' to preserve runtime and to match archived manuscript outputs.
    p.add_argument(
        "--wald_bootstrap",
        type=str,
        default="none",
        choices=["none", "wild"],
        help="Bootstrap method for block Wald tests: 'none' (default) or 'wild' (wild cluster bootstrap).",
    )
    p.add_argument("--wald_n_boot", type=int, default=999, help="Number of wild bootstrap draws for Wald block tests")
    p.add_argument(
        "--wald_bootstrap_weight",
        type=str,
        default="rademacher",
        choices=["rademacher", "webb"],
        help="Wild bootstrap weight distribution (Rademacher or Webb 6-point).",
    )

    return p


def main(argv=None):
    p = build_argparser()
    args = p.parse_args(argv)

    input_csv = resolve_input_csv(args.input)

    if args.disc_variants.strip().lower() == "all":
        disc_variants = None
    else:
        disc_variants = [s.strip() for s in args.disc_variants.split(",") if s.strip()]

    run_pipeline(
        input_csv=input_csv,
        output_dir=args.output,
        seed=args.seed,
        test_size=args.test_size,
        disc_variants=disc_variants,
        shapley_bootstrap=not args.no_shapley_bootstrap,
        shapley_n_boot=args.shapley_n_boot,
        cluster_spec=args.cluster,
        shapley_bootstrap_method=args.shapley_bootstrap_method,
        wald_bootstrap=args.wald_bootstrap,
        wald_n_boot=args.wald_n_boot,
        wald_bootstrap_weight=args.wald_bootstrap_weight,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
