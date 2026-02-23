# AI→RSA Pragmatic Softmax: Code + Outputs

This archive contains the **full Python code** and the **exact exported outputs** used in the paper:

> **From Softmax Correspondence to Identification: The Observation Gap and a Diagnostic Template**
> **One-Step Log-Score rEFE and RSA**
> Ryotaro Saito (Independent Researcher)

The project has two main parts:

1. **Empirical diagnostics on Colors in Context (CiC)**: outcome-level GLM diagnostics (A2_outcome/A5_outcome block tests),
   exhaustive robustness over discriminability (“disc”) variants, and a three-way Shapley attribution.
2. **A5 simulation calibration**: a controlled three-alternative reference-game simulation that
   enforces the A5 decomposition by construction and measures how the same outcome-level diagnostics behave.

---

## Repository layout

```
AI_to_RSA/
  README.md
  requirements.txt                # Minimal pinned requirements used by the pipeline
  LICENSE                         # MIT license

  ai_to_rsa/
    src/                          # Main CiC analysis pipeline (Python package)
    output/                       # ✅ Archived outputs used in the manuscript
      figures/                    # PNG (dpi=1200) + PDF figures
      tables/                     # CSV tables exported by the pipeline
      pip_freeze.txt              # Full environment snapshot (pip freeze)
      environment.json            # OS/Python + key package versions
      run_manifest.json           # Run parameters (seed, split sizes, etc.)

  Sim_ai_to_rsa/
    sim.py                        # A5 simulation script
    outputs/                      # ✅ Archived simulation outputs
      *.png, *.csv                # Regime-wise plots + tables
      pip_freeze.txt              # Full environment snapshot (pip freeze)
      environment.json            # OS/Python + key package versions
      run_manifest.json           # Run parameters + regime metadata
```

---

## Quick start (re-run the empirical CiC pipeline)

### 1) Create an environment and install dependencies

From the repository root:

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

> Tip: if you want to match the archived environment as closely as possible, install from the
> archived freeze file instead:
>
> ```bash
> pip install -ai_to_rsa/output/pip_freeze.txt
> ```

### 2) Obtain the CiC data (`filteredCorpus.csv`)

The pipeline expects the Colors in Context release table **`filteredCorpus.csv`**
(Monroe et al., 2017). The archive does **not** include the dataset.

Place `filteredCorpus.csv` somewhere convenient and point the pipeline to it, e.g.:

```
AI_to_RSA/
  filteredCorpus.csv
```

If the file is in (or under) the repository root, the pipeline can often auto-detect it.
Otherwise, pass `--input /path/to/filteredCorpus.csv`.

### 3) Run the pipeline

Run as a module (recommended because the code uses package-relative imports):

```bash
python -m ai_to_rsa.src.main --input ./filteredCorpus.csv --output ./ai_to_rsa/output --cluster worker+game+listener --shapley_bootstrap_method pigeonhole --disc_variants all --shapley_n_boot 300 --dpi 1200
```

Key CLI options:

- `--disc_variants all` runs the full multiverse of disc variants (recommended for the paper).
- `--no_shapley_bootstrap` disables the cluster bootstrap for Shapley ratios (faster, no error bars).
- `--shapley_n_boot 300` controls the number of bootstrap resamples (default: 300).
- `--dpi 1200` sets PNG export resolution.

---

## What the empirical pipeline produces (`ai_to_rsa/output`)

The output directory is organized as:

- `figures/`  
  - `condition_overview_accuracy_{train,test}.(png|pdf)`  
  - `assumption_suite_*` figures for A2_outcome and A5_outcome diagnostics (Wald + LR diagnostics)  
  - `condition_intercepts_*.(png|pdf)` (A5_outcome visualization)  
  - `shapley_ratio_*.(png|pdf)` (three-factor attribution)  
  - others (VIF plots, split-half plots, feature histograms)
- `tables/`  
  - `condition_overview_{train,test}.csv`  
  - `disc_definitions.csv` (catalog of all disc variants)  
  - `glm_coefficients_cluster_robust_{train,test}_*.csv`  
  - `a5_coefficients_cluster_robust_{train,test}_*.csv`  
  - `shapley_*.csv` (bootstraps + summary tables)
- Reproducibility bundle  
  - `pip_freeze.txt`, `environment.json`, `run_manifest.json`

**Important:** If you point `--output` to `ai_to_rsa/output`, you will overwrite the archived
manuscript outputs. Use a different directory name (e.g., `output_recomputed`) if you want to keep
the archive intact.

---

## A5 simulation calibration (`Sim_ai_to_rsa/`)

### Run the simulation

```bash
python Sim_ai_to_rsa/sim.py --outdir Sim_ai_to_rsa/outputs --seed 0 --n_trials 20000 --n_workers 600 --n_candidates 15 --test_frac_workers 0.2 --n_rep 20
```

### A2_outcome sensitivity sweep (violation calibration)

To probe how strongly A2_outcome slope-interaction rejections respond to different classes of violations (scale differences, ceiling compression, candidate set changes, context-dependent semantics, and a simple unobserved-confounding toy), run:

```bash
python Sim_ai_to_rsa/sim.py --mode violation_sweep --outdir Sim_ai_to_rsa/outputs_violation_sweep --seed 0 --n_trials 20000 --n_workers 600 --n_candidates 15 --n_rep 20
```

This writes per-violation CSV summaries (`sweep_*_reject_summary.csv`) and a
test-split line plot of A2_outcome rejection rates as the violation strength increases
(`sweep_*_a2_reject_test.png`).

The simulation script:

- Constructs a three-alternative reference game with internally generated candidate utterance sets.
- Defines the literal listener by Bayes’ rule, guaranteeing the A5 decomposition by construction.
- Collapses to **observed-only fields** (condition labels, geometry summaries, message length, worker IDs, success)
  to mirror the paper’s “observability gap”.
- Runs the same outcome-level block tests (A2_outcome/A5_outcome proxies) and exports regime-wise plots and CSV tables.
- Writes `pip_freeze.txt`, `environment.json`, and `run_manifest.json` into the output directory.

### Simulation outputs

In each regime the script exports:

- `*_condition_success.(png|csv)` : condition-wise success with Wilson 95% intervals
- `*_rejection_rates.(png|csv)`   : rejection-rate summaries across repetitions (with Wilson intervals)
- `*_wald_stats_example_test.(png|csv)` : an example run’s Wald statistics (test split)
- `*_reject_raw.csv`, `*_reject_summary.csv` : raw + summarized repetition results
- `all_regimes_reject_summary.csv` : combined summary across regimes

---

## Reproducibility records

Both the empirical pipeline and the simulation write three reproducibility artifacts:

- `pip_freeze.txt` — the exact Python package environment (`pip freeze`)
- `environment.json` — OS / Python version / key package versions / CPU count
- `run_manifest.json` — run-time configuration (seeds, split sizes, and other core parameters)

These files are also included in the archived outputs shipped with this ZIP.

---

## License

MIT License (see `LICENSE`).

---

## Contact

Ryotaro Saito  
ryotarosaito136@gmail.com
