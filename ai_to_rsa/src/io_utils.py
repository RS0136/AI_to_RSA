from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


def ensure_dir(path: str | os.PathLike) -> str:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


def save_df(df: pd.DataFrame, out_path: str, *, index: bool = False) -> str:
    ensure_dir(Path(out_path).parent)
    df.to_csv(out_path, index=index)
    return out_path


def save_json(obj: Dict[str, Any], out_path: str) -> str:
    ensure_dir(Path(out_path).parent)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return out_path


def save_text(text: str, out_path: str) -> str:
    """Save a UTF-8 text file."""
    ensure_dir(Path(out_path).parent)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return out_path


def collect_environment_info(packages: Optional[list[str]] = None) -> Dict[str, Any]:
    """Collect a lightweight reproducibility record (system + key package versions)."""
    import datetime
    import platform
    import sys
    import os

    try:
        from importlib import metadata as importlib_metadata  # py3.8+
    except Exception:  # pragma: no cover
        import importlib_metadata  # type: ignore

    if packages is None:
        # Keep this short and directly relevant to the pipeline.
        packages = [
            "numpy",
            "pandas",
            "statsmodels",
            "scipy",
            "matplotlib",
        ]

    pkg_versions: Dict[str, Any] = {}
    for pkg in packages:
        try:
            pkg_versions[pkg] = importlib_metadata.version(pkg)
        except Exception:
            pkg_versions[pkg] = None

    info: Dict[str, Any] = {
        "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "platform": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_build": platform.python_build(),
            "python_compiler": platform.python_compiler(),
        },
        "cpu_count": os.cpu_count(),
        "packages": pkg_versions,
    }
    return info


def pip_freeze() -> str:
    """Return `pip freeze` output as a string (best-effort)."""
    import subprocess
    import sys

    try:
        out = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        return out
    except Exception as e:
        return f"# pip freeze failed: {e}\n"

def resolve_input_csv(input_csv: Optional[str], default_name: str = "filteredCorpus.csv") -> str:
    """Resolve the input CSV path.

    If `input_csv` is provided, it is used as-is.
    Otherwise the function searches the current directory tree for a file named
    `default_name`.
    """

    if input_csv is not None:
        if not os.path.isfile(input_csv):
            raise FileNotFoundError(f"Input CSV not found: {input_csv}")
        return input_csv

    # Common candidates
    candidates = [default_name, os.path.join(".", default_name), os.path.join("..", default_name)]
    for c in candidates:
        if os.path.isfile(c):
            return c

    # Full tree scan (last resort, but deterministic)
    for root, _, files in os.walk("."):
        if default_name in files:
            return os.path.join(root, default_name)

    raise FileNotFoundError(
        f"Input CSV '{default_name}' not found. Pass --input /path/to/{default_name}."
    )
