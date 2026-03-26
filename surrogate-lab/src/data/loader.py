"""Load parsed xplt simulation data from CSV files."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.schema import validate
from src.utils.logging_utils import get_logger

log = get_logger(__name__)


def _add_insertion_depth(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Convert time_step → insertion_depth if not already present."""
    if "insertion_depth" in df.columns:
        return df
    if "time_step" not in df.columns:
        log.warning("Neither 'insertion_depth' nor 'time_step' found — skipping conversion.")
        return df

    method = cfg["data"]["time_to_depth"]["method"]
    scale = float(cfg["data"]["time_to_depth"]["scale"])

    df = df.copy()
    if method == "linear":
        df["insertion_depth"] = df["time_step"] * scale
    else:
        raise ValueError(f"Unknown time_to_depth method: '{method}'")

    log.info("Derived insertion_depth from time_step (method=%s, scale=%.4f)", method, scale)
    return df


def list_available_features(path: str) -> dict[str, list[str]]:
    """Inspect a surrogate CSV and report which columns are available.

    Useful when xplt-parser adds new variables and you want to know which
    column names can be added to configs/config.yaml.

    Args:
        path: Path to a surrogate CSV file (e.g. ``sample_surrogate.csv``).

    Returns:
        A dict with keys ``'all_columns'``, ``'suggested_inputs'``, and
        ``'suggested_target'`` based on the conventional naming used by
        xplt-parser.  Columns ending in ``_pressure`` or ``_force`` are
        treated as candidate targets; everything else as inputs.

    Example::

        from src.data.loader import list_available_features
        info = list_available_features("path/to/sample_surrogate.csv")
        print("Add to config.yaml inputs:", info["suggested_inputs"])
    """
    df = pd.read_csv(path, nrows=0)   # header only — no data read
    cols = list(df.columns)

    target_hints = {"pressure", "force", "stress", "strain", "traction"}
    suggested_targets = [c for c in cols if any(h in c.lower() for h in target_hints)]
    suggested_inputs  = [c for c in cols if c not in suggested_targets]

    log.info("Available columns in %s: %s", Path(path).name, cols)
    return {
        "all_columns":       cols,
        "suggested_inputs":  suggested_inputs,
        "suggested_target":  suggested_targets,
    }


def load_simulation_data(
    cfg: dict,
    path: str | None = None,
) -> pd.DataFrame:
    """
    Load simulation CSV(s) and return a validated DataFrame.

    Args:
        cfg:  Pipeline config dict.
        path: Optional path to a single CSV file. If None, loads all files
              matching cfg['data']['file_pattern'] in cfg['data']['source'].
    """
    if path is not None:
        files = [Path(path)]
    else:
        source = Path(cfg["data"]["source"])
        pattern = cfg["data"]["file_pattern"]
        files = sorted(source.glob(pattern))
        if not files:
            raise FileNotFoundError(
                f"No files matching '{pattern}' in {source.resolve()}"
            )

    frames: list[pd.DataFrame] = []
    for f in files:
        df = pd.read_csv(f)
        df = _add_insertion_depth(df, cfg)
        frames.append(df)
        log.info("Loaded %-40s  rows=%d", f.name, len(df))

    combined = pd.concat(frames, ignore_index=True)
    validate(combined, cfg)
    log.info("Dataset ready — %d rows, %d columns", *combined.shape)
    return combined
