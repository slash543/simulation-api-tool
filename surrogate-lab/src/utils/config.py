"""Config loader — single source of truth is configs/config.yaml."""
from pathlib import Path
import yaml


def load_config(path: str = "configs/config.yaml") -> dict:
    """Load YAML config. Raises FileNotFoundError if missing.

    Relative paths in mlflow.tracking_uri and training.checkpoint.dir are
    resolved relative to the config file's directory so the config works
    correctly regardless of the caller's working directory.
    """
    config_path = Path(path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    base_dir = config_path.parent
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Resolve relative paths against the config file's directory
    mlflow_uri = cfg.get("mlflow", {}).get("tracking_uri", "")
    if mlflow_uri and not mlflow_uri.startswith(("http://", "https://", "file://", "/")):
        cfg["mlflow"]["tracking_uri"] = str(base_dir / mlflow_uri)

    ckpt_dir = cfg.get("training", {}).get("checkpoint", {}).get("dir", "")
    if ckpt_dir and not Path(ckpt_dir).is_absolute():
        cfg["training"]["checkpoint"]["dir"] = str(base_dir / ckpt_dir)

    return cfg


def get_feature_names(cfg: dict) -> list[str]:
    return cfg["features"]["inputs"]


def get_target_name(cfg: dict) -> str:
    return cfg["features"]["target"]
