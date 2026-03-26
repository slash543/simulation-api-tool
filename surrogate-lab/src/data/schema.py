"""Config-driven schema validation for simulation dataframes."""
import pandas as pd
from src.utils.config import get_feature_names, get_target_name
from src.utils.logging_utils import get_logger

log = get_logger(__name__)


def validate(df: pd.DataFrame, cfg: dict) -> None:
    """Raise ValueError if required columns are missing."""
    required = set(get_feature_names(cfg)) | {get_target_name(cfg)}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )
    log.info("Schema OK — shape=%s", df.shape)
