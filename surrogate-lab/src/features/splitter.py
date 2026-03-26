"""Train / validation / test split."""
import numpy as np
from sklearn.model_selection import train_test_split

from src.utils.logging_utils import get_logger

log = get_logger(__name__)


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return X_train, X_val, X_test, y_train, y_val, y_test.

    Ratios and random seed are read from cfg['split'].
    """
    split = cfg["split"]
    seed = split["random_seed"]
    test_ratio = float(split["test"])
    val_ratio = float(split["val"])

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=seed
    )
    relative_val = val_ratio / (1.0 - test_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=relative_val, random_state=seed
    )
    log.info(
        "Split → train=%d  val=%d  test=%d",
        len(X_train), len(X_val), len(X_test),
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
