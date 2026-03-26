"""Training loop with early stopping, checkpointing, and MLflow logging."""
from __future__ import annotations

import time
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.utils.logging_utils import get_logger

log = get_logger(__name__)


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float = 1e-6) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best = np.inf
        self.counter = 0
        self.triggered = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
        return self.triggered


def _make_loader(
    X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool
) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=True)


def train(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: dict,
    run_name: str | None = None,
) -> tuple[nn.Module, str]:
    """
    Train the model and return ``(best_model, mlflow_run_id)``.

    All hyperparameters, metrics, and the best model checkpoint are logged
    to MLflow. Early stopping and checkpointing are controlled by config.
    The run ID is returned so callers can attach extra tags / artifacts
    and register the model in the MLflow Model Registry.
    """
    tcfg = cfg["training"]
    mlcfg = cfg["mlflow"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg["lr"])
    criterion = nn.MSELoss()
    batch_size: int = tcfg["batch_size"]
    epochs: int = tcfg["epochs"]

    train_loader = _make_loader(X_train, y_train, batch_size, shuffle=True)
    val_loader = _make_loader(X_val, y_val, batch_size, shuffle=False)

    es_cfg = tcfg["early_stopping"]
    stopper = (
        EarlyStopping(es_cfg["patience"], es_cfg["min_delta"])
        if es_cfg["enabled"]
        else None
    )

    ckpt_dir = Path(tcfg["checkpoint"]["dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = ckpt_dir / "best_model.pt"
    best_val_loss = np.inf

    mlflow.set_tracking_uri(mlcfg["tracking_uri"])
    mlflow.set_experiment(mlcfg["experiment_name"])

    params = {
        "model_type": cfg["model"]["type"],
        "layers": str(cfg["model"]["layers"]),
        "activation": cfg["model"].get("activation"),
        "dropout": cfg["model"].get("dropout", 0.0),
        "lr": tcfg["lr"],
        "batch_size": batch_size,
        "epochs": epochs,
        "optimizer": tcfg["optimizer"],
        "normalization": cfg["features"]["normalization"]["method"],
        "features": ",".join(cfg["features"]["inputs"]),
        "target": cfg["features"]["target"],
        "device": str(device),
    }

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        log.info("MLflow run %s started on %s", run.info.run_id[:8], device)

        t0 = time.time()
        for epoch in range(1, epochs + 1):
            # ── train ──────────────────────────────────────────────────────
            model.train()
            train_loss = 0.0
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(Xb), yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(Xb)
            train_loss /= len(X_train)

            # ── validate ───────────────────────────────────────────────────
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for Xb, yb in val_loader:
                    Xb, yb = Xb.to(device), yb.to(device)
                    val_loss += criterion(model(Xb), yb).item() * len(Xb)
            val_loss /= len(X_val)

            mlflow.log_metrics(
                {"train_loss": train_loss, "val_loss": val_loss}, step=epoch
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if tcfg["checkpoint"]["save_best"]:
                    torch.save(model.state_dict(), best_ckpt)

            if epoch % 10 == 0 or epoch == 1:
                log.info(
                    "Epoch %4d/%d  train=%.6f  val=%.6f  [%.1fs]",
                    epoch, epochs, train_loss, val_loss, time.time() - t0,
                )

            if stopper and stopper.step(val_loss):
                log.info(
                    "Early stopping at epoch %d — best val=%.6f", epoch, stopper.best
                )
                break

        # Restore best weights
        if best_ckpt.exists():
            model.load_state_dict(torch.load(best_ckpt, map_location=device))

        mlflow.log_metric("best_val_loss", best_val_loss)

        if mlcfg.get("log_artifacts") and best_ckpt.exists():
            mlflow.log_artifact(str(best_ckpt))

        if mlcfg.get("register_model"):
            mlflow.pytorch.log_model(
                model, "model",
                registered_model_name=mlcfg["model_name"],
            )

        log.info("Training complete — best val loss: %.6f", best_val_loss)
        run_id = run.info.run_id

    return model, run_id
