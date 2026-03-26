#!/usr/bin/env python3
"""
SurrogateLab training entrypoint.

Runs locally or as an Azure ML job (pass --config and --data as CLI args).

Usage:
    python scripts/train.py
    python scripts/train.py --config configs/config.yaml --data path/to/sim.csv
    python scripts/train.py --run-name experiment-01
"""
from __future__ import annotations

import argparse

import mlflow
import torch

from src.data.loader import load_simulation_data
from src.evaluation.metrics import evaluate_model
from src.evaluation.visualization import plot_predicted_vs_actual, plot_residuals
from src.features.engineer import FeaturePipeline, build_xy
from src.features.splitter import split_data
from src.models.factory import build_model
from src.training.trainer import train
from src.utils.config import load_config
from src.utils.logging_utils import get_logger

log = get_logger("train")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SurrogateLab — train a surrogate model")
    p.add_argument("--config", default="configs/config.yaml", help="Path to config YAML")
    p.add_argument("--data", default=None, help="Single CSV file (overrides config source dir)")
    p.add_argument("--run-name", default=None, help="MLflow run name")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    log.info("Config: %s", args.config)

    # ── Data ──────────────────────────────────────────────────────────────
    df = load_simulation_data(cfg, path=args.data)
    X, y = build_xy(df, cfg)

    # ── Split ─────────────────────────────────────────────────────────────
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, cfg)

    # ── Feature pipeline (fit only on train) ──────────────────────────────
    pipeline = FeaturePipeline(cfg)
    X_train_s, y_train_s = pipeline.fit_transform(X_train, y_train)
    X_val_s,   y_val_s   = pipeline.transform(X_val,   y_val)
    X_test_s,  y_test_s  = pipeline.transform(X_test,  y_test)

    scaler_dir = "artifacts/scalers"
    pipeline.save(scaler_dir)

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_model(X_train_s.shape[1], cfg)

    # ── Train ─────────────────────────────────────────────────────────────
    model = train(model, X_train_s, y_train_s, X_val_s, y_val_s, cfg,
                  run_name=args.run_name)

    # ── Evaluate on held-out test set ─────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metrics = evaluate_model(model, X_test_s, y_test_s, pipeline, device_str=device)
    log.info("Test → %s", metrics)

    # ── Log test metrics + plots ──────────────────────────────────────────
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    with mlflow.start_run(run_name=f"{args.run_name or 'train'}-test-eval"):
        mlflow.log_params({"config": args.config, "data": str(args.data)})
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})

        model.eval()
        with torch.no_grad():
            y_pred_s = model(
                torch.from_numpy(X_test_s).to(device)
            ).cpu().numpy()

        y_pred = pipeline.inverse_transform_y(y_pred_s)
        y_true = pipeline.inverse_transform_y(y_test_s)

        fig1 = plot_predicted_vs_actual(
            y_true, y_pred, save_path="artifacts/pred_vs_actual.png"
        )
        fig2 = plot_residuals(y_true, y_pred, save_path="artifacts/residuals.png")

        mlflow.log_artifact("artifacts/pred_vs_actual.png")
        mlflow.log_artifact("artifacts/residuals.png")
        mlflow.log_artifacts(scaler_dir, artifact_path="scalers")

    log.info("Done.")


if __name__ == "__main__":
    main()
