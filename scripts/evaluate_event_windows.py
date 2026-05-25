"""Evaluate event-window forecast errors from saved runs.

This script reloads a trained run, re-runs inference on its test dataset, and
computes MAE in short windows after control/event flags occur inside the forecast
horizon. It is intended as post-processing: no training is performed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import metrics as M
from src.data.feature_groups import ACTUATOR_EVENT_FLAGS, SETPOINT_EVENT_FLAGS
from src.data.window_dataset import WindowDataset
from src.models import ForecastingModel


DEFAULT_EVENT_COLS = ACTUATOR_EVENT_FLAGS + SETPOINT_EVENT_FLAGS
DEFAULT_WINDOWS = {
    "1h": 12,
    "3h": 36,
    "6h": 72,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute event-window MAE for saved forecasting runs."
    )
    parser.add_argument("--runs-dir", type=Path, default=Path("results/runs"))
    parser.add_argument("--run-name", nargs="*", default=None)
    parser.add_argument("--run-glob", default=None,
                        help="Glob pattern under runs-dir, e.g. 'cross_*_full_*_p10'.")
    parser.add_argument("--out-csv", type=Path, default=None,
                        help="Optional combined output CSV path.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--event-cols", nargs="*", default=DEFAULT_EVENT_COLS)
    parser.add_argument("--windows", nargs="*", default=list(DEFAULT_WINDOWS),
                        help="Window labels among 1h/3h/6h, or custom label=steps.")
    return parser.parse_args()


def parse_windows(items: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in items:
        if item in DEFAULT_WINDOWS:
            out[item] = DEFAULT_WINDOWS[item]
        elif "=" in item:
            label, steps = item.split("=", 1)
            out[label] = int(steps)
        else:
            raise ValueError(
                f"Unknown window {item!r}. Use one of {list(DEFAULT_WINDOWS)} "
                "or custom format label=steps."
            )
    return out


def resolve_runs(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    if args.run_name:
        paths.extend(args.runs_dir / name for name in args.run_name)
    if args.run_glob:
        paths.extend(sorted(args.runs_dir.glob(args.run_glob)))
    if not paths:
        raise ValueError("Provide --run-name or --run-glob.")
    unique = []
    seen = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        if not (path / "config.json").exists():
            raise FileNotFoundError(f"Missing config.json in {path}")
        if not (path / "model_best.pt").exists():
            raise FileNotFoundError(f"Missing model_best.pt in {path}")
        unique.append(path)
    return unique


def test_compartment(config: dict[str, Any]) -> str:
    if config["mode"] == "cross":
        return config["holdout"]
    return config["compartment"]


def build_test_dataset(config: dict[str, Any]) -> WindowDataset:
    return WindowDataset(
        compartment=test_compartment(config),
        split="test",
        lookback=int(config["lookback"]),
        horizon=int(config["horizon"]),
        stride=int(config["stride"]),
        data_dir=Path(config["data_dir"]),
    )


@torch.no_grad()
def predict_scaled(
    run_dir: Path,
    config: dict[str, Any],
    dataset: WindowDataset,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model = ForecastingModel(
        input_dim=dataset.feature_dim,
        backbone_name=config["backbone"],
        d_model=int(config["d_model"]),
        horizon=dataset.horizon,
        target_dim=dataset.target_dim,
        decoder_hidden_dim=int(config["decoder_hidden_dim"]),
        decoder_type=config.get("decoder_type", "mlp"),
        graph_mode=config.get("graph_mode"),
        feature_cols=dataset.feature_cols,
        gate_temperature=float(config.get("gate_temperature", 1.0)),
        prediction_mode=config.get("prediction_mode", "absolute"),
        target_cols=dataset.target_cols,
    )
    state = torch.load(run_dir / "model_best.pt", map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    feature_cols = list(dataset.feature_cols)
    target_cols = list(dataset.target_cols)
    target_feature_idx = [feature_cols.index(c) for c in target_cols]

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_true, all_pred, all_persist = [], [], []
    for x, y in loader:
        persist = x[:, -1, target_feature_idx].numpy()
        pred = model(x.to(device)).cpu().numpy()
        all_true.append(y.numpy())
        all_pred.append(pred)
        all_persist.append(M.persistence_predict(persist, dataset.horizon))

    return (
        np.concatenate(all_true, axis=0),
        np.concatenate(all_pred, axis=0),
        np.concatenate(all_persist, axis=0),
    )


def inverse_targets(
    dataset: WindowDataset,
    y_true_scaled: np.ndarray,
    y_pred_scaled: np.ndarray,
    y_persist_scaled: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_cols = dataset.target_cols
    numeric_cols = dataset.numeric_cols
    scaler = dataset.scaler
    return (
        M.inverse_transform_Y(y_true_scaled, scaler, target_cols, numeric_cols),
        M.inverse_transform_Y(y_pred_scaled, scaler, target_cols, numeric_cols),
        M.inverse_transform_Y(y_persist_scaled, scaler, target_cols, numeric_cols),
    )


def load_test_frame(config: dict[str, Any]) -> pd.DataFrame:
    comp = test_compartment(config)
    df = pd.read_parquet(Path(config["data_dir"]) / f"{comp}.parquet")
    return df[df["split"].eq("test")].copy()


def summarize_event(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_persist: np.ndarray,
    target_cols: list[str],
    event_starts: list[tuple[int, int]],
    window_label: str,
    window_steps: int,
) -> list[dict[str, Any]]:
    rows = []
    for target_i, target in enumerate(target_cols):
        model = fast_event_window_mae(
            np.abs(y_true[:, :, target_i] - y_pred[:, :, target_i]),
            event_starts,
            window_steps,
        )
        persist = fast_event_window_mae(
            np.abs(y_true[:, :, target_i] - y_persist[:, :, target_i]),
            event_starts,
            window_steps,
        )
        rows.append({
            "target": target,
            "window": window_label,
            "window_steps": window_steps,
            "n_events": model["n"],
            "model_mae": model["mean"],
            "model_std": model["std"],
            "persistence_mae": persist["mean"],
            "relative_mae": model["mean"] / max(persist["mean"], 1e-12),
        })

    return rows


def fast_event_window_mae(
    abs_error: np.ndarray,
    event_starts: list[tuple[int, int]],
    window_steps: int,
) -> dict[str, float | int]:
    """Vectorized event-window MAE from a precomputed absolute-error matrix."""
    if not event_starts:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}

    err = np.asarray(abs_error, dtype=np.float64)
    n_samples, horizon = err.shape
    events = np.asarray(event_starts, dtype=np.int64)
    sample_i = events[:, 0]
    start = events[:, 1]
    valid = (
        (sample_i >= 0) & (sample_i < n_samples)
        & (start >= 0) & (start < horizon)
    )
    if not np.any(valid):
        return {"mean": float("nan"), "std": float("nan"), "n": 0}

    sample_i = sample_i[valid]
    start = start[valid]
    end = np.minimum(start + window_steps, horizon)

    finite = np.isfinite(err)
    values = np.where(finite, err, 0.0)
    value_prefix = np.concatenate(
        [np.zeros((n_samples, 1), dtype=np.float64), np.cumsum(values, axis=1)],
        axis=1,
    )
    count_prefix = np.concatenate(
        [np.zeros((n_samples, 1), dtype=np.float64), np.cumsum(finite, axis=1)],
        axis=1,
    )

    sums = value_prefix[sample_i, end] - value_prefix[sample_i, start]
    counts = count_prefix[sample_i, end] - count_prefix[sample_i, start]
    per_event = sums / np.maximum(counts, 1.0)
    per_event[counts <= 0] = np.nan
    return {
        "mean": float(np.nanmean(per_event)),
        "std": float(np.nanstd(per_event)),
        "n": int(np.sum(np.isfinite(per_event))),
    }


def evaluate_run(
    run_dir: Path,
    event_cols: list[str],
    windows: dict[str, int],
    batch_size: int,
    device: torch.device,
) -> pd.DataFrame:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    dataset = build_test_dataset(config)
    y_true_s, y_pred_s, y_persist_s = predict_scaled(
        run_dir, config, dataset, batch_size, device
    )
    y_true, y_pred, y_persist = inverse_targets(
        dataset, y_true_s, y_pred_s, y_persist_s
    )

    df_test = load_test_frame(config)
    target_cols = list(dataset.target_cols)
    rows = []
    for event_col in event_cols:
        event_starts = M.detect_event_steps_in_forecast(
            df_test, dataset.meta, event_col, dataset.horizon
        )
        if not event_starts:
            continue
        for window_label, window_steps in windows.items():
            for row in summarize_event(
                y_true, y_pred, y_persist, target_cols,
                event_starts, window_label, window_steps,
            ):
                row.update({
                    "run": run_dir.name,
                    "mode": config["mode"],
                    "test_compartment": test_compartment(config),
                    "backbone": config["backbone"],
                    "seed": config["seed"],
                    "event_col": event_col,
                })
                rows.append(row)

    result = pd.DataFrame(rows)
    result.to_csv(run_dir / "event_windows.csv", index=False)
    return result


def main() -> None:
    args = parse_args()
    windows = parse_windows(args.windows)
    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    frames = []
    for run_dir in resolve_runs(args):
        print(f"evaluating {run_dir.name} on {device}")
        result = evaluate_run(
            run_dir=run_dir,
            event_cols=list(args.event_cols),
            windows=windows,
            batch_size=args.batch_size,
            device=device,
        )
        frames.append(result)
        print(f"  rows={len(result)} -> {run_dir / 'event_windows.csv'}")

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(args.out_csv, index=False)
        print(f"combined -> {args.out_csv}")


if __name__ == "__main__":
    main()
