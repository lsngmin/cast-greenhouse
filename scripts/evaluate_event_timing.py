"""Evaluate event-response timing errors from saved forecasting runs.

This post-processing script measures whether a model predicts the *timing* of
the largest post-event response, not just the average error inside an event
window. For each actuator/setpoint event inside a forecast horizon, it compares
the true and predicted response curves relative to their value at the event
step, then reports peak-timing lag in minutes.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import metrics as M
from src.data.feature_groups import ACTUATOR_EVENT_FLAGS, SETPOINT_EVENT_FLAGS
from scripts.evaluate_event_windows import (
    build_test_dataset,
    inverse_targets,
    load_test_frame,
    parse_windows,
    predict_scaled,
    resolve_runs,
    test_compartment,
)


DEFAULT_EVENT_COLS = ACTUATOR_EVENT_FLAGS + SETPOINT_EVENT_FLAGS
DEFAULT_WINDOWS = {
    "1h": 12,
    "3h": 36,
    "6h": 72,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute post-event peak timing errors for saved runs."
    )
    parser.add_argument("--runs-dir", type=Path, default=Path("results/runs"))
    parser.add_argument("--run-name", nargs="*", default=None)
    parser.add_argument(
        "--run-glob",
        nargs="*",
        default=None,
        help="Glob pattern(s) under runs-dir.",
    )
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--event-cols", nargs="*", default=DEFAULT_EVENT_COLS)
    parser.add_argument("--windows", nargs="*", default=list(DEFAULT_WINDOWS))
    parser.add_argument("--step-minutes", type=float, default=5.0)
    parser.add_argument(
        "--min-true-amplitude",
        type=float,
        default=0.0,
        help="Drop events whose true absolute response peak is below this value.",
    )
    parser.add_argument(
        "--response-fraction",
        type=float,
        default=0.5,
        help="Fraction of |peak response| used to define response delay (default 0.5 = 50%% rise time).",
    )
    return parser.parse_args()


def _first_crossing_step(response: np.ndarray, threshold: float) -> int | None:
    """First step (excluding t=0) where |response| >= threshold."""
    finite = np.isfinite(response)
    if not finite.any():
        return None
    abs_resp = np.where(finite, np.abs(response), 0.0)
    above = abs_resp >= threshold
    above[0] = False  # exclude trivial t=0
    if not above.any():
        return None
    return int(np.argmax(above))


def resolve_timing_runs(args: argparse.Namespace) -> list[Path]:
    """Resolve run directories, allowing multiple --run-glob patterns."""
    if isinstance(args.run_glob, list):
        paths: list[Path] = []
        if args.run_name:
            paths.extend(args.runs_dir / name for name in args.run_name)
        for pattern in args.run_glob:
            paths.extend(sorted(args.runs_dir.glob(pattern)))
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

    return resolve_runs(args)


def _event_peak_records(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    event_starts: list[tuple[int, int]],
    window_steps: int,
    step_minutes: float,
    min_true_amplitude: float = 0.0,
    response_fraction: float = 0.5,
) -> list[dict[str, float]]:
    """Return per-event peak timing + response-delay records for a single target.

    y_true/y_pred are shaped (n_samples, horizon). Response curves are measured
    relative to each curve's value at the event step to reduce sensitivity to
    absolute bias. Response delay is the time-to-`response_fraction * |peak|`
    crossing (default 50%, the standard rise-time metric).
    """
    n_samples, horizon = y_true.shape
    records: list[dict[str, float]] = []
    for sample_i, start in event_starts:
        if sample_i < 0 or sample_i >= n_samples or start < 0 or start >= horizon:
            continue
        end = min(start + window_steps, horizon)
        if end - start < 2:
            continue

        true_seg = y_true[sample_i, start:end].astype(np.float64)
        pred_seg = y_pred[sample_i, start:end].astype(np.float64)
        finite = np.isfinite(true_seg) & np.isfinite(pred_seg)
        if finite.sum() < 2:
            continue

        true_resp = true_seg - true_seg[0]
        pred_resp = pred_seg - pred_seg[0]

        # Keep the native time index even if some values are non-finite.
        true_score = np.where(np.isfinite(true_resp), np.abs(true_resp), -np.inf)
        pred_score = np.where(np.isfinite(pred_resp), np.abs(pred_resp), -np.inf)
        true_score[0] = -np.inf
        pred_score[0] = -np.inf
        if not np.isfinite(true_score).any() or not np.isfinite(pred_score).any():
            continue

        true_peak_step = int(np.argmax(true_score))
        pred_peak_step = int(np.argmax(pred_score))
        true_amp = float(true_resp[true_peak_step])
        pred_amp = float(pred_resp[pred_peak_step])
        if abs(true_amp) < min_true_amplitude:
            continue

        signed_lag_steps = pred_peak_step - true_peak_step
        record = {
            "true_peak_step": float(true_peak_step),
            "pred_peak_step": float(pred_peak_step),
            "signed_lag_min": float(signed_lag_steps * step_minutes),
            "abs_lag_min": float(abs(signed_lag_steps) * step_minutes),
            "true_peak_amp": true_amp,
            "pred_peak_amp": pred_amp,
            "peak_amp_abs_error": float(abs(pred_amp - true_amp)),
        }

        # Response delay: time to reach response_fraction * |peak|.
        # Use each curve's own peak as reference (consistent with peak-timing).
        true_thresh = response_fraction * abs(true_amp)
        pred_thresh = response_fraction * abs(pred_amp)
        true_delay_step = _first_crossing_step(true_resp, true_thresh)
        pred_delay_step = _first_crossing_step(pred_resp, pred_thresh)
        if true_delay_step is not None and pred_delay_step is not None:
            signed_delay_steps = pred_delay_step - true_delay_step
            record["true_delay_step"] = float(true_delay_step)
            record["pred_delay_step"] = float(pred_delay_step)
            record["true_delay_min"] = float(true_delay_step * step_minutes)
            record["pred_delay_min"] = float(pred_delay_step * step_minutes)
            record["response_delay_signed_min"] = float(signed_delay_steps * step_minutes)
            record["response_delay_abs_min"] = float(abs(signed_delay_steps) * step_minutes)

        records.append(record)
    return records


def _summarize_records(
    records: list[dict[str, float]],
    target: str,
    window_label: str,
    window_steps: int,
    step_minutes: float,
) -> dict[str, Any]:
    if not records:
        return {
            "target": target,
            "window": window_label,
            "window_steps": window_steps,
            "n_events": 0,
            "true_peak_min_mean": np.nan,
            "pred_peak_min_mean": np.nan,
            "signed_lag_min_mean": np.nan,
            "signed_lag_min_median": np.nan,
            "abs_lag_min_mean": np.nan,
            "abs_lag_min_median": np.nan,
            "peak_amp_abs_error_mean": np.nan,
            "n_events_delay": 0,
            "response_delay_signed_mean": np.nan,
            "response_delay_signed_median": np.nan,
            "response_delay_abs_mean": np.nan,
            "response_delay_abs_median": np.nan,
        }
    df = pd.DataFrame(records)
    out = {
        "target": target,
        "window": window_label,
        "window_steps": window_steps,
        "n_events": int(len(df)),
        "true_peak_min_mean": float(df["true_peak_step"].mean() * step_minutes),
        "pred_peak_min_mean": float(df["pred_peak_step"].mean() * step_minutes),
        "signed_lag_min_mean": float(df["signed_lag_min"].mean()),
        "signed_lag_min_median": float(df["signed_lag_min"].median()),
        "abs_lag_min_mean": float(df["abs_lag_min"].mean()),
        "abs_lag_min_median": float(df["abs_lag_min"].median()),
        "peak_amp_abs_error_mean": float(df["peak_amp_abs_error"].mean()),
    }
    if "response_delay_signed_min" in df.columns:
        delay_df = df.dropna(subset=["response_delay_signed_min"])
        out["n_events_delay"] = int(len(delay_df))
        if len(delay_df) > 0:
            out["response_delay_signed_mean"] = float(delay_df["response_delay_signed_min"].mean())
            out["response_delay_signed_median"] = float(delay_df["response_delay_signed_min"].median())
            out["response_delay_abs_mean"] = float(delay_df["response_delay_abs_min"].mean())
            out["response_delay_abs_median"] = float(delay_df["response_delay_abs_min"].median())
        else:
            out["response_delay_signed_mean"] = np.nan
            out["response_delay_signed_median"] = np.nan
            out["response_delay_abs_mean"] = np.nan
            out["response_delay_abs_median"] = np.nan
    else:
        out["n_events_delay"] = 0
        out["response_delay_signed_mean"] = np.nan
        out["response_delay_signed_median"] = np.nan
        out["response_delay_abs_mean"] = np.nan
        out["response_delay_abs_median"] = np.nan
    return out


def summarize_event_timing(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_cols: list[str],
    event_starts: list[tuple[int, int]],
    window_label: str,
    window_steps: int,
    step_minutes: float,
    min_true_amplitude: float,
    response_fraction: float = 0.5,
) -> list[dict[str, Any]]:
    rows = []
    for target_i, target in enumerate(target_cols):
        records = _event_peak_records(
            y_true[:, :, target_i],
            y_pred[:, :, target_i],
            event_starts,
            window_steps,
            step_minutes,
            min_true_amplitude,
            response_fraction,
        )
        rows.append(
            _summarize_records(
                records, target, window_label, window_steps, step_minutes
            )
        )

    return rows


@torch.no_grad()
def evaluate_run(
    run_dir: Path,
    event_cols: list[str],
    windows: dict[str, int],
    batch_size: int,
    device: torch.device,
    step_minutes: float,
    min_true_amplitude: float,
    response_fraction: float = 0.5,
) -> pd.DataFrame:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    dataset = build_test_dataset(config)
    y_true_s, y_pred_s, y_persist_s = predict_scaled(
        run_dir, config, dataset, batch_size, device
    )
    y_true, y_pred, _ = inverse_targets(dataset, y_true_s, y_pred_s, y_persist_s)

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
            for row in summarize_event_timing(
                y_true, y_pred, target_cols, event_starts,
                window_label, window_steps, step_minutes, min_true_amplitude,
                response_fraction,
            ):
                row.update({
                    "run": run_dir.name,
                    "mode": config["mode"],
                    "test_compartment": test_compartment(config),
                    "backbone": config["backbone"],
                    "graph_mode": config.get("graph_mode") or "none",
                    "decoder_type": config.get("decoder_type", "mlp"),
                    "seed": config["seed"],
                    "event_col": event_col,
                    "response_fraction": response_fraction,
                })
                rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(run_dir / "event_timing.csv", index=False)
    return result


def main() -> None:
    args = parse_args()
    windows = parse_windows(args.windows)
    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    frames = []
    for run_dir in resolve_timing_runs(args):
        print(f"timing {run_dir.name} on {device}")
        result = evaluate_run(
            run_dir=run_dir,
            event_cols=list(args.event_cols),
            windows=windows,
            batch_size=args.batch_size,
            device=device,
            step_minutes=args.step_minutes,
            min_true_amplitude=args.min_true_amplitude,
            response_fraction=args.response_fraction,
        )
        frames.append(result)
        print(f"  rows={len(result)} -> {run_dir / 'event_timing.csv'}")

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(args.out_csv, index=False)
        print(f"combined -> {args.out_csv}")


if __name__ == "__main__":
    main()
