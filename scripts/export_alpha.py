"""Export per-(sample, timestep) source-attention weights from a trained
SourceAwareEmbedding model. Output is saved as `alpha_export.npz` inside the
run directory and is the proper artifact for interpretability analysis —
unlike `model.embed.last_alpha`, which only holds the most recent batch.

Examples
--------
Single run, full test split:
    python scripts/export_alpha.py --run-name cross_holdout-Reference_mamba_esa_seed42

Multiple runs by glob, val + test:
    python scripts/export_alpha.py --run-glob 'cross_*esa*' --splits val test

Notes
-----
Inference runs ONLY the embedding sub-module (not the backbone/decoder), so
this is much cheaper than re-evaluation. The alpha tensor for one window is
(L=288, n_sources=5); for N windows the export is (N, L, n_sources) float32.

Output `.npz` contains:
    alpha       : (N, L, n_sources) float32
    sources     : (n_sources,) str   — source names in column order
    t0_ts       : (N,) datetime64    — window start timestamp (forecast-issue time)
    t_pred_end_ts: (N,) datetime64    — window end timestamp
    split       : str                — 'train' / 'val' / 'test'
    compartment : str                — source compartment for this split
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.window_dataset import WindowDataset
from src.models import ForecastingModel, SourceAwareEmbedding


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--runs-dir", type=Path, default=Path("results/runs"))
    p.add_argument("--run-name", nargs="*", default=None,
                   help="Specific run dir name(s) under runs-dir.")
    p.add_argument("--run-glob", default=None,
                   help="Glob pattern (e.g. 'cross_*esa*').")
    p.add_argument("--splits", nargs="+", default=["test"],
                   choices=["train", "val", "test"],
                   help="Which split(s) to export alpha for.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default=None)
    return p.parse_args()


def resolve_runs(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    if args.run_name:
        paths.extend(args.runs_dir / n for n in args.run_name)
    if args.run_glob:
        paths.extend(sorted(args.runs_dir.glob(args.run_glob)))
    if not paths:
        raise ValueError("Provide --run-name or --run-glob.")
    out, seen = [], set()
    for p in paths:
        if p in seen:
            continue
        seen.add(p)
        if not (p / "config.json").exists() or not (p / "model_best.pt").exists():
            print(f"[skip] missing config or weights: {p}")
            continue
        out.append(p)
    return out


def test_compartment(config: dict[str, Any], split: str) -> str:
    """Compartment whose data is used for `split`."""
    if config["mode"] == "cross" and split == "test":
        return config["holdout"]
    return config["compartment"]


def build_dataset(config: dict[str, Any], split: str) -> WindowDataset:
    """Build dataset matching the training config (no feature_group anymore)."""
    if config["mode"] == "cross" and split != "test":
        # train/val for cross-mode are the union of non-holdout compartments;
        # exporting alpha per-compartment is cleaner — caller can re-export.
        raise NotImplementedError(
            "alpha export for cross-mode train/val is not implemented "
            "(would mix compartments). Use --splits test, or run per-compartment."
        )
    comp = test_compartment(config, split)
    return WindowDataset(
        compartment=comp,
        split=split,
        lookback=int(config["lookback"]),
        horizon=int(config["horizon"]),
        stride=int(config["stride"]),
        data_dir=Path(config["data_dir"]),
    )


@torch.no_grad()
def collect_alpha(model: ForecastingModel, dataset: WindowDataset,
                  batch_size: int, device: torch.device) -> np.ndarray:
    """Run the embedding sub-module over the entire dataset and return alpha.

    Returns
    -------
    alpha : (N, L, n_sources) float32
    """
    assert isinstance(model.embed, SourceAwareEmbedding)
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    chunks: list[np.ndarray] = []
    for x, _ in loader:
        x = x.to(device)
        _ = model.embed(x)               # populates last_alpha
        chunks.append(model.embed.last_alpha.cpu().numpy().astype("float32"))
    return np.concatenate(chunks, axis=0)


def export_run(run_dir: Path, splits: list[str], batch_size: int,
               device: torch.device) -> None:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))

    # Build model once; reuse for all splits
    # (input_dim is set by first dataset; verify on subsequent splits)
    first_ds = build_dataset(config, splits[0])
    model = ForecastingModel(
        input_dim=first_ds.feature_dim,
        backbone_name=config["backbone"],
        d_model=int(config["d_model"]),
        horizon=first_ds.horizon,
        target_dim=first_ds.target_dim,
        decoder_hidden_dim=int(config["decoder_hidden_dim"]),
        decoder_type=config.get("decoder_type", "mlp"),
        graph_mode=config.get("graph_mode"),
        feature_cols=first_ds.feature_cols,
        gate_temperature=float(config.get("gate_temperature", 1.0)),
    )
    state = torch.load(run_dir / "model_best.pt", map_location=device)
    model.load_state_dict(state)
    model.to(device)

    for split in splits:
        ds = first_ds if split == splits[0] else build_dataset(config, split)
        comp = test_compartment(config, split)
        print(f"  [{run_dir.name}] split={split} comp={comp} n={len(ds)} ...")
        alpha = collect_alpha(model, ds, batch_size, device)
        out = run_dir / f"alpha_{split}.npz"
        np.savez_compressed(
            out,
            alpha=alpha,
            sources=np.array(model.embed.sources),
            t0_ts=ds.meta["t0_ts"].astype("datetime64[ns]").values,
            t_pred_end_ts=ds.meta["t_pred_end_ts"].astype("datetime64[ns]").values,
            split=np.array(split),
            compartment=np.array(comp),
        )
        print(f"    -> {out}  shape={alpha.shape}")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    runs = resolve_runs(args)
    print(f"exporting alpha for {len(runs)} runs on {device}")
    for run_dir in runs:
        export_run(run_dir, args.splits, args.batch_size, device)


if __name__ == "__main__":
    main()
