"""Train one forecasting run from the command line.

Examples
--------
Single-compartment sanity run:
    python scripts/train_one.py --mode single --compartment Reference --backbone lstm --max-epochs 5

Cross-compartment holdout run:
    python scripts/train_one.py --mode cross --holdout Reference --feature-group sensor+weather+state+sp+vip
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.window_dataset import WindowDataset, make_concat_dataset
from src.models import ForecastingModel
from src.train import Trainer


DEFAULT_COMPARTMENTS = (
    "AICU",
    "Automatoes",
    "Digilog",
    "IUACAAS",
    "Reference",
    "TheAutomators",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one greenhouse forecasting model.")

    parser.add_argument("--mode", choices=("single", "cross"), default="single")
    parser.add_argument("--compartment", default="Reference",
                        help="Compartment used for single mode.")
    parser.add_argument("--holdout", default="Reference",
                        help="Held-out compartment used for cross mode.")
    parser.add_argument("--compartments", nargs="*", default=list(DEFAULT_COMPARTMENTS),
                        help="All compartments available for cross mode.")
    parser.add_argument("--feature-group", default="sensor+weather+state+sp+vip")
    parser.add_argument("--backbone", default="lstm",
                        help="Backbone name passed to ForecastingModel.")

    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/runs"))
    parser.add_argument("--run-name", default=None)

    parser.add_argument("--lookback", type=int, default=288)
    parser.add_argument("--horizon", type=int, default=288)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--decoder-hidden-dim", type=int, default=256)

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--device", default=None,
                        help="Override device, e.g. cuda, cuda:0, or cpu.")
    parser.add_argument("--quiet", action="store_true")

    return parser.parse_args()


def make_run_name(args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name
    split_name = args.compartment if args.mode == "single" else f"holdout-{args.holdout}"
    group = args.feature_group.replace("+", "_")
    return f"{args.mode}_{split_name}_{args.backbone}_{group}_seed{args.seed}"


def build_datasets(args: argparse.Namespace) -> tuple[Any, Any, WindowDataset]:
    common = {
        "feature_group": args.feature_group,
        "lookback": args.lookback,
        "horizon": args.horizon,
        "stride": args.stride,
        "data_dir": args.data_dir,
    }

    if args.mode == "single":
        train_ds = WindowDataset(args.compartment, split="train", **common)
        val_ds = WindowDataset(args.compartment, split="val", **common)
        test_ds = WindowDataset(args.compartment, split="test", **common)
        return train_ds, val_ds, test_ds

    if args.holdout not in args.compartments:
        raise ValueError(f"holdout={args.holdout!r} is not in compartments={args.compartments!r}")

    train_comps = [c for c in args.compartments if c != args.holdout]
    train_ds = make_concat_dataset(train_comps, split="train", **common)
    val_ds = make_concat_dataset(train_comps, split="val", **common)
    test_ds = WindowDataset(args.holdout, split="test", **common)
    return train_ds, val_ds, test_ds


def dataset_for_shape(dataset: Any) -> WindowDataset:
    if isinstance(dataset, WindowDataset):
        return dataset
    if hasattr(dataset, "datasets") and dataset.datasets:
        return dataset.datasets[0]
    raise TypeError(f"Cannot infer model shape from dataset type {type(dataset)!r}")


def dataloader(dataset: Any, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def json_default(obj: Any) -> Any:
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)


def main() -> None:
    args = parse_args()
    run_name = make_run_name(args)
    out_dir = args.out_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds = build_datasets(args)
    shape_ds = dataset_for_shape(train_ds)

    model = ForecastingModel(
        input_dim=shape_ds.feature_dim,
        backbone_name=args.backbone,
        d_model=args.d_model,
        horizon=args.horizon,
        target_dim=shape_ds.target_dim,
        decoder_hidden_dim=args.decoder_hidden_dim,
    )

    train_loader = dataloader(train_ds, args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    val_loader = dataloader(val_ds, args.eval_batch_size, shuffle=False,
                            num_workers=args.num_workers)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer_kwargs={"lr": args.lr, "weight_decay": args.weight_decay},
        early_stopping=args.patience,
        device=args.device,
        seed=args.seed,
        max_grad_norm=args.max_grad_norm,
        verbose=not args.quiet,
    )

    config = vars(args) | {
        "run_name": run_name,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
        "feature_dim": shape_ds.feature_dim,
        "target_dim": shape_ds.target_dim,
        "model_params": model.count_parameters(),
    }
    (out_dir / "config.json").write_text(
        json.dumps(config, indent=2, default=json_default),
        encoding="utf-8",
    )

    history = trainer.fit(max_epochs=args.max_epochs)
    history.to_csv(out_dir / "history.csv", index=False)

    metrics = trainer.test(test_ds, batch_size=args.eval_batch_size)
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, default=json_default),
        encoding="utf-8",
    )

    torch.save(trainer.model.state_dict(), out_dir / "model_best.pt")

    print(f"saved run to {out_dir.resolve()}")
    print(json.dumps({
        "run_name": run_name,
        "best_epoch": metrics.get("best_epoch"),
        "best_val_loss": metrics.get("best_val_loss"),
        "metrics_path": str((out_dir / "metrics.json").resolve()),
    }, indent=2, default=json_default))


if __name__ == "__main__":
    main()
