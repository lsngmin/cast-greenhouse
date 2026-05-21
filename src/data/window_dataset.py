"""WindowDataset — PyTorch Dataset wrapper for AGC2 sliding windows.

────────────────────────────────────────────────────────────────────────────
역할 (한 줄)
    `preprocessing` + `windowing` + `feature_groups` 3개 sub-module을 묶어
    학습/추론에서 단일 인터페이스 (`torch.utils.data.Dataset`)로 제공.

────────────────────────────────────────────────────────────────────────────
왜 필요한가?

ForecastingModel은 `forward(x: (B, L, F)) → (B, H, V)` 시그니처를 따른다.
즉 학습 코드는 numpy/parquet에서 직접 X tensor를 만들어 model에 넣어야 한다.
세 sub-module을 매번 학습 코드에서 따로 호출하면 boilerplate가 길어진다.

WindowDataset은 다음을 한 번에 처리:
  1. processed parquet load
  2. scaler·column meta load
  3. feature_group 기반 컬럼 선택
  4. (X, Y, meta) tensor 생성 (scaled)
  5. `__len__`, `__getitem__` 인터페이스 → DataLoader 호환

────────────────────────────────────────────────────────────────────────────
사용 예

    from torch.utils.data import DataLoader
    from src.data import WindowDataset

    ds = WindowDataset(compartment='Reference', split='train')
    loader = DataLoader(ds, batch_size=32, shuffle=True)

    for x, y in loader:                      # x: (B, L, F), y: (B, H, V)
        y_hat = model(x)
        loss = criterion(y_hat, y)
        ...

────────────────────────────────────────────────────────────────────────────
설계 결정

- **`torch.utils.data.Dataset` 상속 (plain class 대신):**
    PyTorch 표준 인터페이스. DataLoader가 batch/shuffle/multi-worker 처리.
    PyTorch Lightning, distributed training과 호환.

- **`__getitem__`은 `(x, y)` 만 반환:**
    학습 루프에서 가장 흔히 쓰는 형태. meta(timestamps, compartment 등)는
    `ds.meta`로 별도 접근 가능 (event-window 분석용).

- **Tensor 변환은 eager로 (numpy → torch 한 번):**
    `__init__`에서 numpy array를 한 번 torch tensor로 변환. `__getitem__`마다
    변환하면 overhead. 메모리 사용량은 04 결과 기준 1 compartment·1 split
    train ≈ 250 MB (float32) — 무난.

- **Cross-compartment용 helper 별도 제공:**
    여러 compartment를 합쳐서 train 만들 때 `ConcatDataset` 사용 권장.
    또는 `make_concat_dataset` helper.

────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset

from . import windowing as win


DEFAULT_DATA_DIR = Path('data/processed')


class WindowDataset(Dataset):
    """Sliding window forecasting dataset.

    Args:
        compartment:    Compartment 이름 (예: 'Reference').
        split:          'train' / 'val' / 'test'.
        lookback:       lookback steps. default 288 (24h).
        horizon:        horizon steps. default 288 (24h).
        stride:         window stride. default 12 (1h).
        data_dir:       processed parquet 디렉토리. default 'data/processed'.

    Feature 컬럼은 `feature_groups.ALL_FEATURES` (53개 curated set) 고정.

    Attributes:
        X     : torch.Tensor (n, L, F=53) float32, scaled features
        Y     : torch.Tensor (n, H, V=3) float32, scaled targets
        meta  : pd.DataFrame with [t0_idx, t0_ts, lookback_start_ts, t_pred_end_ts]
        feature_cols : list[str]   — 사용된 feature 컬럼 이름 (53개)
        target_cols  : list[str]   — 사용된 target 컬럼 이름 (3개)
        cfg          : WindowConfig
        feature_dim  : int  — F (편의)
        target_dim   : int  — V (편의)

    Returns from __getitem__: (x, y)
        x: (L, F) float32
        y: (H, V) float32
    """

    def __init__(
        self,
        compartment: str,
        split: str = 'train',
        lookback: int = 288,
        horizon: int = 288,
        stride: int = 12,
        data_dir: Path | str = DEFAULT_DATA_DIR,
    ):
        super().__init__()
        self.compartment = compartment
        self.split = split
        self.data_dir = Path(data_dir)

        cfg = win.WindowConfig(lookback=lookback, horizon=horizon, stride=stride)
        bundle = win.make_split_windows(
            out_dir=self.data_dir,
            compartment=compartment,
            cfg=cfg,
            splits=(split,),
        )
        split_data = bundle[split]

        # Eager numpy → torch (학습 루프 마다 cast 안 함)
        self.X = torch.from_numpy(split_data['X'])         # (n, L, F)
        self.Y = torch.from_numpy(split_data['Y'])         # (n, H, V)
        self.meta = split_data['meta']                     # pd.DataFrame
        self.feature_cols = bundle['feature_cols']
        self.target_cols = bundle['target_cols']
        self.cfg = cfg

        # Trainer.test()에서 inverse_transform_Y 호출 시 필요
        art = win.load_artifacts(self.data_dir)
        self.scaler = art['scalers'][compartment]
        self.numeric_cols = art['columns']['numeric_cols']

    @property
    def feature_dim(self) -> int:
        return self.X.shape[-1]

    @property
    def target_dim(self) -> int:
        return self.Y.shape[-1]

    @property
    def lookback(self) -> int:
        return self.cfg.lookback

    @property
    def horizon(self) -> int:
        return self.cfg.horizon

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.Y[idx]

    def __repr__(self) -> str:
        return (
            f"WindowDataset(compartment={self.compartment!r}, "
            f"split={self.split!r}, "
            f"n={len(self)}, X={tuple(self.X.shape)}, Y={tuple(self.Y.shape)})"
        )


def make_concat_dataset(
    compartments: Sequence[str],
    split: str = 'train',
    lookback: int = 288,
    horizon: int = 288,
    stride: int = 12,
    data_dir: Path | str = DEFAULT_DATA_DIR,
) -> ConcatDataset:
    """여러 compartment를 합친 dataset (cross-compartment 학습용).

    각 compartment에 대해 `WindowDataset`을 만들고 `ConcatDataset`으로 묶음.

    Example (5-fold cross-compartment, hold-out 'Reference'):
        train_comps = ['AICU','Automatoes','Digilog','IUACAAS','TheAutomators']
        ds_train = make_concat_dataset(train_comps, split='train')
        ds_test  = WindowDataset('Reference', split='test')
    """
    datasets = [
        WindowDataset(
            compartment=c, split=split,
            lookback=lookback, horizon=horizon, stride=stride, data_dir=data_dir,
        )
        for c in compartments
    ]
    return ConcatDataset(datasets)
