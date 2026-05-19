"""Data subpackage — 논문 8-layer 구조의 Layer 1 (Input Layer) 구현.

3개 sub-module이 책임을 분담:
    preprocessing.py    — raw csv → cleaned parquet (clip, VPD, change flags,
                          weather merge, train/val/test split, scaler fit)
    windowing.py        — parquet → sliding window tensor (X, Y, meta)
    feature_groups.py   — feature group 정의 (sensor → +vip 5단계 ablation)
    window_dataset.py   — 위 3개를 묶은 PyTorch Dataset (학습/추론 통일 인터페이스)

학습 코드에서:
    from src.data import WindowDataset
    ds = WindowDataset('Reference', 'train', 'sensor+weather+state+sp+vip')
    loader = DataLoader(ds, batch_size=32, shuffle=True)
"""
from . import preprocessing, windowing, feature_groups
from .window_dataset import WindowDataset

__all__ = ['preprocessing', 'windowing', 'feature_groups', 'WindowDataset']
