"""Trainer — train/val per epoch + test once.

────────────────────────────────────────────────────────────────────────────
역할

    fit(max_epochs):
        - 매 epoch: train_epoch → validate → scheduler.step(val_loss)
        - EarlyStopping(patience=5) 시 중단
        - BestCheckpoint: val loss 최저 시 model state 저장 → 종료 후 복원
        - history (DataFrame: epoch, train_loss, val_loss, lr) 반환

    test(test_loader):
        - 학습 끝난 뒤 한 번
        - forward all batches → (Y_true, Y_pred) tensor
        - inverse_transform 후 metric module 호출
          (MAE/RMSE/NMAE/MASE/R² + horizon별 + derived VPD)
        - dict 반환

────────────────────────────────────────────────────────────────────────────
설계 결정

- **Optimizer = AdamW (default lr=1e-3, weight_decay=1e-4):**
    Transformer/Mamba 시대 표준. SGD보다 빠른 수렴.

- **LR scheduler = ReduceLROnPlateau(mode='min', factor=0.5, patience=3):**
    val_loss가 3 epoch 정체되면 lr 절반. baseline에 안전.

- **Criterion = SmoothL1Loss(beta=0.5):**
    이상치에 MSE보다 강하고 MAE보다 학습 안정. EDA에서 outlier 처리 보류
    상태인데 어차피 음수 outlier는 NaN window drop으로 제거됨. 그래도 Huber
    가 안전.

- **Device 자동:**
    `torch.cuda.is_available()` 이면 'cuda', 아니면 'cpu'.

- **Seed control:**
    학습 시작 시 `torch.manual_seed`, `np.random.seed`, `random.seed` 모두 설정.
    DataLoader worker seed는 별도 (필요 시 `generator`로 추가 가능).

- **Best model in-memory checkpoint:**
    `BestCheckpoint`가 model.state_dict()를 RAM에 보관. fit() 종료 시 복원.
    디스크 저장 없음 (baseline엔 과함). 필요 시 외부에서 `torch.save(model.state_dict())`.

- **Test에서 inverse_transform_Y 사용:**
    raw 단위(°C, %, ppm, kPa)로 metric 계산. test_dataset에서 scaler·
    numeric_cols·target_cols 자동 추출.

────────────────────────────────────────────────────────────────────────────
사용 예

    from src.data import WindowDataset
    from src.models import ForecastingModel
    from src.train import Trainer
    from torch.utils.data import DataLoader

    ds_train = WindowDataset('Reference', 'train', 'sensor+weather+state+sp+vip')
    ds_val   = WindowDataset('Reference', 'val',   'sensor+weather+state+sp+vip')
    ds_test  = WindowDataset('Reference', 'test',  'sensor+weather+state+sp+vip')

    model = ForecastingModel.from_dataset(ds_train, 'lstm')
    trainer = Trainer(
        model=model,
        train_loader=DataLoader(ds_train, batch_size=32, shuffle=True),
        val_loader=DataLoader(ds_val, batch_size=64),
    )

    history = trainer.fit(max_epochs=50)
    test_metrics = trainer.test(ds_test)
"""
from __future__ import annotations
import random
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src import metrics as M
from .callbacks import EarlyStopping, BestCheckpoint


def set_seed(seed: int) -> None:
    """torch / numpy / random / CUDA seed 모두 설정."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _default_criterion() -> nn.Module:
    return nn.SmoothL1Loss(beta=0.5)


class Trainer:
    """Standard train/val/test loop for ForecastingModel.

    Args:
        model:               nn.Module (보통 ForecastingModel).
        train_loader:        DataLoader for training.
        val_loader:          DataLoader for validation.
        criterion:           loss function (default SmoothL1).
        optimizer_kwargs:    dict for AdamW (default lr=1e-3, weight_decay=1e-4).
        scheduler_kwargs:    dict for ReduceLROnPlateau (mode/factor/patience).
        early_stopping:      EarlyStopping 인스턴스 또는 patience 정수. None이면 비활성.
        device:              'cuda'/'cpu'. None이면 자동.
        seed:                random seed. None이면 설정 안 함.
        max_grad_norm:       gradient clipping (None이면 비활성).
        verbose:             epoch 별 print.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module | None = None,
        optimizer_kwargs: dict | None = None,
        scheduler_kwargs: dict | None = None,
        early_stopping: EarlyStopping | int | None = 5,
        device: str | torch.device | None = None,
        seed: int | None = 42,
        max_grad_norm: float | None = 1.0,
        verbose: bool = True,
    ):
        if seed is not None:
            set_seed(seed)

        self.device = torch.device(device) if device else (
            torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        )
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion or _default_criterion()
        self.max_grad_norm = max_grad_norm
        self.verbose = verbose

        opt_kw = {'lr': 1e-3, 'weight_decay': 1e-4, **(optimizer_kwargs or {})}
        self.optimizer = torch.optim.AdamW(self.model.parameters(), **opt_kw)

        sch_kw = {'mode': 'min', 'factor': 0.5, 'patience': 3,
                  **(scheduler_kwargs or {})}
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, **sch_kw,
        )

        # Callbacks
        if isinstance(early_stopping, int):
            self.early_stopping = EarlyStopping(patience=early_stopping, mode='min')
        elif early_stopping is None:
            self.early_stopping = None
        else:
            self.early_stopping = early_stopping
        self.checkpoint = BestCheckpoint(mode='min')

    # -----------------------------------------------------------------------
    # Training / validation loops

    def train_epoch(self) -> float:
        """Run one training epoch. Returns mean batch loss."""
        self.model.train()
        total_loss, total_n = 0.0, 0
        for x, y in self.train_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            y_hat = self.model(x)
            loss = self.criterion(y_hat, y)
            loss.backward()
            if self.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            bs = x.shape[0]
            total_loss += loss.item() * bs
            total_n += bs
        return total_loss / max(total_n, 1)

    @torch.no_grad()
    def validate(self) -> float:
        """Run one validation pass. Returns mean batch loss."""
        self.model.eval()
        total_loss, total_n = 0.0, 0
        for x, y in self.val_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            y_hat = self.model(x)
            loss = self.criterion(y_hat, y)
            bs = x.shape[0]
            total_loss += loss.item() * bs
            total_n += bs
        return total_loss / max(total_n, 1)

    # -----------------------------------------------------------------------
    # Full fit

    def fit(self, max_epochs: int = 50) -> pd.DataFrame:
        """Train for up to max_epochs. Restores best (val_loss) state on exit.

        Returns
        -------
        history : DataFrame with columns [epoch, train_loss, val_loss, lr, time_s]
        """
        rows = []
        for epoch in range(1, max_epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch()
            val_loss = self.validate()
            self.scheduler.step(val_loss)
            improved = self.checkpoint(val_loss, self.model, epoch=epoch)
            elapsed = time.time() - t0
            lr_now = self.optimizer.param_groups[0]['lr']

            row = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'lr': lr_now,
                'time_s': elapsed,
                'best': improved,
            }
            rows.append(row)
            if self.verbose:
                star = ' ★' if improved else ''
                print(f"epoch {epoch:>3d}  train={train_loss:.4f}  "
                      f"val={val_loss:.4f}  lr={lr_now:.2e}  "
                      f"({elapsed:.1f}s){star}")

            if self.early_stopping is not None and self.early_stopping(val_loss):
                if self.verbose:
                    print(f"  → EarlyStopping triggered (best epoch "
                          f"{self.checkpoint.best_epoch}, val={self.checkpoint.best:.4f})")
                break

        # Restore best checkpoint
        if self.checkpoint.best_state is not None:
            self.checkpoint.restore(self.model)

        return pd.DataFrame(rows)

    # -----------------------------------------------------------------------
    # Test (one-shot, with full metrics)

    @torch.no_grad()
    def test(
        self,
        test_dataset: Dataset,
        batch_size: int = 64,
    ) -> dict[str, Any]:
        """Evaluate best model on test dataset with full metric suite (원 단위).

        Args:
            test_dataset: WindowDataset (scaler / numeric_cols / target_cols attribute 필요).
            batch_size:   inference batch size.

        Returns:
            dict with:
              - per-target scalar metrics (원 단위):
                  `mae_<name>`, `rmse_<name>`, `r2_<name>`,
                  `nmae_<name>`, `nrmse_<name>`, `mase_<name>`
              - 'horizon_summary': DataFrame per (variable, horizon, kind)
                  with columns MAE/RMSE/R2/MASE for {1h, 6h, 24h} × {point, cumulative}
              - 'vpd_mae', 'vpd_rmse' (derived from predicted Tair/Rhair)
              - 'naive_scale': MASE 분모 (per target)
              - 'best_epoch', 'best_val_loss' (from fit())
        """
        self.model.eval()
        loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        all_y_true, all_y_pred, all_y_persist = [], [], []
        feature_cols = list(test_dataset.feature_cols)
        target_cols = list(test_dataset.target_cols)
        target_feature_idx = [feature_cols.index(c) for c in target_cols]
        for x, y in loader:
            persist = x[:, -1, target_feature_idx].numpy()
            x = x.to(self.device)
            y_hat = self.model(x).cpu().numpy()
            all_y_true.append(y.numpy())
            all_y_pred.append(y_hat)
            all_y_persist.append(
                M.persistence_predict(persist, horizon=test_dataset.horizon)
            )

        y_true_scaled = np.concatenate(all_y_true, axis=0)   # (n, H, V)
        y_pred_scaled = np.concatenate(all_y_pred, axis=0)
        y_persist_scaled = np.concatenate(all_y_persist, axis=0)

        # Inverse transform to raw units
        scaler = test_dataset.scaler
        numeric_cols = test_dataset.numeric_cols
        y_true = M.inverse_transform_Y(y_true_scaled, scaler, target_cols, numeric_cols)
        y_pred = M.inverse_transform_Y(y_pred_scaled, scaler, target_cols, numeric_cols)
        y_persist = M.inverse_transform_Y(
            y_persist_scaled, scaler, target_cols, numeric_cols
        )

        # Horizon summary (per variable × {1h, 6h, 24h} × {point, cumulative})
        # Naive MAE scale: 학습 시점에 train DataFrame이 필요하지만 여기선
        # test의 raw Y에서 adjacent-step diff로 계산 (MASE는 보조 정보로만 해석).
        var_std = y_true.reshape(-1, len(target_cols)).std(axis=0)
        # 1-step adjacent 차분 (각 sample의 trajectory 내부에서)
        naive_diff = np.abs(np.diff(y_true, axis=1))   # (n, H-1, V)
        naive_scale = (
            np.nanmean(naive_diff.reshape(-1, len(target_cols)), axis=0)
            if naive_diff.size else var_std
        )

        summary = M.horizon_summary(
            y_true, y_pred, target_cols, naive_mae_scale=naive_scale,
        )
        persistence_summary = M.horizon_summary(
            y_true, y_persist, target_cols, naive_mae_scale=naive_scale,
        )

        # Aggregate scalar metrics (전체 horizon × samples)
        results: dict[str, Any] = {
            'horizon_summary': summary,
            'persistence_horizon_summary': persistence_summary,
            'best_epoch': self.checkpoint.best_epoch,
            'best_val_loss': self.checkpoint.best,
            'naive_scale': dict(zip(target_cols, [float(v) for v in naive_scale])),
        }
        for v_i, name in enumerate(target_cols):
            yt = y_true[:, :, v_i]
            yp = y_pred[:, :, v_i]
            yn = y_persist[:, :, v_i]
            model_mae = M.mae(yt, yp)
            persistence_mae = M.mae(yt, yn)
            results[f'mae_{name}']   = float(M.mae(yt, yp))
            results[f'rmse_{name}']  = float(M.rmse(yt, yp))
            results[f'r2_{name}']    = float(M.r2(yt, yp))
            results[f'nmae_{name}']  = float(M.nmae(yt, yp, var_std[v_i]))
            results[f'nrmse_{name}'] = float(M.nrmse(yt, yp, var_std[v_i]))
            results[f'mase_{name}']  = float(M.mase(yt, yp, naive_scale[v_i]))
            results[f'persistence_mae_{name}'] = float(persistence_mae)
            results[f'persistence_rmse_{name}'] = float(M.rmse(yt, yn))
            results[f'persistence_r2_{name}'] = float(M.r2(yt, yn))
            results[f'relative_mae_{name}'] = float(
                model_mae / max(persistence_mae, 1e-12)
            )

        # Derived VPD
        vpd_mae, _, _ = M.vpd_metric_from_targets(y_true, y_pred, target_cols, M.mae)
        vpd_rmse, _, _ = M.vpd_metric_from_targets(y_true, y_pred, target_cols, M.rmse)
        persist_vpd_mae, _, _ = M.vpd_metric_from_targets(
            y_true, y_persist, target_cols, M.mae
        )
        persist_vpd_rmse, _, _ = M.vpd_metric_from_targets(
            y_true, y_persist, target_cols, M.rmse
        )
        results['vpd_mae'] = float(vpd_mae)
        results['vpd_rmse'] = float(vpd_rmse)
        results['persistence_vpd_mae'] = float(persist_vpd_mae)
        results['persistence_vpd_rmse'] = float(persist_vpd_rmse)
        results['relative_mae_VPD'] = float(vpd_mae / max(persist_vpd_mae, 1e-12))

        return results
