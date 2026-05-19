"""Training callbacks — EarlyStopping, BestCheckpoint.

Trainer가 매 epoch 끝에 호출하여 학습 흐름 제어.
외부 의존성 없음 (PyTorch만 사용).
"""
from __future__ import annotations
import copy
import torch.nn as nn


# ---------------------------------------------------------------------------
# Early stopping

class EarlyStopping:
    """Val metric이 patience 동안 개선되지 않으면 stop signal.

    Args:
        patience:    개선 없이 견딜 epoch 수.
        min_delta:   개선으로 인정할 최소 변화량 (절댓값).
        mode:        'min' (loss류) or 'max' (R²류). default 'min'.

    Usage:
        es = EarlyStopping(patience=5)
        for epoch in range(max_epochs):
            ...
            if es(val_loss):
                break       # stop
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.0,
                 mode: str = 'min'):
        if mode not in ('min', 'max'):
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best: float | None = None
        self.bad_epochs = 0
        self.stop = False

    def _is_improved(self, current: float) -> bool:
        if self.best is None:
            return True
        if self.mode == 'min':
            return current < self.best - self.min_delta
        return current > self.best + self.min_delta

    def __call__(self, current: float) -> bool:
        """Returns True if training should stop."""
        if self._is_improved(current):
            self.best = current
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience:
                self.stop = True
        return self.stop


# ---------------------------------------------------------------------------
# Best checkpoint (in-memory)

class BestCheckpoint:
    """Val metric이 가장 좋을 때 model state_dict를 in-memory에 저장.

    Trainer.fit() 종료 시 `restore(model)`로 best state 복원.

    Args:
        mode: 'min' (loss) or 'max' (R²).

    Usage:
        ckpt = BestCheckpoint(mode='min')
        for epoch in ...:
            ...
            improved = ckpt(val_loss, model)
        ckpt.restore(model)    # 학습 끝나면 best state 복원
    """

    def __init__(self, mode: str = 'min'):
        if mode not in ('min', 'max'):
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")
        self.mode = mode
        self.best: float | None = None
        self.best_state: dict | None = None
        self.best_epoch: int | None = None

    def _is_improved(self, current: float) -> bool:
        if self.best is None:
            return True
        if self.mode == 'min':
            return current < self.best
        return current > self.best

    def __call__(self, current: float, model: nn.Module,
                 epoch: int | None = None) -> bool:
        """Returns True if model state was saved."""
        if self._is_improved(current):
            self.best = current
            self.best_state = {k: v.detach().clone()
                               for k, v in model.state_dict().items()}
            self.best_epoch = epoch
            return True
        return False

    def restore(self, model: nn.Module) -> None:
        if self.best_state is None:
            raise RuntimeError("No checkpoint to restore — checkpoint never called.")
        model.load_state_dict(self.best_state)
