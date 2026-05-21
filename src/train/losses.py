"""Loss functions for greenhouse trajectory forecasting."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EventWeightedSmoothL1Loss(nn.Module):
    """SmoothL1 loss with optional per-horizon event weights.

    This is not a new base loss. It applies a temporal weight matrix to the
    standard SmoothL1 error so post-control-event horizons can be emphasized.

    Args:
        beta: SmoothL1 beta passed to PyTorch.
        normalize_by_weight: If True, compute a weighted mean by dividing by
            the sum of weights. This keeps the loss scale comparable across
            batches with different numbers of event horizons.

    Shape:
        y_hat, y:     (B, H, V)
        event_weight: (B, H) or (B, H, 1), values >= 1.0
    """

    supports_event_weight = True

    def __init__(self, beta: float = 0.5, normalize_by_weight: bool = True):
        super().__init__()
        self.beta = beta
        self.normalize_by_weight = normalize_by_weight

    def forward(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
        event_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        loss = F.smooth_l1_loss(y_hat, y, beta=self.beta, reduction="none")
        if event_weight is None:
            return loss.mean()

        weight = event_weight.to(device=loss.device, dtype=loss.dtype)
        if weight.ndim == 2:
            weight = weight.unsqueeze(-1)
        if weight.ndim != 3:
            raise ValueError(
                "event_weight must have shape (B, H) or (B, H, 1), "
                f"got {tuple(event_weight.shape)}."
            )
        if weight.shape[0] != loss.shape[0] or weight.shape[1] != loss.shape[1]:
            raise ValueError(
                f"event_weight shape {tuple(event_weight.shape)} is not "
                f"compatible with loss shape {tuple(loss.shape)}."
            )

        weighted = loss * weight
        if self.normalize_by_weight:
            denom = weight.expand_as(loss).sum().clamp_min(1.0)
            return weighted.sum() / denom
        return weighted.mean()

    def extra_repr(self) -> str:
        return (
            f"beta={self.beta}, "
            f"normalize_by_weight={self.normalize_by_weight}"
        )
