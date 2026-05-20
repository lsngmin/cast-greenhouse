"""TemporalGraphFusion — 논문 8-layer 구조의 Layer 5 (Graph 도입 후, β 패턴).

────────────────────────────────────────────────────────────────────────────
역할 (한 줄)
    Temporal backbone과 Graph module의 (B, L, D) sequence 두 개를 결합 후
    시간 축 축약하여 decoder 입력 context 생성.

────────────────────────────────────────────────────────────────────────────
설계 결정 (β 패턴, 2026-05-20 갱신)

이전 α 패턴은 graph가 (B, D) static 벡터만 출력 → fusion이 단순 가중합으로
귀결되어 architectural complexity 낮음. 본 연구의 핵심(event delayed response)
을 graph가 지원하지 못함.

β 패턴:
  graph_module이 per-timestep `(B, L, D)` 출력 →
  TemporalGraphFusion이 두 sequence를 결합 후 시간 축 축약.

  ```
  temporal_out (B, L, D)  ┐
                          ├── concat → Linear(2D→D) → LayerNorm → (B, L, D)
  graph_out    (B, L, D)  ┘                                            ↓
                                                                LastMeanPooling
                                                                       ↓
                                                                  c (B, D)
  ```

  - concat → Linear: 두 stream의 timestep-wise 가중 결합
  - LayerNorm: 결합된 sequence 정규화 (decoder 입력 안정성)
  - LastMeanPooling: 시간 축 축약 (Pool 모듈 재사용, 옵션 B 패턴)

────────────────────────────────────────────────────────────────────────────
입출력 contract

    Input:
        temporal_out:   (B, L, D)
        graph_context:  (B, L, D)
    Output:
        c:              (B, D)

────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .pooling import LastMeanPooling


class TemporalGraphFusion(nn.Module):
    """Fuse two (B, L, D) sequences into a single context vector (B, D).

    Args:
        d_model:  hidden dim D. default 128.

    Shape:
        in  : (B, L, D), (B, L, D)
        out : (B, D)
    """

    def __init__(self, d_model: int = 128):
        super().__init__()
        self.d_model = d_model
        # Step 1: concat 후 fusion
        self.combine = nn.Linear(2 * d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        # Step 2: 시간 축 축약 (Pool 재사용, 옵션 B 패턴)
        self.temporal_pool = LastMeanPooling(d_model=d_model)

    def forward(
        self,
        temporal_out: torch.Tensor,
        graph_context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            temporal_out:  (B, L, D)
            graph_context: (B, L, D)

        Returns:
            (B, D) fused context.
        """
        # Defensive shape check
        if temporal_out.ndim != 3 or temporal_out.shape[-1] != self.d_model:
            raise ValueError(
                f"temporal_out expected (B, L, {self.d_model}), "
                f"got {tuple(temporal_out.shape)}."
            )
        if graph_context.ndim != 3 or graph_context.shape[-1] != self.d_model:
            raise ValueError(
                f"graph_context expected (B, L, {self.d_model}), "
                f"got {tuple(graph_context.shape)}."
            )
        if temporal_out.shape[:2] != graph_context.shape[:2]:
            raise ValueError(
                f"(B, L) mismatch: temporal={tuple(temporal_out.shape[:2])} vs "
                f"graph={tuple(graph_context.shape[:2])}"
            )

        # Step 1: per-timestep concat + projection
        fused = torch.cat([temporal_out, graph_context], dim=-1)   # (B, L, 2D)
        fused = self.combine(fused)                                # (B, L, D)
        fused = self.norm(fused)                                   # (B, L, D)

        # Step 2: 시간 축 축약
        return self.temporal_pool(fused)                           # (B, D)

    def extra_repr(self) -> str:
        n_params = sum(p.numel() for p in self.parameters())
        return f"d_model={self.d_model}, params={n_params}"
