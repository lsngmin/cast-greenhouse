"""TemporalGraphFusion — 논문 8-layer 구조의 Layer 5 (Graph 도입 후).

────────────────────────────────────────────────────────────────────────────
역할 (한 줄)
    Temporal backbone 출력과 Graph module 출력을 결합하여 decoder 입력 context
    생성.   (temporal_out, graph_context) → c

────────────────────────────────────────────────────────────────────────────
설계 결정 (model_architecture.md 옵션 B 채택)

- **Pool과 Fusion 분리, Fusion이 Pool을 내부에서 재사용:**
    Baseline (graph 없음): `LastMeanPooling` 단독 사용.
    Graph 도입 시: `TemporalGraphFusion`이 `LastMeanPooling`을 contain하여
    내부에서 temporal pool 후 graph context와 concat → Linear → LayerNorm.

- **`concat → Linear(2D, D) → LayerNorm`:**
    Temporal과 graph가 서로 다른 정보를 담는다고 가정. concat이 비중을 학습
    가능 (gating 같은 변형도 가능하지만 baseline엔 단순 linear).

- **Linear bias = False:**
    LayerNorm β가 bias 역할. `FeatureEmbedding`·`LastMeanPooling`과 일관.

────────────────────────────────────────────────────────────────────────────
입출력 contract

    Input:
        temporal_out:    (B, L, D)  from temporal backbone
        graph_context:   (B, D)     from DirectedGraphModule
    Output:
        c:               (B, D)     decoder 입력 context

────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .pooling import LastMeanPooling


class TemporalGraphFusion(nn.Module):
    """Fuse temporal pool output and graph context into single context vector.

    구조:
        temporal_out (B, L, D)
            ↓ LastMeanPooling
        temporal_context (B, D)
        graph_context (B, D)
            ↓ concat → (B, 2D)
            ↓ Linear(2D → D, bias=False)
            ↓ LayerNorm(D)
        c (B, D)

    Args:
        d_model:  hidden dim D. default 128.

    Shape:
        in  : (B, L, D), (B, D)
        out : (B, D)
    """

    def __init__(self, d_model: int = 128):
        super().__init__()
        self.d_model = d_model
        # Pool 재사용 (옵션 B 패턴)
        self.temporal_pool = LastMeanPooling(d_model=d_model)
        # Fusion
        self.combine = nn.Linear(2 * d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        temporal_out: torch.Tensor,
        graph_context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            temporal_out:  (B, L, D)
            graph_context: (B, D)

        Returns:
            (B, D) fused context.
        """
        if temporal_out.ndim != 3 or temporal_out.shape[-1] != self.d_model:
            raise ValueError(
                f"temporal_out expected (B, L, {self.d_model}), got {tuple(temporal_out.shape)}."
            )
        if graph_context.ndim != 2 or graph_context.shape[-1] != self.d_model:
            raise ValueError(
                f"graph_context expected (B, {self.d_model}), got {tuple(graph_context.shape)}."
            )
        if temporal_out.shape[0] != graph_context.shape[0]:
            raise ValueError(
                f"batch size mismatch: temporal={temporal_out.shape[0]} vs "
                f"graph={graph_context.shape[0]}"
            )

        t = self.temporal_pool(temporal_out)            # (B, D)
        x = torch.cat([t, graph_context], dim=-1)       # (B, 2D)
        return self.norm(self.combine(x))               # (B, D)

    def extra_repr(self) -> str:
        n_params = sum(p.numel() for p in self.parameters())
        return f"d_model={self.d_model}, params={n_params}"
