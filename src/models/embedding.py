"""Feature Embedding Layer — 논문 8-layer 구조의 Layer 2.

────────────────────────────────────────────────────────────────────────────
역할 (한 줄)
    Raw feature dim F를 model dim D로 projection.   X: (B, L, F) → H: (B, L, D)

────────────────────────────────────────────────────────────────────────────
구현

`SourceAwareEmbedding` (dynamic-gated, paper-default).
    각 control source (sensor/weather/actuator/setpoint/vip)에 독립 projection.
    Softmax gate가 (B, L)별 소스 가중치를 출력하여 weighted sum.

(과거에 존재했던 `FeatureEmbedding` flat baseline은 실험 결론상 source-aware
보다 우세하지 않아 코드에서 제거됨. 비교 결과는
`source_aware_embedding_results.md` 참고.)

────────────────────────────────────────────────────────────────────────────
입출력 contract

    Input:  x  (B, L, F=53) float32
    Output: h  (B, L, D=128) float32

────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# Source-aware embedding with dynamic gating
# ===========================================================================
#
# 논문 핵심 가설: 미기상 예측에서 control 소스별 중요도가 (B, L)마다 다르다.
#   - 이벤트 직후: actuator 신호 중요
#   - 정상 상태: setpoint·VIP 중요
#   - 낮·밤: weather·sensor 비중 변화
# 따라서 단순 합/concat이 아니라 **dynamic gate**로 (B, L)별 가중치를 학습한다.
#
# 구조:
#   각 source → Linear(F_i → D)  → H_i: (B, L, D)
#   ctx = mean(stack(H_i), dim=source) : (B, L, D)
#   logits = Gate(ctx) / temperature   : (B, L, n_sources)
#   alpha  = softmax(logits)           : (B, L, n_sources)
#   H = Σ alpha_i * H_i                : (B, L, D)
#   out = LayerNorm(H)                 : (B, L, D)
#
# 진단: `embed.last_alpha` 에 마지막 forward의 (B, L, n_sources) 가중치 캐싱
# (detached). 학습 끝난 뒤 시간별·이벤트별 alpha 분포를 분석해서 소스
# 기여도를 해석 가능 — 단 전체 dataset 분석에는 scripts/export_alpha.py 사용.


class SourceAwareEmbedding(nn.Module):
    """Dynamic-gated source-aware embedding (Layer 2).

    각 control source(sensor/weather/actuator/setpoint/vip)에 독립 projection.
    Softmax gate가 (B, L)별 소스 가중치를 출력하여 weighted sum.

    Args:
        feature_cols:     dataset의 feature 컬럼 이름 (X 마지막 축 순서).
        d_model:          출력 임베딩 차원. default 128.
        gate_hidden:      gate MLP 중간 dim. None이면 단일 Linear (default).
        gate_temperature: softmax temperature. default 1.0. 작을수록 hard
                          selection, 클수록 soft uniform.

    Shape:
        in:  (B, L, F)
        out: (B, L, D)

    Attributes:
        sources:         사용된 소스 이름 (정의 순서).
        source_indices:  소스 이름 → feature axis index list.
        projections:     ModuleDict — 소스별 Linear(F_i → D, bias=False).
        gate:            ctx → n_sources logits 변환.
        norm:            LayerNorm(D), weighted sum 직후 한 번.
        last_alpha:      **마지막 1개 batch**의 (B, L, n_sources) gate 값
                         (detached). forward 호출마다 덮어쓰기.
                         ⚠️ 전체 dataset 분석에 그대로 쓰면 안 됨.
                         test loader 전체에 대해 alpha를 모으려면
                         `scripts/export_alpha.py` 사용.
    """

    def __init__(
        self,
        feature_cols: Sequence[str],
        d_model: int = 128,
        gate_hidden: int | None = None,
        gate_temperature: float = 1.0,
    ):
        super().__init__()
        # Late import to avoid potential circulars
        from src.data.feature_groups import source_indices, SOURCE_ORDER

        self.d_model = d_model
        self.gate_temperature = float(gate_temperature)

        # 소스 이름 → feature axis indices.
        idx_map = source_indices(feature_cols)
        # SOURCE_ORDER 기준으로 정렬 (재현성)
        self.sources: list[str] = [s for s in SOURCE_ORDER if s in idx_map]
        if not self.sources:
            raise ValueError("No known sources matched in feature_cols. "
                             "Check feature_groups.FEATURE_SOURCES.")
        n_sources = len(self.sources)

        # source별 index 보관 (forward에서 advanced indexing)
        self.source_indices: dict[str, list[int]] = {
            s: idx_map[s] for s in self.sources
        }
        # buffer로 등록하여 to(device) 자동 이동
        for s in self.sources:
            buf = torch.tensor(self.source_indices[s], dtype=torch.long)
            self.register_buffer(f"_idx_{s}", buf, persistent=False)

        # Per-source projection (bias=False — 뒤 LayerNorm β가 흡수)
        self.projections = nn.ModuleDict({
            s: nn.Linear(len(self.source_indices[s]), d_model, bias=False)
            for s in self.sources
        })

        # Gate: ctx (D) → n_sources logits
        if gate_hidden is None:
            self.gate: nn.Module = nn.Linear(d_model, n_sources)
        else:
            self.gate = nn.Sequential(
                nn.Linear(d_model, gate_hidden),
                nn.GELU(),
                nn.Linear(gate_hidden, n_sources),
            )

        self.norm = nn.LayerNorm(d_model)

        # 진단 캐시: 마지막 forward 1개 batch의 alpha만 저장 (detached).
        # 매번 덮어쓰기 → 전체 dataset 분석에는 사용 X.
        # 전체 alpha export는 scripts/export_alpha.py 참고.
        self.last_alpha: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project per source, apply dynamic gate, weighted sum, LayerNorm.

        Args:
            x: (B, L, F) float tensor.

        Returns:
            (B, L, D) float tensor.
        """
        # Per-source projection → stack (B, L, n_sources, D)
        per_source = []
        for s in self.sources:
            idx = getattr(self, f"_idx_{s}")
            x_src = x.index_select(-1, idx)          # (B, L, F_src)
            h_src = self.projections[s](x_src)        # (B, L, D)
            per_source.append(h_src)
        H = torch.stack(per_source, dim=-2)           # (B, L, n_sources, D)

        # Gate context: source-axis mean → (B, L, D)
        ctx = H.mean(dim=-2)
        gate_logits = self.gate(ctx) / self.gate_temperature  # (B, L, n_sources)
        alpha = F.softmax(gate_logits, dim=-1)                # (B, L, n_sources)

        # Weighted sum across sources
        H_combined = (alpha.unsqueeze(-1) * H).sum(dim=-2)    # (B, L, D)

        # Cache for diagnostics (detached, no grad)
        self.last_alpha = alpha.detach()

        return self.norm(H_combined)

    def extra_repr(self) -> str:
        n_params = sum(p.numel() for p in self.parameters())
        per_src = ", ".join(
            f"{s}({len(self.source_indices[s])})" for s in self.sources
        )
        return (f"sources=[{per_src}], d_model={self.d_model}, "
                f"gate_temperature={self.gate_temperature}, params={n_params}")
