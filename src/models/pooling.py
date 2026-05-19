"""Temporal Pooling Layer — 논문 8-layer 구조의 Layer 5 (baseline 시점).

────────────────────────────────────────────────────────────────────────────
역할 (한 줄)
    Backbone 출력의 시간(lookback)차원 L을 축약.   H: (B, L, D) → C: (B, D)

────────────────────────────────────────────────────────────────────────────
왜 필요한가?

Backbone(LSTM/Transformer/Mamba)은 매 timestep마다 hidden state 벡터를 출력
함 (B, L, D). 그러나 Trajectory Decoder는 단일 context 벡터 (B, D)를 입력으로
받아 미래 24h trajectory를 direct multi-step으로 출력한다. 따라서 L 차원을
하나의 벡터로 축약하는 단계가 필요하며, 이 layer를 Pool이라 부른다.

본 연구는 `LastMeanPooling` 한 가지만 사용한다.

────────────────────────────────────────────────────────────────────────────
설계 결정 및 이유

- **`LastMeanPooling` 채택 (last + mean concat → Linear → LayerNorm):**
    last : `h[:, -1, :]` — forecast 직전 timestep의 backbone hidden state.
           최근 상태(short-horizon에 중요).
    mean : `h.mean(dim=1)` — **lookback-wide hidden representation summary**.
           (raw sensor의 하루 평균이 아니라, backbone을 거친 후의 hidden
            state 평균이다. EDA에서 확인된 강한 일주기 구조 [ACF@24h ≈ 0.92,
            STL seasonal ≈ 0.92]가 backbone hidden state에 반영되어 있다면,
            mean이 그 diurnal context를 retain하는 데 기여할 수 있다.
            단, 일주기 보존을 단정할 수는 없음 — backbone representation에
            따라 달라짐.)
    last + mean concat → Linear → LayerNorm:
           두 신호의 비중을 데이터에서 학습 + post-pooling normalization으로
           decoder 입력 안정성 확보.

- **Post-pooling LayerNorm 추가 (recommended for backbone comparison):**
    LSTM / Transformer / Mamba는 출력 scale·분포가 다를 수 있다 (예: LSTM은
    tanh-bounded, Transformer는 unbounded, Mamba는 SSM-specific). 이 차이가
    decoder 입력으로 그대로 흘러가면 backbone별 성능 차이가 "표현력" 외에도
    "출력 scale 정합성" 영향을 받을 위험. LayerNorm으로 decoder 입력 분포를
    안정화하여 공정 비교 신뢰성을 높임.

- **Linear bias = False:**
    바로 뒤 LayerNorm의 affine `β` (bias-equivalent) 가 동일 역할을 하므로
    redundant. `FeatureEmbedding`과 동일한 결정 패턴.

- **활성화 함수 없음 (linear aggregation):**
    Pool 출력은 decoder 입력으로 직행. 여기서 ReLU/GELU를 넣을 이유 없음.
    pooling을 단순 linear aggregation step으로 유지하고, 비선형성은 backbone
    과 decoder에 맡김.

- **Max / Attention / CLS pooling 미채택:**
    Max는 노이즈 민감. Attention pool은 파라미터 폭증 + Transformer 계열에
    유리하게 작용하여 backbone 간 공정 비교 흐림. CLS token은 backbone마다
    적용 가능성 다름. 모두 baseline 공정성을 해치므로 제외.

- **`LastPooling` 미제공:**
    "mean 추가 효과 ablation"이 매력적이지만 본 실험에서 돌리지 않을 예정
    이므로 dead code 회피.

────────────────────────────────────────────────────────────────────────────
입출력 contract

    Input:  h  (B, L, D)
    Output: c  (B, D)

────────────────────────────────────────────────────────────────────────────
구조적 주의 — single context bottleneck

이 pooling은 `(B, L, D)` 전체를 `(B, D)` 하나로 압축한다. 이후 decoder가 이
context 벡터 하나로 미래 24h trajectory `(B, 288, 3)`를 생성한다.

baseline 단계엔 합리적 선택:
  - 모든 backbone에 공통 적용 → 비교 공정성 확보
  - 단순한 구조 → 실험 해석이 backbone-centric

단, event timing이나 정밀한 response curve 추적에는 병목이 될 수 있음.
event-window evaluation 결과가 약하게 나오면 후속 개선 후보로 검토:
  - Sequence-to-sequence decoder (cross-attention)
  - Temporal attention pooling
  - Horizon-wise / patch-based decoder

현재 논문 기여는 decoder가 아니라 control-aware input + event-based
evaluation이므로, baseline에서는 단순 pooling이 오히려 해석에 유리.

────────────────────────────────────────────────────────────────────────────
Fusion으로의 확장 (Step 4 Graph 도입 후)

본 layer는 **단일 stream의 timestep 축약**만 담당한다. 향후 Step 4
(Directed Graph Module)가 도입되면 graph stream이 추가되어 multi-stream
fusion이 필요해진다. 그때는 별도 `TemporalGraphFusion` 모듈을 만들고,
**내부에서 이 Pool 모듈을 재사용**한다 (composition).

    예시 (future):
        class TemporalGraphFusion(nn.Module):
            def __init__(self, d_model):
                self.temporal_pool = LastMeanPooling(d_model)  # 재사용
                self.graph_branch  = ...
                self.combine = nn.Linear(2*d_model, d_model)

이렇게 함으로써 baseline 단계에서는 Pool에만 집중하고, graph 도입 시 graph
출력 shape이 결정된 후 fusion을 깔끔히 작성할 수 있다.

────────────────────────────────────────────────────────────────────────────
논문 작성 참고 (Methods 섹션에 들어갈 내용)

    "Temporal pooling aggregates the lookback-length backbone output
     (B, L, D) into a single context vector (B, D) for the trajectory
     decoder. We adopt LastMeanPooling, which concatenates the last
     timestep hidden state with the mean over the entire lookback window,
     projects via a single linear layer, and normalizes via LayerNorm.
     The mean component provides a global summary over the lookback
     window, which can help retain information from the observed diurnal
     context (the temporal-structure analysis showed ACF at lag 24h ≈ 0.92
     for Tair and STL seasonal strength ≈ 0.92), while the last-timestep
     component preserves the most recent state for short-horizon accuracy.
     No activation is applied to keep pooling as a simple linear
     aggregation step; non-linearity is left to the backbone and decoder.
     The post-pooling LayerNorm stabilizes the decoder input distribution
     across backbones (LSTM, Transformer, Mamba) whose output scales may
     differ."

────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LastMeanPooling(nn.Module):
    """Pool backbone output over time using last-state and lookback summary.

    구조:
        last = h[:, -1, :]                    # (B, D)  최근 상태
        mean = h.mean(dim=1)                  # (B, D)  lookback-wide hidden
                                              #         representation summary
        cat  = concat([last, mean], -1)       # (B, 2D)
        proj = Linear(2D → D, bias=False)     # (B, D)
        norm = LayerNorm(D)                   # (B, D)  decoder 입력 안정화

    Args:
        d_model: backbone hidden dim D. 출력도 같은 D로 유지.

    Shape:
        in  : (B, L, D)
        out : (B, D)

    Parameters: `2D × D + 2D` (Linear weight + LayerNorm γ,β; Linear bias 없음).
    """

    def __init__(self, d_model: int = 128):
        super().__init__()
        # bias=False: LayerNorm affine β가 bias-equivalent 역할.
        self.proj = nn.Linear(2 * d_model, d_model, bias=False)
        # Decoder 입력 분포 안정화 (backbone별 출력 scale 차이 흡수).
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (B, L, D) backbone hidden states.

        Returns:
            (B, D) context vector for decoder.

        Raises:
            ValueError: ndim != 3 or h.shape[-1] != d_model.
        """
        # Defensive shape check — backbone 출력 dim 불일치 흔한 실수
        if h.ndim != 3:
            raise ValueError(
                f"LastMeanPooling expected shape (B, L, D), got {tuple(h.shape)}."
            )
        if h.shape[-1] != self.d_model:
            raise ValueError(
                f"LastMeanPooling expected d_model={self.d_model}, "
                f"got h.shape[-1]={h.shape[-1]}. "
                f"backbone 출력 dim과 d_model이 일치하는지 확인."
            )

        last = h[:, -1, :]                              # (B, D)
        mean = h.mean(dim=1)                            # (B, D) lookback-wide
                                                        #        representation
                                                        #        summary
        c = torch.cat([last, mean], dim=-1)             # (B, 2D)
        c = self.proj(c)                                # (B, D)
        c = self.norm(c)                                # (B, D)
        return c

    def extra_repr(self) -> str:
        n_params = sum(p.numel() for p in self.parameters())
        return f"d_model={self.d_model}, params={n_params}"
