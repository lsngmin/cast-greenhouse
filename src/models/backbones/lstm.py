"""LSTM Temporal Backbone — 논문 8-layer 구조의 Layer 3 (LSTM 변형).

────────────────────────────────────────────────────────────────────────────
역할 (한 줄)
    `(B, L, D)` embedding sequence를 LSTM으로 시간 의존성 학습.
    Output shape 동일 `(B, L, D)`.

────────────────────────────────────────────────────────────────────────────
왜 LSTM이 baseline에 필요한가?

- LSTM은 시계열 forecasting의 가장 표준적인 recurrent baseline. 본 연구의
  H1 가설 ("control 정보 추가가 예측 성능을 개선") 검증의 최소 비교 대상.
- Mamba(제안)와 Transformer가 LSTM 대비 어떤 우위를 가지는지가 H2 가설의
  핵심.
- Recurrent inductive bias가 일주기 강한 본 데이터에 합당함을 확인.

────────────────────────────────────────────────────────────────────────────
설계 결정 및 이유

- **`num_layers = 2` (default):**
    1 layer는 capacity 부족 (특히 24h lookback 학습), 3+ layer는 vanishing
    gradient + 학습 시간 증가. baseline에 적합한 깊이.

- **`hidden_size = d_model` (proj 불필요):**
    backbone 입력 dim과 출력 dim을 동일하게 유지하여 shape contract
    `(B, L, D) → (B, L, D)` 자연스럽게 만족.

- **`bidirectional = False` (default):**
    forecasting은 미래를 보면 안 됨 (causality). 입력 window 내부에서는
    기술적으로는 bidirectional 가능하지만 baseline 단순성을 위해 비활성화.
    또한 hidden size 처리가 까다로워지는 것도 회피.

- **`dropout = 0.1` (inter-layer dropout, `num_layers ≥ 2`일 때만 활성):**
    PyTorch `nn.LSTM`의 dropout은 layer 사이에만 적용됨. num_layers=1이면
    무시. baseline regularization. embedding/pooling/decoder에 dropout이
    없으므로 backbone에서만 적용 (모듈 헤더 참조).

- **`batch_first = True`:**
    Pytorch 관례. 다른 모든 모듈도 `(B, ...)` 형태라 일관성.

- **출력 projection 없음:**
    bidirectional=False면 LSTM 출력 dim = hidden_size = d_model. 별도 proj
    불필요. (bidirectional=True인 ablation에서는 `2*d_model → d_model` proj
    추가됨.)

────────────────────────────────────────────────────────────────────────────
입출력 contract

    Input:  h_in   (B, L, D)
    Output: h_out  (B, L, D)

────────────────────────────────────────────────────────────────────────────
논문 작성 참고 (Methods 섹션에 들어갈 내용)

    "The LSTM backbone consists of two stacked LSTM layers with hidden size
     128 (matching the embedding dimension d_model), inter-layer dropout
     0.1, and `batch_first` layout. Bidirectional processing is disabled to
     respect causality in forecasting. The output sequence at every
     timestep is passed to the temporal pooling layer; no output projection
     is required as the LSTM hidden size matches d_model."

    Sensitivity / Ablation 부록 후보:
    - num_layers {1, 2, 3}
    - hidden_size (= d_model) {64, 128, 256}
    - dropout rate {0.0, 0.1, 0.2}
    - bidirectional True (causality 가정 완화 시)

────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LSTMBackbone(nn.Module):
    """LSTM-based temporal backbone.

    Args:
        d_model:        embedding/hidden dim D (default 128).
        num_layers:     stacked LSTM layers (default 2).
        dropout:        inter-layer dropout, 활성 조건은 `num_layers >= 2`.
        bidirectional:  forecasting causality상 default False.

    Shape:
        in  : (B, L, D)
        out : (B, L, D)

    Parameters (default D=128, num_layers=2):
        nn.LSTM with hidden_size=128, num_layers=2 → about 264K params.
    """

    def __init__(
        self,
        d_model: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # PyTorch nn.LSTM: num_layers=1일 때 dropout 인자는 무시되고 warning 발생.
        # 명시적으로 0.0 처리하여 noise 방지. 디버깅 위해 원본·실효 둘 다 보관.
        self.dropout = dropout
        self.effective_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=self.effective_dropout,
            bidirectional=bidirectional,
        )

        # bidirectional=True면 출력이 2*d_model이라 projection 필요.
        # 본 default(False)에선 identity로 두어 cost 없음.
        if bidirectional:
            self.out_proj = nn.Linear(2 * d_model, d_model, bias=True)
        else:
            self.out_proj = nn.Identity()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (B, L, D) embedding sequence.

        Returns:
            (B, L, D) — LSTM output at every timestep.

        Raises:
            ValueError: ndim != 3 or h.shape[-1] != d_model.
        """
        # Defensive shape check (backbone interchange 시 흔한 실수)
        if h.ndim != 3:
            raise ValueError(
                f"LSTMBackbone expected shape (B, L, D), got {tuple(h.shape)}."
            )
        if h.shape[-1] != self.d_model:
            raise ValueError(
                f"LSTMBackbone expected d_model={self.d_model}, "
                f"got h.shape[-1]={h.shape[-1]}."
            )

        # output: (B, L, D) or (B, L, 2D) if bidirectional
        # (h_n, c_n) 무시 — 매 timestep output만 사용 (pooling이 처리).
        output, _ = self.lstm(h)
        output = self.out_proj(output)         # (B, L, D)
        return output

    def extra_repr(self) -> str:
        n_params = sum(p.numel() for p in self.parameters())
        return (
            f"d_model={self.d_model}, num_layers={self.num_layers}, "
            f"dropout={self.effective_dropout}, "
            f"bidirectional={self.bidirectional}, params={n_params}"
        )
