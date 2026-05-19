"""Transformer Temporal Backbone — 논문 8-layer 구조의 Layer 3 (Transformer 변형).

────────────────────────────────────────────────────────────────────────────
역할 (한 줄)
    `(B, L, D)` embedding sequence를 vanilla Transformer encoder로 시간 의존성 학습.
    Output shape 동일 `(B, L, D)`.

────────────────────────────────────────────────────────────────────────────
설계 결정 및 이유

- **Vanilla Transformer (Vaswani 2017) — encoder only:**
    PatchTST/iTransformer 같은 시계열 특화 변형은 baseline 비교 단순성을
    해침. encoder-only 표준 구조로 LSTM/Mamba와 capacity·결과 해석을 통일.

- **`nhead = 4` (default):**
    d_model=128 / nhead=4 = head_dim 32. transformer 관례 (32~64 head_dim).
    nhead=8은 head_dim 16으로 작아져 attention 표현력 저하 위험.

- **`num_layers = 2` (default):**
    LSTM과 capacity 맞춤 (LSTM 264K vs Transformer 약 265K). baseline에 적합.

- **`dim_feedforward = 256` (= 2·d_model):**
    표준 4·d_model보다 작게. 작은 dataset (Reference train 2,481 windows)에서
    overfitting 줄이고 param count 절감.

- **`norm_first = True` (Pre-LN):**
    Pre-LN이 Post-LN보다 학습 안정. warmup 없이도 수렴. SOTA 시계열 모델
    (PatchTST, iTransformer)도 Pre-LN 채택.

- **`activation = gelu`:**
    Original ReLU 대비 부드러운 saturation. Pre-LN과 잘 맞음.

- **`batch_first = True`:**
    다른 모듈과 일관 ((B, L, D) 형태). PyTorch 2.x에서도 안정 지원.

- **Sinusoidal positional encoding (학습 X):**
    learnable PE는 lookback 길이가 가변일 때 (sensitivity 6h/12h/48h) 불편.
    sinusoidal은 어떤 L이든 동일 공식. baseline 단순성 + 일반성.

- **Causality?**
    본 연구 입력 X는 lookback (관측된 과거)만 포함. forecast horizon은 decoder
    출력에만 존재. 따라서 backbone 내부에서 양방향 self-attention해도 미래
    누설 없음. causal mask 불필요. LSTM의 `bidirectional=False`는 RNN baseline
    의 표준 관례일 뿐, 정보 관점에서는 Transformer와 동일.

────────────────────────────────────────────────────────────────────────────
입출력 contract

    Input:  h_in   (B, L, D)
    Output: h_out  (B, L, D)

────────────────────────────────────────────────────────────────────────────
논문 작성 참고 (Methods 섹션)

    "The Transformer backbone follows the standard encoder-only architecture
     (Vaswani et al., 2017) with sinusoidal positional encoding, two
     Transformer encoder layers (4 attention heads, feedforward dimension
     256), Pre-LayerNorm (norm_first=True) and GELU activation. As all input
     features lie within the observed lookback window, no causal mask is
     applied — the encoder attends bidirectionally within the lookback,
     analogous to BERT-style encoders. Hidden dimension matches d_model=128,
     keeping the parameter budget comparable to the LSTM backbone (~265K
     vs ~264K)."

    Sensitivity / Ablation 부록 후보:
    - num_layers {1, 2, 4}
    - nhead {2, 4, 8}
    - dim_feedforward {128, 256, 512}
    - Pre-LN vs Post-LN
    - learnable vs sinusoidal PE

────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations
import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani 2017).

    Args:
        d_model: embedding dim.
        max_len: 최대 sequence 길이. lookback sensitivity (max 48h=576 step)
                 + 여유.

    Shape:
        in  : (B, L, D)
        out : (B, L, D)  — input + PE[:L]
    """

    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even for sinusoidal PE, got {d_model}")
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # buffer: 학습 안 됨, device 이동에 자동 따라옴
        self.register_buffer('pe', pe.unsqueeze(0))      # (1, max_len, D)
        self.d_model = d_model
        self.max_len = max_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.size(1)
        if L > self.max_len:
            raise ValueError(
                f"Input length {L} exceeds positional encoding max_len {self.max_len}."
            )
        return x + self.pe[:, :L, :]


class TransformerBackbone(nn.Module):
    """Vanilla Transformer encoder backbone (Pre-LN, GELU).

    Args:
        d_model:          hidden dim D. default 128.
        nhead:            attention head 수. default 4.
        num_layers:       stacked encoder layers. default 2.
        dim_feedforward:  FFN inner dim. default 256.
        dropout:          attention + FFN dropout. default 0.1.
        max_len:          PE 지원 최대 L. default 1024 (48h lookback 576 < 1024).

    Shape:
        in  : (B, L, D)
        out : (B, L, D)
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_len: int = 1024,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        self.pos = SinusoidalPositionalEncoding(d_model, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,           # Pre-LN
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (B, L, D) embedding sequence.

        Returns:
            (B, L, D) — Transformer encoder output.

        Raises:
            ValueError: ndim != 3 or h.shape[-1] != d_model.
        """
        if h.ndim != 3:
            raise ValueError(
                f"TransformerBackbone expected shape (B, L, D), got {tuple(h.shape)}."
            )
        if h.shape[-1] != self.d_model:
            raise ValueError(
                f"TransformerBackbone expected d_model={self.d_model}, "
                f"got h.shape[-1]={h.shape[-1]}."
            )
        h = self.pos(h)                                  # + PE
        return self.encoder(h)                           # bidirectional self-attention

    def extra_repr(self) -> str:
        n_params = sum(p.numel() for p in self.parameters())
        return (
            f"d_model={self.d_model}, nhead={self.nhead}, "
            f"num_layers={self.num_layers}, dim_feedforward={self.dim_feedforward}, "
            f"dropout={self.dropout}, params={n_params}"
        )
