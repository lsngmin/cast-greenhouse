"""Mamba Temporal Backbone — 논문 8-layer 구조의 Layer 3 (Mamba 변형, 제안 backbone).

────────────────────────────────────────────────────────────────────────────
역할 (한 줄)
    `(B, L, D)` embedding sequence를 Mamba (Selective State Space Model)로
    시간 의존성 학습. Output shape 동일 `(B, L, D)`.

────────────────────────────────────────────────────────────────────────────
왜 Mamba?

본 연구의 **제안 backbone**. 다음 이유로 LSTM/Transformer 대비 우위 가능성:

1. **장기 의존성 + 선형 복잡도:**
    Transformer self-attention은 O(L²) 메모리. Mamba SSM은 O(L). 24h lookback
    (L=288), 48h sensitivity (L=576)에서 시간 효율적.

2. **Selective scan:**
    Mamba는 state space parameter를 input-dependent하게 만들어 RNN의
    information bottleneck을 완화. 일주기 + event 직후 변동 모두 잘 추적
    기대.

3. **시계열 forecasting SOTA:**
    2024 이후 long-horizon forecasting에서 PatchTST/iTransformer와 경쟁
    (Time-Series Mamba, S-Mamba 등).

────────────────────────────────────────────────────────────────────────────
설계 결정 및 이유

- **`mamba_ssm` 패키지 의존:**
    공식 reference 구현 (Gu & Dao 2023). pure PyTorch implementation은
    참고용으로 부정확. CUDA kernel 호출 → **로컬 Windows CPU 환경에서는
    설치 어려움. Linux + CUDA 12.x + PyTorch 2.x 환경에서 동작.**

- **Pre-LN residual block 구조 (Transformer와 패턴 일관):**
    ```
    h ← h + Dropout(Mamba(LayerNorm(h)))
    ```
    공정 비교: Transformer의 Pre-LN MHA block과 구조적으로 동일한 residual
    패턴. 학습 안정성 + capacity 비교 용이.

- **`num_layers = 2` (default):**
    LSTM/Transformer와 capacity 맞춤.

- **Mamba 내부 hyperparameter (default):**
    - `d_state = 16`  — SSM 상태 차원. Mamba 논문 권장.
    - `d_conv = 4`    — short-range conv1d kernel. local convolution.
    - `expand = 2`    — internal projection expansion (`d_inner = expand × d_model`).
    Mamba 논문 default.

- **`dropout = 0.1`:**
    residual block 사이. LSTM/Transformer와 동일 수준.

- **출력 projection 없음:**
    Mamba block은 (B, L, D) → (B, L, D) (input dim = output dim 자동). 별도
    proj 불필요.

────────────────────────────────────────────────────────────────────────────
입출력 contract

    Input:  h_in   (B, L, D)
    Output: h_out  (B, L, D)

────────────────────────────────────────────────────────────────────────────
환경 요구사항

    pip install mamba-ssm causal-conv1d
    - Linux + CUDA 12.x 권장
    - PyTorch 2.x

    Windows + CPU 환경에서는 설치 실패. Trainer에서 Mamba 미사용 시 영향
    없음 (lazy import — `build_backbone('mamba')` 호출 시점에만 import 시도).

────────────────────────────────────────────────────────────────────────────
논문 작성 참고 (Methods 섹션)

    "The Mamba backbone uses two Pre-LayerNorm residual blocks, each wrapping
     a Mamba selective state-space module (Gu & Dao, 2023) with state
     dimension 16, convolution kernel 4, and internal expansion 2. Hidden
     dimension is fixed to d_model=128, matching LSTM and Transformer
     baselines. Inter-block dropout is 0.1. Mamba's linear-time selective
     scan provides an efficient alternative to attention's O(L²) cost,
     particularly relevant for the 24-hour (L=288) and longer sensitivity
     lookbacks evaluated in this study."

    Sensitivity / Ablation 부록 후보:
    - num_layers {1, 2, 4}
    - d_state {8, 16, 32}
    - expand {1, 2, 4}
    - lookback sensitivity (Mamba가 LSTM·Transformer 대비 long L에서 우위 확인)

────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _MambaResidualBlock(nn.Module):
    """Pre-LN residual wrapper around a single Mamba module.

    `h ← h + Dropout(Mamba(LayerNorm(h)))`
    """

    def __init__(self, d_model: int, d_state: int, d_conv: int,
                 expand: int, dropout: float):
        super().__init__()
        # Late import — mamba_ssm 미설치 환경에서 backbones 모듈 import만으로
        # 에러가 발생하지 않게.
        from mamba_ssm import Mamba

        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.dropout(self.mamba(self.norm(h)))


class MambaBackbone(nn.Module):
    """Mamba-based temporal backbone (제안 backbone).

    Args:
        d_model:     hidden dim D. default 128.
        num_layers:  stacked residual blocks. default 2.
        d_state:     SSM state dim. default 16.
        d_conv:      Mamba internal conv1d kernel. default 4.
        expand:      Mamba internal expansion ratio. default 2.
        dropout:     residual block dropout. default 0.1.

    Shape:
        in  : (B, L, D)
        out : (B, L, D)

    환경:
        mamba-ssm + causal-conv1d 설치 + CUDA. Windows CPU 미지원.
    """

    def __init__(
        self,
        d_model: int = 128,
        num_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        # __init__ 호출 시 mamba_ssm import 시도 → 미설치면 명확한 에러.
        try:
            import mamba_ssm  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "MambaBackbone requires `mamba-ssm` package. "
                "Install in Linux+CUDA env: `pip install mamba-ssm causal-conv1d`. "
                f"Original error: {e}"
            ) from e

        self.d_model = d_model
        self.num_layers = num_layers
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.dropout = dropout

        self.blocks = nn.ModuleList([
            _MambaResidualBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        # 마지막 block 출력 정규화 (Transformer encoder 종단 norm과 일관)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (B, L, D) embedding sequence.

        Returns:
            (B, L, D) — Mamba blocks 통과 후 출력.

        Raises:
            ValueError: ndim != 3 or h.shape[-1] != d_model.
        """
        if h.ndim != 3:
            raise ValueError(
                f"MambaBackbone expected shape (B, L, D), got {tuple(h.shape)}."
            )
        if h.shape[-1] != self.d_model:
            raise ValueError(
                f"MambaBackbone expected d_model={self.d_model}, "
                f"got h.shape[-1]={h.shape[-1]}."
            )

        for blk in self.blocks:
            h = blk(h)
        return self.final_norm(h)

    def extra_repr(self) -> str:
        n_params = sum(p.numel() for p in self.parameters())
        return (
            f"d_model={self.d_model}, num_layers={self.num_layers}, "
            f"d_state={self.d_state}, d_conv={self.d_conv}, expand={self.expand}, "
            f"dropout={self.dropout}, params={n_params}"
        )
