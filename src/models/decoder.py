"""Trajectory Decoder — 논문 8-layer 구조의 Layer 6.

────────────────────────────────────────────────────────────────────────────
역할 (한 줄)
    단일 context vector를 받아 미래 H step × V variable trajectory를
    direct multi-step으로 한 번에 출력.
    C: (B, D) → Y_hat: (B, H, V)

────────────────────────────────────────────────────────────────────────────
왜 필요한가?

Pool layer (Step 5)가 backbone 출력 (B, L, D)를 (B, D)로 압축한 context
벡터를 만들었다. 이제 이 context로부터 forecast trajectory `[t0+1, ..., t0+H]`
의 환경변수 V개 (Tair, Rhair, CO2air) 를 예측해야 한다.

방식은 크게 셋:
  1) **Direct multi-step**: (B, D) → (B, H·V) → reshape (B, H, V). 한 번에 전체.
  2) Autoregressive: step별 순차 출력. 이전 step output을 다음 입력으로.
  3) Patch-based: H를 patch로 나눠 patch별 처리.

본 연구는 **Direct (#1)** 채택.

────────────────────────────────────────────────────────────────────────────
설계 결정 및 이유

- **Direct multi-step (한 번에 H step 출력):**
    Pros: 빠름 (한 번 forward), error compounding 없음, 모든 backbone 비교에서
          decoder 동작이 동일하므로 backbone 효과 isolate 가능.
    Cons: H가 클수록 마지막 Linear weight 크기 증가 (`hidden × H·V`).
    본 연구의 H=288, V=3 → 마지막 layer 222,048 param (hidden=256 기준).
    부담 가능한 수준.

- **3-layer MLP (`D → hidden → hidden → H·V`):**
    - 2-layer: capacity 부족 위험.
    - 3-layer: 표현력 + 적정 깊이. baseline에 적합.
    - 4+ layer: 학습 안정성·overfitting 위험 증가.

- **hidden_dim = 256 (default):**
    `d_model=128 → 256`. backbone 출력 dim의 2배 정도면 expand 효과.
    sensitivity는 부록에서 비교 가능 (128/256/512).

- **Activation = GELU:**
    ReLU 대비 부드러운 saturation, transformer-era 표준. 음수 영역도 작게
    유지하여 정보 보존.

- **Dropout 없음:**
    `FeatureEmbedding`과 동일 정책. backbone에서 이미 regularization 적용.
    target도 scaled되어 있어 추가 stochasticity 필요성 낮음. Overfitting
    관찰 시 backbone dropout을 조정하거나 dropout을 마지막 hidden 직후에만
    선택적으로 추가.

- **LayerNorm 없음 (hidden layer 사이):**
    MLP decoder에서 layer norm은 흔치 않음. 단순함 유지.
    Pool에서 이미 input(=decoder input) 정규화됨.

- **Output Linear bias = True:**
    LayerNorm/Activation 없는 final Linear → bias가 실제 offset 학습.
    target은 scaled되어 mean≈0이지만, val/test 분포 shift 때문에 bias가
    조정 여지를 제공.

────────────────────────────────────────────────────────────────────────────
입출력 contract

    Input:  c       (B, D)       context vector from Pool layer
    Output: y_hat   (B, H, V)    forecast trajectory in scaled units

    H = horizon steps (default 288 = 24h)
    V = target dim (default 3 = Tair, Rhair, CO2air)

────────────────────────────────────────────────────────────────────────────
구조적 주의 — single context bottleneck (상속)

Pool layer가 이미 (B, L, D) → (B, D)로 압축했으므로, Decoder가 받는 정보량
한계가 있다. 이는 baseline 단순성을 위한 선택. event timing이나 정밀한
response curve 추적엔 한계 가능. 후속 개선 후보:
  - Cross-attention decoder (Transformer-style, query-key-value)
  - Horizon-wise / patch-based decoder
  - Two-stage: coarse trajectory → refinement

현재 baseline에선 단순 MLP direct로 충분 (논문 기여는 decoder가 아니라
control-aware input + event-based evaluation).

────────────────────────────────────────────────────────────────────────────
논문 작성 참고 (Methods 섹션에 들어갈 내용)

    "The trajectory decoder maps the pooled context vector (B, D) to a
     full 24-hour trajectory (B, H=288, V=3) via direct multi-step
     prediction. We use a 3-layer MLP with GELU activations:
     Linear(D → 256) → GELU → Linear(256 → 256) → GELU →
     Linear(256 → H·V), followed by a reshape to (B, H, V). Direct
     prediction avoids error compounding of autoregressive decoding and
     keeps the decoder identical across all backbones (LSTM, Transformer,
     Mamba), isolating the comparison to the temporal encoder choice.
     No dropout is applied in the decoder, as regularization is handled
     inside the backbone."

    Sensitivity / Ablation 부록 후보:
    - hidden_dim {128, 256, 512}
    - layer 수 {2, 3, 4}
    - ReLU vs GELU
    - Direct vs autoregressive (학습 시간 허용 시)

────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MLPDecoder(nn.Module):
    """Trajectory Decoder (Layer 6) — MLP-based direct multi-step implementation.

    논문 layer 명칭은 "Trajectory Decoder"이지만, 본 클래스는 그 layer를
    **MLP 구조로 구현**한 것이므로 `MLPDecoder`로 명명. 향후 다른 decoder
    구현(autoregressive, cross-attention 등)을 비교할 경우 별도 클래스로
    추가 가능.

    구조:
        Linear(D → hidden_dim)
        GELU
        Linear(hidden_dim → hidden_dim)
        GELU
        Linear(hidden_dim → H·V)
        reshape → (B, H, V)

    Args:
        d_model:     input context dim D.
        horizon:     output time steps H (default 288 = 24h with 5-min grid).
        target_dim:  output variable count V (default 3).
        hidden_dim:  MLP hidden dim (default 256).

    Shape:
        in  : (B, D)
        out : (B, H, V)

    Parameters (default D=128, hidden=256, H=288, V=3):
        Linear1 (D → 256)         : 128·256 + 256 = 33,024
        Linear2 (256 → 256)       : 256·256 + 256 = 65,792
        Linear3 (256 → 288·3=864) : 256·864 + 864 = 222,048
        Total                     : 320,864
    """

    def __init__(
        self,
        d_model: int = 128,
        horizon: int = 288,
        target_dim: int = 3,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.d_model = d_model
        self.horizon = horizon
        self.target_dim = target_dim
        self.hidden_dim = hidden_dim

        # 3-layer MLP: D → hidden → hidden → H·V
        # bias=True everywhere — 사이엔 LayerNorm 없고, final 출력도 offset 필요.
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_dim, horizon * target_dim, bias=True),
        )

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            c: (B, D) context vector from Pool layer.

        Returns:
            (B, H, V) forecast trajectory in scaled units.

        Raises:
            ValueError: ndim != 2 or c.shape[-1] != d_model.
        """
        # Defensive shape check
        if c.ndim != 2:
            raise ValueError(
                f"MLPDecoder expected shape (B, D), got {tuple(c.shape)}."
            )
        if c.shape[-1] != self.d_model:
            raise ValueError(
                f"MLPDecoder expected d_model={self.d_model}, "
                f"got c.shape[-1]={c.shape[-1]}."
            )

        flat = self.net(c)                                                  # (B, H·V)
        return flat.reshape(c.shape[0], self.horizon, self.target_dim)      # (B, H, V)

    def extra_repr(self) -> str:
        n_params = sum(p.numel() for p in self.parameters())
        return (f"d_model={self.d_model}, hidden_dim={self.hidden_dim}, "
                f"horizon={self.horizon}, target_dim={self.target_dim}, "
                f"params={n_params}")


class HorizonQueryDecoder(nn.Module):
    """Horizon-query decoder with target-specific output heads.

    Compared with ``MLPDecoder``, this decoder does not produce all H x V
    outputs from one final flat linear layer. Instead, each horizon step has
    a learned query embedding, and each target has its own small output head.
    This keeps the direct multi-step contract while making the decoder
    horizon-aware and target-specific.

    Structure:
        context c: (B, D)
        learned horizon queries: (H, D)
        z = LayerNorm(c[:, None, :] + query[None, :, :])  # (B, H, D)
        shared MLP: D -> hidden -> hidden                 # (B, H, hidden)
        target heads: hidden -> 1 per target              # (B, H, V)

    Shape:
        in  : (B, D)
        out : (B, H, V)
    """

    def __init__(
        self,
        d_model: int = 128,
        horizon: int = 288,
        target_dim: int = 3,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.d_model = d_model
        self.horizon = horizon
        self.target_dim = target_dim
        self.hidden_dim = hidden_dim

        self.horizon_query = nn.Embedding(horizon, d_model)
        self.query_norm = nn.LayerNorm(d_model)
        self.shared = nn.Sequential(
            nn.Linear(d_model, hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.GELU(),
        )
        self.target_heads = nn.ModuleList(
            nn.Linear(hidden_dim, 1, bias=True) for _ in range(target_dim)
        )

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            c: (B, D) context vector from Pool/Fusion layer.

        Returns:
            (B, H, V) forecast trajectory in scaled units.
        """
        if c.ndim != 2:
            raise ValueError(
                f"HorizonQueryDecoder expected shape (B, D), got {tuple(c.shape)}."
            )
        if c.shape[-1] != self.d_model:
            raise ValueError(
                f"HorizonQueryDecoder expected d_model={self.d_model}, "
                f"got c.shape[-1]={c.shape[-1]}."
            )

        horizon_idx = torch.arange(self.horizon, device=c.device)
        q = self.horizon_query(horizon_idx)                       # (H, D)
        z = self.query_norm(c.unsqueeze(1) + q.unsqueeze(0))       # (B, H, D)
        z = self.shared(z)                                        # (B, H, hidden)
        y = torch.cat([head(z) for head in self.target_heads], dim=-1)
        return y                                                  # (B, H, V)

    def extra_repr(self) -> str:
        n_params = sum(p.numel() for p in self.parameters())
        return (f"d_model={self.d_model}, hidden_dim={self.hidden_dim}, "
                f"horizon={self.horizon}, target_dim={self.target_dim}, "
                f"params={n_params}")


class TargetHeadMLPDecoder(nn.Module):
    """Trajectory Decoder with per-target output heads (no horizon queries).

    `MLPDecoder`가 마지막 Linear에서 H·V를 한꺼번에 출력하는 것과 달리, 본 변형은
    shared trunk 뒤에 **target별 독립 head**를 둔다. trunk는 모든 target 공통,
    각 head는 해당 target의 trajectory만 출력하므로 target-specific nonlinearity가
    가능하다.

    `HorizonQueryDecoder`와의 차이: 본 클래스는 horizon-query embedding을 쓰지
    않고, 단일 context vector에서 각 head가 H step 전체를 직접 출력한다 (MLPDecoder
    동일 패턴). 즉 `MLPDecoder`의 "마지막 Linear만 target별로 분리"한 최소 변경
    버전.

    구조:
        shared trunk:
            Linear(D → shared_hidden_dim)
            GELU
            Linear(shared_hidden_dim → shared_hidden_dim)
            GELU
        target heads (V개, 독립):
            Linear(shared_hidden_dim → head_hidden_dim)
            GELU
            Linear(head_hidden_dim → H)

    Args:
        d_model:           input context dim D.
        horizon:           output time steps H.
        target_dim:        output variable count V (= head 개수).
        shared_hidden_dim: shared trunk hidden dim. default 256.
        head_hidden_dim:   target-head hidden dim. default 128.

    Shape:
        in  : (B, D)
        out : (B, H, V)

    Parameters (default D=128, shared=256, head=128, H=288, V=3):
        Shared trunk:
            Linear1 (D → 256)             : 128·256 + 256 = 33,024
            Linear2 (256 → 256)           : 256·256 + 256 = 65,792
        Per head (3개):
            Linear_a (256 → 128)          : 256·128 + 128 = 32,896
            Linear_b (128 → H=288)        : 128·288 + 288 = 37,152
            head 1개                       : 70,048
            heads 3개                      : 210,144
        Total                              : 308,960

    Comparison with ``MLPDecoder`` (320,864 params):
        TargetHead는 약 12k 적음 (−3.7%). shared trunk capacity 동일,
        마지막 Linear가 (256 → H·V) flat → target별 (256 → head_hidden → H)로
        분리된 게 핵심 변경.
    """

    def __init__(
        self,
        d_model: int = 128,
        horizon: int = 288,
        target_dim: int = 3,
        shared_hidden_dim: int = 256,
        head_hidden_dim: int = 128,
    ):
        super().__init__()
        self.d_model = d_model
        self.horizon = horizon
        self.target_dim = target_dim
        self.shared_hidden_dim = shared_hidden_dim
        self.head_hidden_dim = head_hidden_dim

        # Shared trunk: D → shared → shared, GELU sandwich (MLPDecoder 동일 구조)
        self.shared = nn.Sequential(
            nn.Linear(d_model, shared_hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(shared_hidden_dim, shared_hidden_dim, bias=True),
            nn.GELU(),
        )

        # Per-target heads: shared_hidden → head_hidden → H
        # 각 head는 독립 파라미터 (ModuleList). H step 전체를 한 번에 출력.
        self.target_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(shared_hidden_dim, head_hidden_dim, bias=True),
                nn.GELU(),
                nn.Linear(head_hidden_dim, horizon, bias=True),
            )
            for _ in range(target_dim)
        ])

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            c: (B, D) context vector from Pool layer.

        Returns:
            (B, H, V) forecast trajectory in scaled units.
        """
        if c.ndim != 2:
            raise ValueError(
                f"TargetHeadMLPDecoder expected shape (B, D), got {tuple(c.shape)}."
            )
        if c.shape[-1] != self.d_model:
            raise ValueError(
                f"TargetHeadMLPDecoder expected d_model={self.d_model}, "
                f"got c.shape[-1]={c.shape[-1]}."
            )

        s = self.shared(c)                                           # (B, shared_hidden)
        # 각 head: (B, shared_hidden) → (B, H). V개 stack → (B, H, V)
        per_target = [head(s) for head in self.target_heads]         # V개 (B, H)
        y = torch.stack(per_target, dim=-1)                          # (B, H, V)
        return y

    def extra_repr(self) -> str:
        n_params = sum(p.numel() for p in self.parameters())
        return (f"d_model={self.d_model}, "
                f"shared_hidden_dim={self.shared_hidden_dim}, "
                f"head_hidden_dim={self.head_hidden_dim}, "
                f"horizon={self.horizon}, target_dim={self.target_dim}, "
                f"params={n_params}")
