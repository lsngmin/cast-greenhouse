"""Feature Embedding Layer — 논문 8-layer 구조의 Layer 2.

────────────────────────────────────────────────────────────────────────────
역할 (한 줄)
    Raw feature dim F를 model dim D로 projection.   X: (B, L, F) → H: (B, L, D)

────────────────────────────────────────────────────────────────────────────
왜 필요한가?

1. **Feature group ablation을 공정하게 비교하기 위해.**
   본 연구의 핵심 ablation은 control 정보를 단계적으로 추가하는 것:
       sensor (F=8) ⊂ +weather (F=18) ⊂ +state (F=38) ⊂ +sp (F=58) ⊂ +vip (F=74)
   feature group마다 F가 다르므로, F를 그대로 backbone(LSTM/Transformer/Mamba)
   에 넣으면 backbone hidden size·파라미터 수가 그룹별로 달라진다. 그러면
   성능 차이가 'control 정보 추가의 효과'인지 'backbone capacity 차이'인지
   해석이 안 된다. F → D=128로 통일하면 backbone 구조가 모든 그룹에서 동일
   해진다.

2. **Scaled feature 분포의 통일.**
   `src/preprocessing.py`에서 StandardScaler를 train split 기준으로 fit
   했으므로 numeric feature는 대략 mean=0, std=1. 하지만:
     - flag 컬럼 (`_up`, `_down`, `_changed`, `co2_dos_on`)은 0/1 raw binary
     - VPD 같은 derived feature는 분포가 비대칭 (long tail)
     - scale 후에도 feature 간 variance가 완벽히 동일하지 않음
   LayerNorm으로 token마다(=time step마다) 다시 정규화하여 backbone이 특정
   feature의 large variance에 dominate되지 않도록 함.

────────────────────────────────────────────────────────────────────────────
설계 결정 및 이유

- **Linear (bias=False):**
  바로 뒤의 LayerNorm이 affine `β` (bias-equivalent) 파라미터를 학습하므로
  Linear의 bias는 수학적으로 redundant. `(Wx + b)` 출력이 LayerNorm에서
  per-token zero-mean으로 강제되며, 학습된 `b`는 LayerNorm `β`로 흡수됨.
  의미상 중복 제거 + 파라미터 128개 감소.

- **활성화 함수 없음 (linear projection만):**
  ReLU/GELU 같은 비선형성을 여기서 적용하면 negative scaled value의 정보가
  손실됨. Scaled feature는 음수도 의미 있는 값 (예: train mean 이하 = 비교적
  추운 날씨). 비선형성은 backbone(LSTM gate, Transformer FFN, Mamba SSM)이
  담당.

- **LayerNorm 위치 = projection 후 (post-norm 스타일):**
  Linear → LayerNorm 순서. backbone 내부에서는 별도 norm을 적용하므로 여기는
  post-norm이 안전.

- **Dropout 없음:**
  Transformer 계열에서는 embedding 직후 dropout이 관행이지만 본 연구에서는
  의도적으로 제외. 이유:
    (a) backbone 내부에 이미 dropout이 존재 (LSTM inter-layer, Transformer
        attention/FFN, Mamba block) — 같은 regularization을 두 번 적용하는
        과도한 경향
    (b) Feature group에 binary event flag (`<act>_up/_down`, `co2_dos_on`,
        `<sp>_changed`)가 8~13개 포함. embedded representation이 dropout으로
        random masking되면 event 신호가 학습 노이즈로 들어갈 위험
    (c) 표준 nn.Dropout은 i.i.d. masking이라 time series 시간 일관성을
        해침 (variational dropout이면 다르지만 baseline에서는 단순화)
  Overfitting이 관찰되면 backbone 내부 dropout rate를 올리는 방향이 우선.

- **d_model = 128 default:**
  계획서/공통 합의값. baseline 비교 후 sensitivity (64/128/256)로 부록 가능.

────────────────────────────────────────────────────────────────────────────
입출력 contract

    Input:  x  (B, L, F) float32
    Output: h  (B, L, D) float32

    B = batch size
    L = lookback steps (288 default, 또는 72/144/576 in sensitivity)
    F = feature dim (feature_group dependent; 8/18/38/58/74)
    D = d_model (128 default)

────────────────────────────────────────────────────────────────────────────
논문 작성 참고 (Methods 섹션에 들어갈 내용)

    "All input features are projected to a common embedding dimension
     d_model=128 via a linear layer followed by LayerNorm. This standardizes
     the input dimensionality across feature ablation groups (8/18/38/58/74
     dimensions) and allows the temporal backbone to be compared under
     identical hidden capacity. No non-linear activation is applied at this
     stage, as the scaled features (including control signals such as
     setpoint values and binary event flags) carry informative negative
     values; non-linearity is deferred to the backbone. Dropout is omitted
     at the embedding layer to avoid stochastic masking of binary event
     flag channels (e.g., setpoint changes, actuator state transitions);
     regularization is applied inside the temporal backbone instead."

    Sensitivity / Ablation 부록:
    - d_model {64, 128, 256} 비교는 부록에서 한 backbone (e.g., LSTM)으로만
      보고하면 충분.
    - LayerNorm 제거 ablation → embedding 안정성에 미치는 영향 (선택).

────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FeatureEmbedding(nn.Module):
    """Feature Embedding (Layer 2): F → D projection + LayerNorm.

    All baseline models (LSTM, Transformer, Mamba)와 향후 Mamba-STGNN까지
    공통으로 이 layer를 첫 단계로 사용한다. Dropout은 의도적으로 제외
    (자세한 이유는 모듈 헤더 docstring 참조).

    Args:
        input_dim:  feature dim F. feature_group에 따라 가변.
        d_model:    embedding dim D. 모든 그룹/모델에서 동일하게 유지 (기본 128).

    Shape:
        in  : (B, L, F)
        out : (B, L, D)
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
    ):
        super().__init__()

        # F → D linear projection (bias=False — LayerNorm의 affine β가 동일 역할).
        # 이 단계가 feature_group ablation에서 backbone capacity를 통일시키는
        # 핵심. F=8(sensor)과 F=74(full)이 둘 다 D=128로 매핑됨.
        self.linear = nn.Linear(input_dim, d_model, bias=False)

        # LayerNorm: token 차원 D=128에 대해 mean/std 정규화 (affine 학습 포함).
        # 매 timestep을 독립적으로 normalize함.
        self.norm = nn.LayerNorm(d_model)

        # 메타 정보 (forward에서 shape 체크 + extra_repr에서 사용)
        self.input_dim = input_dim
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project input features to model embedding.

        Args:
            x: (B, L, F) float tensor. F는 self.input_dim과 일치해야 함.

        Returns:
            (B, L, D) float tensor.

        Raises:
            ValueError: input shape의 마지막 차원이 self.input_dim과 다를 때.
                        feature_group을 바꾸고 모델 재초기화를 안 한 경우 흔히 발생.
        """
        # Defensive shape check. ablation 실험 시 feature_group 변경 후
        # model을 재생성하지 않아 dim mismatch가 흔히 발생함.
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"FeatureEmbedding expected input_dim={self.input_dim}, "
                f"got x.shape[-1]={x.shape[-1]}. "
                f"feature_group을 바꿨다면 모델을 재초기화해야 함."
            )

        h = self.linear(x)     # (B, L, F) → (B, L, D)
        h = self.norm(h)       # token-wise LayerNorm over D
        return h

    def extra_repr(self) -> str:
        """torch print 시 추가 정보."""
        n_params = sum(p.numel() for p in self.parameters())
        return (f"input_dim={self.input_dim}, d_model={self.d_model}, "
                f"params={n_params}")
