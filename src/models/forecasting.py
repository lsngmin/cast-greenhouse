"""ForecastingModel — 학습 가능한 4-layer end-to-end wrapper.

────────────────────────────────────────────────────────────────────────────
역할 (한 줄)
    FeatureEmbedding + Backbone + Pool + Decoder 4개 layer를 한 클래스로 결합.

    X: (B, L, F) → Y_hat: (B, H, V)

────────────────────────────────────────────────────────────────────────────
구조

    X (B, L, F)
      │
      ▼  Step 2  FeatureEmbedding(F → D)
    H_emb (B, L, D)
      │
      ▼  Step 3  Backbone(LSTM/Transformer/Mamba)         ← swap target
    H_bb (B, L, D)
      │
      ▼  Step 5  LastMeanPooling(L → 1)
    C (B, D)
      │
      ▼  Step 6  MLPDecoder(D → H·V → reshape)
    Y_hat (B, H, V)

────────────────────────────────────────────────────────────────────────────
포함되지 않은 것 (의도)

- **Step 1 (Input)**: 학습 외부의 데이터 파이프라인. `src/data/WindowDataset`.
- **Step 7 (Derived VPD)**: 결정론적 후처리. `src/metrics.derive_vpd_from_targets`.
- **Step 8 (Event-window MAE)**: 평가 metric. `src/metrics.event_window_mae`.

ForecastingModel은 **학습 가능한 layer**만 보유한다. metric/derived는 model의
출력을 받아 외부에서 계산. 이렇게 분리하면:
  - backbone swap이 자유로움 (metric 코드 변경 없음)
  - PyTorch 표준 패턴 (Model = forward function only)
  - 학습 graph가 가벼움 (VPD inverse 등이 backprop에 포함되지 않음)

────────────────────────────────────────────────────────────────────────────
설계 결정

- **`backbone_name` 인자 + 내부 `build_backbone` 호출:**
    사용자는 'lstm' / 'transformer' / 'mamba' 문자열만 명시. 학습 코드에서
    backbone swap이 한 줄로 끝남:
        for name in ('lstm', 'transformer', 'mamba'):
            model = ForecastingModel(input_dim=F, backbone_name=name)
            ...

- **`from_dataset` classmethod (편의):**
    WindowDataset의 feature_dim·target_dim·horizon을 자동 추출:
        model = ForecastingModel.from_dataset(ds, backbone_name='lstm')

- **`d_model=128`, decoder hidden=256 default:**
    모든 layer가 일관된 default 보유. baseline 비교 시 추가 인자 불필요.

- **`backbone_kwargs` override (ablation용):**
    sensitivity 실험에서 backbone hyperparameter 변경 시:
        model = ForecastingModel(
            input_dim=F, backbone_name='lstm',
            backbone_kwargs={'num_layers': 3},
        )

────────────────────────────────────────────────────────────────────────────
사용 예

    from torch.utils.data import DataLoader
    from src.data import WindowDataset
    from src.models import ForecastingModel

    ds = WindowDataset('Reference', 'train', 'sensor+weather+state+sp+vip')
    model = ForecastingModel.from_dataset(ds, backbone_name='lstm')

    loader = DataLoader(ds, batch_size=32, shuffle=True)
    for x, y in loader:
        y_hat = model(x)               # (B, 288, 3)
        loss = (y_hat - y).pow(2).mean()
        loss.backward()
        ...

────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from .embedding import FeatureEmbedding, SourceAwareEmbedding
from .pooling import LastMeanPooling
from .decoder import HorizonQueryDecoder, MLPDecoder
from .backbones import build_backbone
from .graph import DirectedGraphModule, build_prior_adjacency
from .fusion import TemporalGraphFusion

if TYPE_CHECKING:
    from src.data.window_dataset import WindowDataset


class ForecastingModel(nn.Module):
    """End-to-end forecasting model: Embed → Backbone → Pool → Decoder.

    Args:
        input_dim:           F. 53 (chain-preserving curated subset).
        backbone_name:       'lstm' / 'transformer' / 'mamba'. default 'lstm'.
        d_model:             내부 hidden dim D. default 128.
        horizon:             출력 step 수 H. default 288 (24h).
        target_dim:          출력 변수 수 V. default 3 (Tair, Rhair, CO2air).
        decoder_hidden_dim:  MLPDecoder hidden. default 256.
        decoder_type:        'mlp' | 'horizon_query'. default 'mlp'.
        backbone_kwargs:     backbone factory에 전달할 override (ablation용).
        graph_mode:          None | 'prior' | 'learned' | 'prior_learned'.
                             None이면 graph 미사용 (baseline). 나머지는
                             Layer 4 (Directed Graph Module) 활성화.
        feature_cols:        embedding_type='source_aware' 또는 graph_mode != None
                             일 때 필요. WindowDataset.feature_cols.
        embedding_type:      'flat' (FeatureEmbedding) 또는 'source_aware'
                             (SourceAwareEmbedding with dynamic gating).
                             default 'flat' (baseline).
        gate_temperature:    embedding_type='source_aware'의 softmax temperature.
                             default 1.0.

    Shape:
        in  : (B, L, F)
        out : (B, H, V)
    """

    def __init__(
        self,
        input_dim: int,
        backbone_name: str = 'lstm',
        d_model: int = 128,
        horizon: int = 288,
        target_dim: int = 3,
        decoder_hidden_dim: int = 256,
        decoder_type: str = 'mlp',
        backbone_kwargs: dict | None = None,
        graph_mode: str | None = None,
        feature_cols: list[str] | None = None,
        embedding_type: str = 'flat',
        gate_temperature: float = 1.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.backbone_name = backbone_name
        self.d_model = d_model
        self.horizon = horizon
        self.target_dim = target_dim
        self.graph_mode = graph_mode
        self.decoder_type = decoder_type
        self.embedding_type = embedding_type

        if decoder_type not in ('mlp', 'horizon_query'):
            raise ValueError(
                f"Unknown decoder_type={decoder_type!r}. "
                "Expected 'mlp' or 'horizon_query'."
            )

        if embedding_type not in ('flat', 'source_aware'):
            raise ValueError(
                f"Unknown embedding_type={embedding_type!r}. "
                "Expected 'flat' or 'source_aware'."
            )

        bb_kwargs = dict(backbone_kwargs) if backbone_kwargs else {}
        # backbone factory에 d_model을 명시 전달 (override 없으면 default 128)
        bb_kwargs.setdefault('d_model', d_model)

        # Layer 2
        if embedding_type == 'source_aware':
            if feature_cols is None:
                raise ValueError(
                    "embedding_type='source_aware' requires feature_cols "
                    "(WindowDataset.feature_cols)."
                )
            self.embed = SourceAwareEmbedding(
                feature_cols=feature_cols,
                d_model=d_model,
                gate_temperature=gate_temperature,
            )
        else:
            self.embed = FeatureEmbedding(input_dim=input_dim, d_model=d_model)
        # Layer 3 (swappable)
        self.backbone = build_backbone(backbone_name, **bb_kwargs)

        # Layer 4 (optional) + Layer 5 fusion / pool
        if graph_mode is None:
            self.graph = None
            self.pool = LastMeanPooling(d_model=d_model)
        else:
            # Prior adjacency 생성 (feature_cols 기반)
            if graph_mode in ('prior', 'prior_learned'):
                if feature_cols is None:
                    raise ValueError(
                        f"graph_mode={graph_mode!r} requires feature_cols (변수 이름 list)."
                    )
                if len(feature_cols) != input_dim:
                    raise ValueError(
                        f"len(feature_cols)={len(feature_cols)} != input_dim={input_dim}"
                    )
                adj = build_prior_adjacency(feature_cols)
            else:  # 'learned' — random init
                adj = None

            self.graph = DirectedGraphModule(
                input_dim=input_dim,
                d_model=d_model,
                adjacency=adj,
                mode=graph_mode,
            )
            # Pool 위치에 fusion 모듈 (Pool을 내부에서 contain)
            self.pool = TemporalGraphFusion(d_model=d_model)

        # Layer 6
        decoder_cls = MLPDecoder if decoder_type == 'mlp' else HorizonQueryDecoder
        self.decoder = decoder_cls(
            d_model=d_model,
            horizon=horizon,
            target_dim=target_dim,
            hidden_dim=decoder_hidden_dim,
        )

    @classmethod
    def from_dataset(
        cls,
        dataset: 'WindowDataset',
        backbone_name: str = 'lstm',
        d_model: int = 128,
        decoder_hidden_dim: int = 256,
        decoder_type: str = 'mlp',
        backbone_kwargs: dict | None = None,
        graph_mode: str | None = None,
        embedding_type: str = 'flat',
        gate_temperature: float = 1.0,
    ) -> 'ForecastingModel':
        """WindowDataset에서 input_dim, horizon, target_dim, feature_cols 자동 추출.

        Example:
            ds = WindowDataset('Reference', 'train')
            model = ForecastingModel.from_dataset(ds)                                       # baseline
            model = ForecastingModel.from_dataset(ds, 'mamba', embedding_type='source_aware')  # 제안 임베딩
        """
        need_cols = embedding_type == 'source_aware' or graph_mode is not None
        return cls(
            input_dim=dataset.feature_dim,
            backbone_name=backbone_name,
            d_model=d_model,
            horizon=dataset.horizon,
            target_dim=dataset.target_dim,
            decoder_hidden_dim=decoder_hidden_dim,
            decoder_type=decoder_type,
            backbone_kwargs=backbone_kwargs,
            graph_mode=graph_mode,
            feature_cols=dataset.feature_cols if need_cols else None,
            embedding_type=embedding_type,
            gate_temperature=gate_temperature,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, F) input window.

        Returns:
            (B, H, V) — scaled forecast trajectory.
        """
        h = self.embed(x)            # (B, L, F)  → (B, L, D)
        h = self.backbone(h)         # (B, L, D)  → (B, L, D)
        if self.graph is not None:
            g = self.graph(x)        # (B, L, F)  → (B, L, D)  — uses raw input
            c = self.pool(h, g)      # TemporalGraphFusion: (B, L, D), (B, L, D) → (B, D)
        else:
            c = self.pool(h)         # LastMeanPooling: (B, L, D) → (B, D)
        y = self.decoder(c)          # (B, D)     → (B, H, V)
        return y

    def count_parameters(self) -> dict:
        """Layer별 trainable parameter 수 (디버깅·로깅용)."""
        per_layer = {
            'embed':    sum(p.numel() for p in self.embed.parameters()),
            'backbone': sum(p.numel() for p in self.backbone.parameters()),
            'pool':     sum(p.numel() for p in self.pool.parameters()),
            'decoder':  sum(p.numel() for p in self.decoder.parameters()),
        }
        if self.graph is not None:
            per_layer['graph'] = sum(p.numel() for p in self.graph.parameters())
        per_layer['total'] = sum(v for k, v in per_layer.items() if k != 'total')
        return per_layer

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, backbone={self.backbone_name!r}, "
            f"d_model={self.d_model}, horizon={self.horizon}, "
            f"target_dim={self.target_dim}, decoder_type={self.decoder_type!r}, "
            f"params={self.count_parameters()['total']}"
        )
