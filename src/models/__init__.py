"""Model components for AGC2 microclimate forecasting.

논문 8-layer 구조의 모델 부분 (Layer 2~6).

진행 순서 (한 파일 = 한 layer):
    1. embedding.py    — Feature Embedding (Layer 2)             [현재]
    2. pooling.py      — Temporal Pooling (Layer 5 자리, baseline)
    3. decoder.py      — Trajectory Decoder (Layer 6)
    4. backbones/      — Temporal Backbone (Layer 3): LSTM/Transformer/Mamba
    5. forecasting.py  — ForecastingModel wrapper

Layer 4 (Directed Graph) 및 Layer 5 (Fusion)은 baseline 통과 후 도입.
Layer 7 (Derived Variable: VPD), Layer 8 (Event-based metric)은 model 외부
(`src/metrics.py`)에서 처리.
"""
from .embedding import FeatureEmbedding
from .pooling import LastMeanPooling
from .decoder import HorizonQueryDecoder, MLPDecoder
from .backbones import build_backbone, LSTMBackbone, TransformerBackbone, BACKBONE_NAMES
from .graph import DirectedGraphModule, build_prior_adjacency, PRIOR_EDGES
from .fusion import TemporalGraphFusion
from .forecasting import ForecastingModel

__all__ = [
    'FeatureEmbedding',
    'LastMeanPooling',
    'MLPDecoder',
    'HorizonQueryDecoder',
    'build_backbone', 'LSTMBackbone', 'TransformerBackbone', 'BACKBONE_NAMES',
    'DirectedGraphModule', 'build_prior_adjacency', 'PRIOR_EDGES',
    'TemporalGraphFusion',
    'ForecastingModel',
]
