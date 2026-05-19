"""Temporal Encoder Backbones (논문 8-layer 구조의 Layer 3).

각 backbone은 동일한 입출력 contract를 따른다:
    forward(h: (B, L, D)) → (B, L, D)

이를 통해 ForecastingModel에서 backbone을 한 줄 교체로 비교 가능:
    backbone = build_backbone('lstm')
    backbone = build_backbone('transformer')
    backbone = build_backbone('mamba')

모든 backbone은 default 인자만으로 baseline 비교용 인스턴스를 생성한다.
Sensitivity 실험 시에는 kwargs로 override:
    build_backbone('lstm', num_layers=3)
"""
from __future__ import annotations
from typing import TYPE_CHECKING
import torch.nn as nn

if TYPE_CHECKING:
    pass

from .lstm import LSTMBackbone
from .transformer import TransformerBackbone


BACKBONE_NAMES = ('lstm', 'transformer', 'mamba')


def build_backbone(name: str, **overrides) -> nn.Module:
    """Factory: backbone name → instance.

    Args:
        name: 'lstm' / 'transformer' / 'mamba'.
        **overrides: backbone-specific kwargs (e.g. num_layers=3). 기본 설정은
                     각 backbone 클래스의 default 값에 정의되어 있으므로 baseline
                     비교 시 추가 인자 불필요.

    Returns:
        nn.Module — backbone instance.

    Raises:
        ValueError: unknown name.
        ImportError: 'mamba' 선택 시 mamba-ssm 패키지 미설치.
    """
    name = name.lower()
    if name == 'lstm':
        return LSTMBackbone(**overrides)
    elif name == 'transformer':
        return TransformerBackbone(**overrides)
    elif name == 'mamba':
        from .mamba import MambaBackbone               # late import (optional dep)
        return MambaBackbone(**overrides)
    raise ValueError(
        f"Unknown backbone '{name}'. Available: {BACKBONE_NAMES}"
    )


__all__ = ['build_backbone', 'BACKBONE_NAMES', 'LSTMBackbone', 'TransformerBackbone']
