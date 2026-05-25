"""Sliding window dataset creation for AGC2 forecasting.

핵심 설계:
  - Input X: 과거 lookback step의 모든 feature (numeric scaled + flag 그대로)
  - Output Y: 미래 horizon step의 target 변수 (scaled)
  - Window는 target NaN-free만 사용 (drop_target_nan=True)
  - lookback presets: 6h/12h/24h/48h (계획서 lookback sensitivity 실험용)
  - horizon: 24h 고정 (event-window evaluation 위해)
  - stride: 12 step = 1h (sample 중복 줄임)

EDA (notebooks/eda/03)에서 24h lookback의 EDA 근거 확보됨. 최종 lookback은 sensitivity로 결정.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Sequence

import numpy as np
import pandas as pd


# Lookback presets in 5-min steps
LOOKBACK_PRESETS = {
    '6h':  72,
    '12h': 144,
    '24h': 288,
    '48h': 576,
}
HORIZON_DEFAULT = 288   # 24 hours
STRIDE_DEFAULT  = 12    # 1 hour

# VPD는 derived variable (Tair·Rhair로 계산 가능)이므로 직접 예측하지 않음.
# 평가 단계에서 예측된 Tair/Rhair로 VPD 계산하여 metric 산출.
DEFAULT_TARGETS = ('Tair', 'Rhair', 'CO2air')
EVENT_WEIGHT_NONE = 'none'
EVENT_TARGET_MODE_SHARED = 'shared'
EVENT_TARGET_MODE_MATCHED = 'matched'


@dataclass
class WindowConfig:
    lookback: int = 288   # steps (24h × 12 steps/h)
    horizon:  int = 288   # steps (24h)
    stride:   int = 12    # steps (1h)

    @property
    def total_steps(self) -> int:
        return self.lookback + self.horizon

    def label(self) -> str:
        lb_h = self.lookback / 12
        hr_h = self.horizon / 12
        st_h = self.stride / 12
        return f'lb={lb_h:g}h, hr={hr_h:g}h, stride={st_h:g}h'


# ---------------------------------------------------------------------------
# Window start indices

def get_window_starts(n_steps: int, cfg: WindowConfig) -> np.ndarray:
    """모든 가능한 window 시작 인덱스 (NaN 체크 안 함)."""
    max_start = n_steps - cfg.total_steps
    if max_start < 0:
        return np.empty(0, dtype=np.int64)
    return np.arange(0, max_start + 1, cfg.stride, dtype=np.int64)


def clean_window_starts(df: pd.DataFrame, target_cols: Sequence[str],
                        cfg: WindowConfig) -> np.ndarray:
    """target에 NaN이 하나도 없는 window의 시작 index만 반환.

    Lookback + horizon 합쳐서 NaN-free여야 함 (target만 체크 — feature는 fillna로 처리).
    """
    miss = df[list(target_cols)].isna().any(axis=1).values
    cs = np.concatenate(([0], np.cumsum(miss)))
    starts = get_window_starts(len(df), cfg)
    if len(starts) == 0:
        return starts
    # cs[end] - cs[start] = 해당 window 내 NaN row 수
    bad_count = cs[starts + cfg.total_steps] - cs[starts]
    return starts[bad_count == 0]


# ---------------------------------------------------------------------------
# Build windows

def make_windows(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_cols: Sequence[str],
    cfg: WindowConfig,
    drop_target_nan: bool = True,
    starts: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Sliding window tensor 생성.

    Parameters
    ----------
    df : DataFrame with DatetimeIndex (5-min grid).
    feature_cols : X에 들어갈 컬럼 (모든 feature, scaled 상태여야 함).
    target_cols  : Y에 들어갈 컬럼 (target만).
    cfg : WindowConfig.
    drop_target_nan : True면 target NaN이 있는 window 제외.
    starts : 직접 지정하고 싶을 때.

    Returns
    -------
    X : (n_samples, lookback, n_features) float32
    Y : (n_samples, horizon, n_targets) float32
    meta : DataFrame with [t0_idx, t0_ts, lookback_start_ts, t_pred_end_ts]
         - t0_ts = forecast issue time (lookback 끝, horizon 시작)
    """
    feature_cols = list(feature_cols)
    target_cols = list(target_cols)

    if starts is None:
        starts = (clean_window_starts(df, target_cols, cfg)
                  if drop_target_nan
                  else get_window_starts(len(df), cfg))

    if len(starts) == 0:
        return (np.empty((0, cfg.lookback, len(feature_cols)), dtype=np.float32),
                np.empty((0, cfg.horizon, len(target_cols)), dtype=np.float32),
                pd.DataFrame(columns=['t0_idx', 't0_ts',
                                      'lookback_start_ts', 't_pred_end_ts']))

    feat = df[feature_cols].values.astype(np.float32)
    targ = df[target_cols].values.astype(np.float32)

    n = len(starts)
    X = np.empty((n, cfg.lookback, len(feature_cols)), dtype=np.float32)
    Y = np.empty((n, cfg.horizon, len(target_cols)), dtype=np.float32)
    for i, s in enumerate(starts):
        e_lb = s + cfg.lookback
        X[i] = feat[s:e_lb]
        Y[i] = targ[e_lb:e_lb + cfg.horizon]

    ts = df.index
    t0_idx = starts + cfg.lookback
    meta = pd.DataFrame({
        't0_idx': t0_idx,
        't0_ts': ts[t0_idx],
        'lookback_start_ts': ts[starts],
        't_pred_end_ts': ts[t0_idx + cfg.horizon - 1],
    })
    return X, Y, meta


def make_windows_scaled(
    df: pd.DataFrame,
    scaler,
    numeric_cols: Sequence[str],
    feature_cols: Sequence[str],
    target_cols: Sequence[str],
    cfg: WindowConfig,
    feature_nan_fill: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Numeric feature에 scaler 적용 후 window 생성.

    Flag 컬럼 (`_changed`, `_up`, `_down`)은 scaling 안 함, 0/1 그대로.
    Feature의 NaN은 scaling 후 `feature_nan_fill` (default 0)로 채움.
    Target은 scaler로 변환됨 (numeric_cols에 포함되어 있다고 가정).
    """
    numeric_cols = list(numeric_cols)
    feature_cols = list(feature_cols)
    target_cols = list(target_cols)

    df = df.copy()
    # pandas 3.0이 int → float 할당을 막아 명시적 캐스팅 필요
    arr = df[numeric_cols].values.astype(np.float64)
    scaled = scaler.transform(arr)
    df[numeric_cols] = scaled.astype(np.float64)
    # 1) target NaN-free window 결정 (fillna 전에)
    starts = clean_window_starts(df, target_cols, cfg)
    # 2) feature NaN fill — target은 fillna에서 제외 (target NaN window는 위에서 이미 drop)
    fill_cols = [c for c in feature_cols if c not in target_cols]
    df[fill_cols] = df[fill_cols].fillna(feature_nan_fill)
    return make_windows(df, feature_cols, target_cols, cfg, starts=starts)


def make_event_weights(
    df: pd.DataFrame,
    meta: pd.DataFrame,
    cfg: WindowConfig,
    target_cols: Sequence[str] = DEFAULT_TARGETS,
    mode: str = EVENT_WEIGHT_NONE,
    event_target_mode: str = EVENT_TARGET_MODE_SHARED,
    event_cols: Sequence[str] | None = None,
    event_loss_lambda: float = 0.5,
    event_window_steps: int = 12,
    event_decay_window_steps: int = 72,
    event_decay_tau_steps: int = 24,
) -> np.ndarray | None:
    """Build per-horizon loss weights from future control-event flags.

    The returned matrix is used only during training/validation loss
    computation. It does not change model inputs or inference.

    Returns
    -------
    event_weight : float32 array, or None when disabled.
        Shape is (n_samples, horizon) for shared weighting and
        (n_samples, horizon, n_targets) for matched weighting. Values are
        >= 1.0. General horizons keep weight 1.0, while horizons following
        actuator/setpoint events receive additional weight.
    """
    if mode in (None, EVENT_WEIGHT_NONE, 'base'):
        return None
    if mode not in ('event_1h', 'event_decay'):
        raise ValueError(
            "mode must be one of {'none', 'base', 'event_1h', 'event_decay'}, "
            f"got {mode!r}."
        )
    if event_target_mode not in (EVENT_TARGET_MODE_SHARED, EVENT_TARGET_MODE_MATCHED):
        raise ValueError(
            "event_target_mode must be one of {'shared', 'matched'}, "
            f"got {event_target_mode!r}."
        )

    n = len(meta)
    target_cols = list(target_cols)
    if event_target_mode == EVENT_TARGET_MODE_MATCHED:
        weights = np.ones((n, cfg.horizon, len(target_cols)), dtype=np.float32)
    else:
        weights = np.ones((n, cfg.horizon), dtype=np.float32)
    if n == 0 or event_loss_lambda <= 0:
        return weights

    if event_cols is None:
        from src.data.feature_groups import ACTUATOR_EVENT_FLAGS, SETPOINT_EVENT_FLAGS
        event_cols = list(ACTUATOR_EVENT_FLAGS) + list(SETPOINT_EVENT_FLAGS)
    event_cols = [c for c in event_cols if c in df.columns]
    if not event_cols:
        return weights

    events = df[event_cols].fillna(0).to_numpy(dtype=np.float32, copy=True)
    t0_idx = meta['t0_idx'].to_numpy(dtype=np.int64)

    if event_target_mode == EVENT_TARGET_MODE_MATCHED:
        profiles = np.stack(
            [_event_target_profile(c, target_cols) for c in event_cols],
            axis=0,
        )
        active = profiles.sum(axis=1) > 0
        if not active.any():
            return weights

        if mode == 'event_1h':
            window = max(int(event_window_steps), 1)
            for i, t0 in enumerate(t0_idx):
                horizon_events = events[t0:min(t0 + cfg.horizon, len(events))]
                for col_i in np.flatnonzero(active):
                    for e in np.flatnonzero(horizon_events[:, col_i] > 0):
                        hi = min(cfg.horizon, e + window)
                        boost = 1.0 + float(event_loss_lambda) * profiles[col_i]
                        weights[i, e:hi, :] = np.maximum(weights[i, e:hi, :], boost)
            return weights

        window = max(int(event_decay_window_steps), 1)
        tau = max(float(event_decay_tau_steps), 1.0)
        decay = float(event_loss_lambda) * np.exp(
            -np.arange(window, dtype=np.float32) / tau
        )
        for i, t0 in enumerate(t0_idx):
            horizon_events = events[t0:min(t0 + cfg.horizon, len(events))]
            for col_i in np.flatnonzero(active):
                for e in np.flatnonzero(horizon_events[:, col_i] > 0):
                    hi = min(cfg.horizon, e + window)
                    boost = 1.0 + decay[:hi - e, None] * profiles[col_i][None, :]
                    weights[i, e:hi, :] = np.maximum(weights[i, e:hi, :], boost)
        return weights

    event_any = (events > 0).any(axis=1)

    if mode == 'event_1h':
        window = max(int(event_window_steps), 1)
        boost_value = 1.0 + float(event_loss_lambda)
        for i, t0 in enumerate(t0_idx):
            horizon_events = event_any[t0:min(t0 + cfg.horizon, len(event_any))]
            for e in np.flatnonzero(horizon_events):
                hi = min(cfg.horizon, e + window)
                weights[i, e:hi] = np.maximum(weights[i, e:hi], boost_value)
        return weights

    window = max(int(event_decay_window_steps), 1)
    tau = max(float(event_decay_tau_steps), 1.0)
    decay = float(event_loss_lambda) * np.exp(-np.arange(window, dtype=np.float32) / tau)
    for i, t0 in enumerate(t0_idx):
        horizon_events = event_any[t0:min(t0 + cfg.horizon, len(event_any))]
        for e in np.flatnonzero(horizon_events):
            hi = min(cfg.horizon, e + window)
            boost = 1.0 + decay[:hi - e]
            weights[i, e:hi] = np.maximum(weights[i, e:hi], boost)
    return weights


def _event_target_profile(event_col: str, target_cols: Sequence[str]) -> np.ndarray:
    """Return target multipliers for one control-event column.

    Multipliers encode the expected directness of the control pathway:
    1.0 = direct target emphasis, 0.5 = secondary coupling, 0.0 = no boost.
    """
    profile_by_target = {name: 0.0 for name in target_cols}

    if event_col in {
        'VentLee_up', 'VentLee_down',
        'Ventwind_up', 'Ventwind_down',
        't_vent_sp_changed',
    }:
        profile_by_target.update({'Tair': 1.0, 'Rhair': 1.0, 'CO2air': 1.0})
    elif event_col in {'co2_dos_on', 'co2_sp_changed'}:
        profile_by_target.update({'CO2air': 1.0})
    elif event_col == 't_heat_sp_changed':
        profile_by_target.update({'Tair': 1.0, 'Rhair': 0.5})
    elif event_col in {
        'BlackScr_up', 'BlackScr_down',
        'EnScr_up', 'EnScr_down',
        'scr_blck_sp_changed', 'scr_enrg_sp_changed',
    }:
        profile_by_target.update({'Tair': 1.0, 'Rhair': 0.5})

    return np.asarray([profile_by_target.get(t, 0.0) for t in target_cols],
                      dtype=np.float32)


# ---------------------------------------------------------------------------
# Bulk helpers

def load_processed(out_dir: Path, compartment: str) -> pd.DataFrame:
    return pd.read_parquet(Path(out_dir) / f'{compartment}.parquet')


def load_artifacts(out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    with open(out_dir / 'scalers.pkl', 'rb') as f:
        scalers = pickle.load(f)
    with open(out_dir / 'columns.pkl', 'rb') as f:
        cols = pickle.load(f)
    return {'scalers': scalers, 'columns': cols}


def default_feature_cols(cols_info: dict) -> list[str]:
    """numeric (scaled) + flag (0/1) 모두 X에 포함."""
    return list(cols_info['numeric_cols']) + list(cols_info['flag_cols'])


def make_split_windows(
    out_dir: Path,
    compartment: str,
    cfg: WindowConfig,
    target_cols: Sequence[str] = DEFAULT_TARGETS,
    splits: Sequence[str] = ('train', 'val', 'test'),
    feature_cols: Sequence[str] | None = None,
    event_weight_mode: str = EVENT_WEIGHT_NONE,
    event_target_mode: str = EVENT_TARGET_MODE_SHARED,
    event_loss_lambda: float = 0.5,
    event_window_steps: int = 12,
    event_decay_window_steps: int = 72,
    event_decay_tau_steps: int = 24,
    event_cols: Sequence[str] | None = None,
) -> dict:
    """단일 compartment에 대해 train/val/test 각각의 (X, Y, meta) 생성.

    Parameters
    ----------
    feature_cols : 직접 컬럼 list 지정. None이면 ALL_FEATURES (53개 curated set) 사용.

    Returns
    -------
    {'train': {'X':..., 'Y':..., 'meta':...}, 'val': {...}, 'test': {...},
     'feature_cols': [...], 'target_cols': [...], 'cfg': cfg, 'compartment': str}
    """
    art = load_artifacts(out_dir)
    df = load_processed(out_dir, compartment)
    scaler = art['scalers'][compartment]
    numeric_cols = art['columns']['numeric_cols']

    if feature_cols is None:
        from src.data.feature_groups import get_feature_cols  # late import
        feature_cols = get_feature_cols(available_cols=df.columns)
    else:
        feature_cols = list(feature_cols)

    out = {'feature_cols': feature_cols,
           'target_cols': list(target_cols),
           'cfg': cfg, 'compartment': compartment}
    for split in splits:
        sub = df[df['split'] == split]
        X, Y, meta = make_windows_scaled(
            sub, scaler, numeric_cols, feature_cols, target_cols, cfg)
        split_out = {'X': X, 'Y': Y, 'meta': meta}
        event_weight = make_event_weights(
            sub,
            meta,
            cfg,
            target_cols=target_cols,
            mode=event_weight_mode,
            event_target_mode=event_target_mode,
            event_cols=event_cols,
            event_loss_lambda=event_loss_lambda,
            event_window_steps=event_window_steps,
            event_decay_window_steps=event_decay_window_steps,
            event_decay_tau_steps=event_decay_tau_steps,
        )
        if event_weight is not None:
            split_out['event_weight'] = event_weight
        out[split] = split_out
    return out


def count_windows_per_split(
    out_dir: Path,
    cfg: WindowConfig,
    target_cols: Sequence[str] = DEFAULT_TARGETS,
    compartments: Sequence[str] | None = None,
) -> pd.DataFrame:
    """모든 compartment × split의 clean window 수."""
    if compartments is None:
        # Late import to avoid circular
        from src.data.preprocessing import COMPARTMENTS
        compartments = COMPARTMENTS

    rows = []
    for c in compartments:
        df = load_processed(out_dir, c)
        for split in ('train', 'val', 'test'):
            sub = df[df['split'] == split]
            starts = clean_window_starts(sub, target_cols, cfg)
            total = len(get_window_starts(len(sub), cfg))
            rows.append({
                'comp': c, 'split': split,
                'n_clean': len(starts),
                'n_total': total,
                'clean_pct': len(starts) / max(total, 1) * 100,
            })
    return pd.DataFrame(rows)
