"""Forecasting metrics for AGC2 microclimate prediction.

기본 metric:
  MAE, RMSE, NMAE, NRMSE, MASE, R²

확장:
  - per-variable, per-horizon-step
  - cumulative-over-horizon
  - event-window MAE (event 시점 t = forecast 시작 후 step)

inverse_transform_Y: scaled Y를 원래 단위로 복원해서 metric 계산.

설계 원칙:
  - 모든 함수는 (n_samples, H, V) tensor input을 받음
  - NaN-aware (np.nanmean 등 사용)
  - axis 인자로 집계 차원 선택
"""
from __future__ import annotations
from typing import Sequence
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Basic point metrics

def mae(y_true, y_pred, axis=None):
    return np.nanmean(np.abs(y_true - y_pred), axis=axis)


def rmse(y_true, y_pred, axis=None):
    return np.sqrt(np.nanmean((y_true - y_pred) ** 2, axis=axis))


def r2(y_true, y_pred, axis=None):
    """Coefficient of determination. axis=None → scalar, axis=0 → per-feature."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = np.nansum((y_true - y_pred) ** 2, axis=axis)
    if axis is None:
        mean = np.nanmean(y_true)
        ss_tot = np.nansum((y_true - mean) ** 2)
    else:
        mean = np.nanmean(y_true, axis=axis, keepdims=True)
        ss_tot = np.nansum((y_true - mean) ** 2, axis=axis)
    return 1.0 - ss_res / np.maximum(ss_tot, 1e-12)


def nmae(y_true, y_pred, scale, axis=None):
    """NMAE = MAE / scale. scale: scalar or (V,) array."""
    return mae(y_true, y_pred, axis=axis) / np.asarray(scale)


def nrmse(y_true, y_pred, scale, axis=None):
    return rmse(y_true, y_pred, axis=axis) / np.asarray(scale)


def mase(y_true, y_pred, naive_mae_scale, axis=None):
    """MASE = MAE / mean(|naive_baseline_error|). naive scale 미리 계산 필요."""
    return mae(y_true, y_pred, axis=axis) / np.asarray(naive_mae_scale)


# ---------------------------------------------------------------------------
# Naive baseline (persistence) — for MASE denominator

def persistence_predict(X_last_target: np.ndarray, horizon: int) -> np.ndarray:
    """Replicate the last observed target value for the entire horizon.

    Parameters
    ----------
    X_last_target : (n_samples, V) — target values at t0 (forecast issue time)
    horizon : H

    Returns
    -------
    (n_samples, H, V) — same value repeated horizon times.
    """
    n, v = X_last_target.shape
    return np.broadcast_to(X_last_target[:, None, :], (n, horizon, v)).copy()


def compute_naive_mae_scale(df_train: pd.DataFrame, target_cols: Sequence[str],
                            horizon_steps: int) -> np.ndarray:
    """Persistence-baseline MAE on train (MASE 분모).

    naive[t+h] = y[t] → |y[t+h] - y[t]|
    """
    arr = df_train[list(target_cols)].values  # (T, V)
    diffs = arr[horizon_steps:] - arr[:-horizon_steps]
    return np.nanmean(np.abs(diffs), axis=0)


# ---------------------------------------------------------------------------
# Per-horizon helpers

def per_step_metric(y_true: np.ndarray, y_pred: np.ndarray, metric_fn=mae):
    """Per-time-step metric. Returns (H, V) array.

    Each (h, v) cell = metric over n_samples for that horizon step and variable.
    """
    # y_true, y_pred: (n, H, V) → reduce axis=0 (samples)
    return metric_fn(y_true, y_pred, axis=0)


HORIZON_KEY_STEPS = {
    '1h':  12,   # 1h = 12 step (5min × 12)
    '6h':  72,
    '24h': 288,
}


def horizon_summary(y_true: np.ndarray, y_pred: np.ndarray,
                    target_names: Sequence[str],
                    key_steps: dict = None,
                    naive_mae_scale: np.ndarray | None = None) -> pd.DataFrame:
    """Per-variable metric at key horizon points (point) + cumulative.

    Returns a DataFrame with rows = (variable, horizon, kind),
    columns = [MAE, RMSE, R2, MASE (if naive provided)].
    """
    if key_steps is None:
        key_steps = HORIZON_KEY_STEPS

    rows = []
    for var_i, name in enumerate(target_names):
        yt_v = y_true[:, :, var_i]   # (n, H)
        yp_v = y_pred[:, :, var_i]

        for label, step in key_steps.items():
            # Point metric at step h-1 (1-indexed → 0-indexed)
            idx = min(step - 1, yt_v.shape[1] - 1)
            mae_pt = mae(yt_v[:, idx], yp_v[:, idx])
            rmse_pt = rmse(yt_v[:, idx], yp_v[:, idx])
            r2_pt = r2(yt_v[:, idx], yp_v[:, idx])
            row = {'variable': name, 'horizon': label, 'kind': 'point',
                   'MAE': mae_pt, 'RMSE': rmse_pt, 'R2': r2_pt}
            if naive_mae_scale is not None:
                row['MASE'] = mae_pt / naive_mae_scale[var_i]
            rows.append(row)

            # Cumulative metric over 0:step
            end = min(step, yt_v.shape[1])
            mae_c = mae(yt_v[:, :end], yp_v[:, :end])
            rmse_c = rmse(yt_v[:, :end], yp_v[:, :end])
            r2_c = r2(yt_v[:, :end], yp_v[:, :end])
            row = {'variable': name, 'horizon': label, 'kind': 'cumulative',
                   'MAE': mae_c, 'RMSE': rmse_c, 'R2': r2_c}
            if naive_mae_scale is not None:
                row['MASE'] = mae_c / naive_mae_scale[var_i]
            rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Event-window MAE

def event_window_mae(y_true: np.ndarray, y_pred: np.ndarray,
                     event_starts: list[tuple[int, int]],
                     window_steps: int,
                     target_index: int | None = None) -> dict:
    """Event 시점 직후 window의 MAE.

    Parameters
    ----------
    y_true, y_pred : (n_samples, H, V)
    event_starts : list of (sample_i, t_step_in_horizon)
        sample_i: which forecasting sample
        t_step:  event 발생 시점 (forecast horizon 내 step idx, 0-indexed)
    window_steps : event 후 몇 step까지 포함할지
    target_index : 특정 variable만 보고 싶을 때. None이면 전 variable.

    Returns
    -------
    dict with 'mean', 'std', 'n', and per-event errors.
    """
    n_h = y_true.shape[1]
    errs = []
    for i, t in event_starts:
        if t < 0 or t >= n_h:
            continue
        end = min(t + window_steps, n_h)
        if target_index is None:
            seg_true = y_true[i, t:end]            # (W, V)
            seg_pred = y_pred[i, t:end]
        else:
            seg_true = y_true[i, t:end, target_index]   # (W,)
            seg_pred = y_pred[i, t:end, target_index]
        err = np.nanmean(np.abs(seg_true - seg_pred))
        errs.append(err)

    errs = np.array(errs) if errs else np.array([np.nan])
    return {
        'mean': float(np.nanmean(errs)),
        'std':  float(np.nanstd(errs)),
        'n':    int(np.sum(~np.isnan(errs))),
        'per_event': errs,
    }


def detect_event_steps_in_forecast(
    df_full: pd.DataFrame,
    meta: pd.DataFrame,
    event_col: str,
    cfg_horizon: int,
) -> list[tuple[int, int]]:
    """Forecast horizon 내의 event step list 생성.

    Parameters
    ----------
    df_full : original processed df (with all flag cols).
    meta    : window meta from make_windows (has t0_idx column).
    event_col : binary flag column in df_full (예: 'VentLee_up', 'co2_sp_changed').
    cfg_horizon : H (steps).

    Returns
    -------
    list of (sample_i, step_idx_in_horizon).
    """
    if event_col not in df_full.columns:
        return []
    ev_arr = df_full[event_col].values.astype(bool)
    out = []
    for i, row in meta.reset_index(drop=True).iterrows():
        t0 = int(row['t0_idx'])
        for t in range(cfg_horizon):
            if t0 + t < len(ev_arr) and ev_arr[t0 + t]:
                out.append((i, t))
    return out


# ---------------------------------------------------------------------------
# Inverse transform Y (scaled → original units)

def inverse_transform_Y(Y_scaled: np.ndarray,
                        scaler,
                        target_cols: Sequence[str],
                        numeric_cols: Sequence[str]) -> np.ndarray:
    """Y_scaled (n, H, V) → 원 단위.

    scaler는 numeric_cols 전체에 대해 fit되었으므로 target에 해당하는 idx 추출.
    """
    target_idx = [list(numeric_cols).index(c) for c in target_cols]
    means = scaler.mean_[target_idx]              # (V,)
    stds = scaler.scale_[target_idx]              # (V,)
    return Y_scaled * stds + means


# ---------------------------------------------------------------------------
# Variable scales (for NMAE/NRMSE) — use train std

def target_scales(df_train: pd.DataFrame, target_cols: Sequence[str]) -> np.ndarray:
    """NMAE/NRMSE의 normalization scale = train std per variable (unscaled)."""
    return df_train[list(target_cols)].std(ddof=0).values


# ---------------------------------------------------------------------------
# VPD as derived variable
#
# Target은 [Tair, Rhair, CO2air] 3개만. VPD는 직접 예측하지 않고
# 예측된 Tair/Rhair로 계산하여 metric 산출. 이렇게 해야
# vpd_pred = f(tair_pred, rhair_pred) ≡ 모델 출력의 물리적 일관성 유지.

def vpd_from_tair_rh(tair_c, rh_pct):
    """VPD (kPa) — Magnus formula. numpy/pandas 모두 OK."""
    es = 0.6108 * np.exp(17.27 * tair_c / (tair_c + 237.3))
    return es * (1.0 - rh_pct / 100.0)


def derive_vpd_from_targets(Y_orig: np.ndarray,
                            target_cols: Sequence[str]) -> np.ndarray:
    """Y_orig: (..., V) 원 단위 target tensor. Returns VPD (...) — V축 제거.

    target_cols에 'Tair'와 'Rhair'가 반드시 있어야 함.
    """
    cols = list(target_cols)
    i_t = cols.index('Tair')
    i_r = cols.index('Rhair')
    return vpd_from_tair_rh(Y_orig[..., i_t], Y_orig[..., i_r])


def vpd_metric_from_targets(y_true_orig: np.ndarray, y_pred_orig: np.ndarray,
                            target_cols: Sequence[str],
                            metric_fn=mae, **kwargs):
    """예측 Tair/Rhair로 계산한 VPD vs 실제 Tair/Rhair로 계산한 VPD 비교.

    Returns (metric_value, vpd_true_array, vpd_pred_array)
    """
    vpd_true = derive_vpd_from_targets(y_true_orig, target_cols)
    vpd_pred = derive_vpd_from_targets(y_pred_orig, target_cols)
    return metric_fn(vpd_true, vpd_pred, **kwargs), vpd_true, vpd_pred
