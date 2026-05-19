"""Preprocessing pipeline for AGC2 dataset.

EDA 단계 (notebooks/eda/01~03)에서 검증된 정책을 모델 학습용 dataset으로 고정한다.

핵심 원칙:
  - Scaler는 train split에서만 fit (val/test 정보 누설 방지)
  - Setpoint는 piecewise-constant signal → ffill + change-event flag
  - Actuator는 방향 분리 (_up / _down) event flag
  - 이상치는 물리 경계 밖 값만 NaN (derivative spike 처리는 일단 보류)

사용:
    from src.preprocessing import process_all_compartments, SplitFractions
    results = process_all_compartments(Path('data/raw'), Path('data/processed'))
"""
from __future__ import annotations
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Constants

COMPARTMENTS = ('AICU', 'Automatoes', 'Digilog', 'IUACAAS', 'Reference', 'TheAutomators')

TARGETS = ('Tair', 'Rhair', 'CO2air')

ACTUATORS = ('VentLee', 'Ventwind', 'AssimLight', 'EnScr', 'BlackScr',
             'PipeLow', 'PipeGrow', 'co2_dos', 'water_sup')

SETPOINTS = ('co2_sp', 't_heat_sp', 't_vent_sp', 'scr_blck_sp', 'scr_enrg_sp')

VIPS = ('co2_vip', 't_heat_vip', 't_ventlee_vip', 't_ventwind_vip',
        'scr_blck_vip', 'scr_enrg_vip')

WEATHER_COLS = ('Tout', 'Rhout', 'Iglob', 'PARout', 'Pyrgeo', 'RadSum',
                'Rain', 'Windsp', 'Winddir', 'AbsHumOut')

# Physical bounds — outlier_probe.py 결과 기반
PHYSICAL_BOUNDS = {
    'Tair':   (0,   50),
    'Rhair':  (0,  100),
    'CO2air': (200, 3000),
    'HumDef': (0,   30),
}

# Actuator change thresholds — notebooks/eda/02 결과와 일치
ACTUATOR_CHANGE_TH = {
    'VentLee':  10.0,
    'Ventwind': 10.0,
    'BlackScr': 20.0,
    'EnScr':    20.0,
}


# ---------------------------------------------------------------------------
# Loading

def excel_to_dt(s):
    """Excel serial date (origin 1899-12-30) → pandas datetime."""
    return pd.to_datetime(s, unit='D', origin='1899-12-30')


def load_compartment(root: Path, compartment: str) -> pd.DataFrame:
    """Load GreenhouseClimate.csv.

    - skipinitialspace=True: CSV의 31k+ 행이 ' 18.1' 처럼 선행공백 padded
    - timestamp는 5분 grid로 snap, 중복 제거, sort
    """
    fp = Path(root) / compartment / 'GreenhouseClimate.csv'
    df = pd.read_csv(fp, skipinitialspace=True)
    df['ts'] = excel_to_dt(df['%time']).dt.round('5min')
    df = df.drop(columns=['%time']).set_index('ts').sort_index()
    df = df[~df.index.duplicated(keep='first')]
    return df


def load_weather(root: Path) -> pd.DataFrame:
    fp = Path(root) / 'Weather' / 'Weather.csv'
    df = pd.read_csv(fp, skipinitialspace=True)
    df['ts'] = excel_to_dt(df['%time']).dt.round('5min')
    df = df.drop(columns=['%time']).set_index('ts').sort_index()
    df = df[~df.index.duplicated(keep='first')]
    return df


# ---------------------------------------------------------------------------
# Outlier handling

def clip_physical(df: pd.DataFrame, bounds=None) -> pd.DataFrame:
    """Replace out-of-physical-range values with NaN.

    Reference compartment에서 Tair=-1, Rhair=-22, CO2air=-499 같은 sensor 오류 8행 제거.
    다른 compartment에는 영향 없음.
    """
    if bounds is None:
        bounds = PHYSICAL_BOUNDS
    df = df.copy()
    for col, (lo, hi) in bounds.items():
        if col in df.columns:
            df[col] = df[col].where((df[col] >= lo) & (df[col] <= hi))
    return df


# ---------------------------------------------------------------------------
# Derived features

def compute_vpd(tair_c: pd.Series, rh_pct: pd.Series) -> pd.Series:
    """VPD (kPa) — Magnus formula.

    es(T) = 0.6108 * exp(17.27 * T / (T + 237.3))
    VPD = es(T) * (1 - RH/100)
    """
    es = 0.6108 * np.exp(17.27 * tair_c / (tair_c + 237.3))
    return es * (1.0 - rh_pct / 100.0)


def add_vpd(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'Tair' in df.columns and 'Rhair' in df.columns:
        df['VPD'] = compute_vpd(df['Tair'], df['Rhair'])
    return df


# ---------------------------------------------------------------------------
# Event flags

def add_setpoint_change_flags(df: pd.DataFrame,
                              setpoint_cols=SETPOINTS,
                              ffill: bool = True) -> pd.DataFrame:
    """Setpoint signal은 piecewise-constant. 값 자체는 ffill로 dense화, 변경 시점에 flag.

    추가 컬럼: <sp>_changed (int8, 0/1)
    """
    df = df.copy()
    for sp in setpoint_cols:
        if sp not in df.columns:
            continue
        s_ffill = df[sp].ffill()
        changed = (s_ffill.diff().fillna(0).abs() > 0).astype(np.int8)
        df[f'{sp}_changed'] = changed.values
        if ffill:
            df[sp] = s_ffill
    return df


def add_actuator_change_flags(df: pd.DataFrame,
                              thresholds=None) -> pd.DataFrame:
    """Actuator state 변화를 방향별로 분리.

    추가 컬럼: <act>_up, <act>_down (int8, 0/1)
    """
    if thresholds is None:
        thresholds = ACTUATOR_CHANGE_TH
    df = df.copy()
    for col, th in thresholds.items():
        if col not in df.columns:
            continue
        d = df[col].ffill().diff()
        df[f'{col}_up']   = (d >=  th).fillna(False).astype(np.int8).values
        df[f'{col}_down'] = (d <= -th).fillna(False).astype(np.int8).values
    return df


def add_co2_dos_on_flag(df: pd.DataFrame, col: str = 'co2_dos') -> pd.DataFrame:
    """CO2 dosing이 OFF → ON으로 전이하는 시점 = 'co2_dos_on' event flag.

    threshold-based diff와 달리 rise 패턴 (0 → positive). 02 노트북에서 핵심 event.
    """
    df = df.copy()
    if col not in df.columns:
        return df
    s = df[col].ffill()
    prev = s.shift(1).fillna(0)
    df['co2_dos_on'] = ((s > 0) & (prev == 0)).fillna(False).astype(np.int8).values
    return df


# ---------------------------------------------------------------------------
# Weather merge

def merge_weather(df_gc: pd.DataFrame, df_wx: pd.DataFrame) -> pd.DataFrame:
    """timestamp index 기준 left-join."""
    cols = [c for c in WEATHER_COLS if c in df_wx.columns]
    return df_gc.join(df_wx[cols], how='left')


# ---------------------------------------------------------------------------
# Split

@dataclass
class SplitFractions:
    train: float = 0.70
    val:   float = 0.15
    test:  float = 0.15

    def __post_init__(self):
        s = self.train + self.val + self.test
        if not np.isclose(s, 1.0):
            raise ValueError(f'fractions must sum to 1.0, got {s}')


def time_split_indices(n: int, fractions: SplitFractions) -> dict:
    """Time-ordered split, no shuffle. 미래 누설 방지."""
    n_tr = int(n * fractions.train)
    n_va = int(n * fractions.val)
    return {
        'train': (0, n_tr),
        'val':   (n_tr, n_tr + n_va),
        'test':  (n_tr + n_va, n),
    }


def assign_split_column(df: pd.DataFrame, splits: dict) -> pd.DataFrame:
    df = df.copy()
    split_col = np.array(['test'] * len(df), dtype=object)
    split_col[splits['train'][0]:splits['train'][1]] = 'train'
    split_col[splits['val'][0]:splits['val'][1]] = 'val'
    df['split'] = pd.Categorical(split_col, categories=['train', 'val', 'test'])
    return df


# ---------------------------------------------------------------------------
# Scaling

def _select_scaler_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Numeric columns minus flag columns."""
    flag_cols = [c for c in df.columns
                 if c.endswith('_up') or c.endswith('_down') or c.endswith('_changed')]
    numeric_cols = [c for c in df.select_dtypes(include='number').columns
                    if c not in flag_cols]
    return numeric_cols, flag_cols


def fit_standard_scaler(df_train: pd.DataFrame, cols: list[str]) -> StandardScaler:
    """train split에서 컬럼별 NaN 무시하고 fit.

    sklearn StandardScaler는 native NaN handling이 없어서 column-wise nanmean/nanstd로
    수동 채움. 상수 컬럼(std=0)은 scale=1로 처리하여 transform 안전.
    """
    arr = df_train[cols].values.astype(np.float64)
    sc = StandardScaler()
    sc.mean_ = np.nanmean(arr, axis=0)
    sc.var_  = np.nanvar(arr, axis=0)
    sc.scale_ = np.sqrt(sc.var_)
    sc.scale_[sc.scale_ == 0] = 1.0  # constant column → no rescaling
    sc.n_features_in_ = len(cols)
    sc.feature_names_in_ = np.array(cols, dtype=object)
    sc.n_samples_seen_ = np.sum(~np.isnan(arr), axis=0)
    # Replace NaN means/scales (column entirely NaN in train) with 0 / 1
    nan_mean = np.isnan(sc.mean_)
    if nan_mean.any():
        sc.mean_[nan_mean] = 0.0
        sc.scale_[nan_mean] = 1.0
        sc.var_[nan_mean] = 1.0
    return sc


# ---------------------------------------------------------------------------
# Full pipeline per compartment

def process_one_compartment(root: Path, compartment: str, df_wx: pd.DataFrame,
                            fractions: SplitFractions = None) -> dict:
    """Single compartment pipeline.

    Returns dict with df (with split column), scaler, splits, numeric_cols, flag_cols.
    """
    if fractions is None:
        fractions = SplitFractions()
    df = load_compartment(root, compartment)
    df = clip_physical(df)
    df = add_vpd(df)
    df = add_setpoint_change_flags(df)
    df = add_actuator_change_flags(df)
    df = add_co2_dos_on_flag(df)
    df = merge_weather(df, df_wx)

    splits = time_split_indices(len(df), fractions)
    numeric_cols, flag_cols = _select_scaler_columns(df)
    df_train = df.iloc[splits['train'][0]:splits['train'][1]]
    scaler = fit_standard_scaler(df_train, numeric_cols)

    df = assign_split_column(df, splits)
    return {
        'compartment': compartment,
        'df': df,
        'splits': splits,
        'scaler': scaler,
        'numeric_cols': numeric_cols,
        'flag_cols': flag_cols,
    }


def process_all_compartments(root: Path, out_dir: Path,
                             fractions: SplitFractions = None,
                             save: bool = True) -> dict:
    """Process all 6 compartments. Save parquet + scalers."""
    if fractions is None:
        fractions = SplitFractions()
    root = Path(root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_wx = load_weather(root)
    results = {}
    for c in COMPARTMENTS:
        r = process_one_compartment(root, c, df_wx, fractions)
        results[c] = r
        if save:
            r['df'].to_parquet(out_dir / f'{c}.parquet')

    if save:
        with open(out_dir / 'scalers.pkl', 'wb') as f:
            pickle.dump({c: results[c]['scaler'] for c in COMPARTMENTS}, f)
        with open(out_dir / 'columns.pkl', 'wb') as f:
            pickle.dump({
                'numeric_cols': results[COMPARTMENTS[0]]['numeric_cols'],
                'flag_cols':    results[COMPARTMENTS[0]]['flag_cols'],
                'targets':      list(TARGETS),
            }, f)
    return results
