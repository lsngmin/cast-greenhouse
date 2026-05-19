"""Feature group definition for control-source ablation.

논문 핵심 ablation:
    sensor only
    sensor + weather
    sensor + weather + actuator state (state + actuator output + event flags)
    sensor + weather + state + setpoint (sp values + change flags)
    sensor + weather + state + sp + VIP (realized intent)

`get_feature_cols(group_name, available_cols)` — 그룹에 정의된 컬럼 중 df에 실재하는 것만 반환.
"""
from __future__ import annotations
from typing import Sequence


# ---------------------------------------------------------------------------
# Atomic groups

# 순수 환경 sensor (도메인 측정값)
SENSOR_ONLY = [
    'Tair', 'Rhair', 'CO2air', 'HumDef', 'VPD',
    'Tot_PAR', 'EC_drain_PC', 'pH_drain_PC',
]

# 외부 기상
WEATHER = [
    'Tout', 'Rhout', 'Iglob', 'PARout', 'Pyrgeo',
    'RadSum', 'Rain', 'Windsp', 'Winddir', 'AbsHumOut',
]

# Actuator state (0-100% 연속값)
ACTUATOR_STATE_RAW = [
    'VentLee', 'Ventwind', 'AssimLight', 'EnScr', 'BlackScr',
]

# Actuator의 결과/누적 (heating, lamps, irrigation, CO2 dosing)
ACTUATOR_OUTPUT = [
    'PipeLow', 'PipeGrow',
    'Tot_PAR_Lamps',
    'Cum_irr', 'water_sup',
    'co2_dos',
]

# Actuator event flags (binary 0/1)
ACTUATOR_EVENT_FLAGS = [
    'VentLee_up', 'VentLee_down',
    'Ventwind_up', 'Ventwind_down',
    'BlackScr_up', 'BlackScr_down',
    'EnScr_up', 'EnScr_down',
    'co2_dos_on',          # rise event (0 → positive)
]

# Setpoint (requested control target)
SETPOINT_COLS = [
    'co2_sp', 't_heat_sp', 't_vent_sp',
    'scr_blck_sp', 'scr_enrg_sp',
    'assim_sp', 'dx_sp',
    'int_blue_sp', 'int_farred_sp', 'int_red_sp', 'int_white_sp',
    't_grow_min_sp', 't_rail_min_sp', 'window_pos_lee_sp',
    'water_sup_intervals_sp_min',
]

SETPOINT_EVENT_FLAGS = [
    'co2_sp_changed', 't_heat_sp_changed', 't_vent_sp_changed',
    'scr_blck_sp_changed', 'scr_enrg_sp_changed',
]

# VIP (realized intent — 실제 적용된 control value)
VIP_COLS = [
    'co2_vip', 't_heat_vip', 't_ventlee_vip', 't_ventwind_vip',
    'scr_blck_vip', 'scr_enrg_vip',
    'assim_vip', 'dx_vip',
    'int_blue_vip', 'int_farred_vip', 'int_red_vip', 'int_white_vip',
    't_grow_min_vip', 't_rail_min_vip', 'window_pos_lee_vip',
    'water_sup_intervals_vip_min',
]


# ---------------------------------------------------------------------------
# Cumulative feature groups — control source ablation

FEATURE_GROUPS: dict[str, list[str]] = {
    'sensor':
        SENSOR_ONLY,
    'sensor+weather':
        SENSOR_ONLY + WEATHER,
    'sensor+weather+state':
        SENSOR_ONLY + WEATHER
        + ACTUATOR_STATE_RAW + ACTUATOR_OUTPUT
        + ACTUATOR_EVENT_FLAGS,
    'sensor+weather+state+sp':
        SENSOR_ONLY + WEATHER
        + ACTUATOR_STATE_RAW + ACTUATOR_OUTPUT
        + ACTUATOR_EVENT_FLAGS
        + SETPOINT_COLS + SETPOINT_EVENT_FLAGS,
    'sensor+weather+state+sp+vip':
        SENSOR_ONLY + WEATHER
        + ACTUATOR_STATE_RAW + ACTUATOR_OUTPUT
        + ACTUATOR_EVENT_FLAGS
        + SETPOINT_COLS + SETPOINT_EVENT_FLAGS
        + VIP_COLS,
}

GROUP_ORDER = list(FEATURE_GROUPS.keys())


# ---------------------------------------------------------------------------
# Helpers

def get_feature_cols(group_name: str,
                     available_cols: Sequence[str] | None = None) -> list[str]:
    """그룹에 정의된 컬럼 중 df에 실재하는 것만 반환.

    Parameters
    ----------
    group_name : 'sensor', 'sensor+weather', ...
    available_cols : df.columns (없으면 모든 정의된 컬럼 반환).

    Returns
    -------
    list of column names (정의 순서 유지).
    """
    if group_name not in FEATURE_GROUPS:
        raise ValueError(f"Unknown group '{group_name}'. "
                         f"Available: {list(FEATURE_GROUPS)}")
    cols = FEATURE_GROUPS[group_name]
    if available_cols is None:
        return list(cols)
    avail = set(available_cols)
    return [c for c in cols if c in avail]


def list_feature_groups(available_cols: Sequence[str] | None = None) -> dict:
    """모든 그룹의 컬럼 수 (df 기준) 요약."""
    out = {}
    for name in GROUP_ORDER:
        cols = get_feature_cols(name, available_cols)
        out[name] = {'n_cols': len(cols), 'cols': cols}
    return out
