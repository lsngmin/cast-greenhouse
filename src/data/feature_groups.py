"""Feature definition for greenhouse microclimate forecasting.

총 **53개 피쳐** — chain-preserving curation.
원본 74개에서 21개 제거. 제거 기준·근거·실험 결과는 `feature_selection_results.md` 참고.

핵심 원칙: 각 제어축마다 `sp → VIP → actuator → response` 4단 체인 보존.

API:
    ALL_FEATURES           — 53개 피쳐 정의된 순서대로 (list[str])
    get_feature_cols(df)   — ALL_FEATURES 중 df.columns에 실재하는 것만 반환
"""
from __future__ import annotations
from typing import Sequence


# ---------------------------------------------------------------------------
# Atomic groups
#
# 제거된 21개 피쳐 (참고용 — 자세한 사유는 feature_selection_results.md):
#   Sensor (3): VPD, EC_drain_PC, pH_drain_PC
#   Weather (4): Pyrgeo, RadSum, Winddir, AbsHumOut
#   Setpoint (7): int_{blue,farred,red,white}_sp, t_grow_min_sp, t_rail_min_sp,
#                 water_sup_intervals_sp_min
#   VIP (7):      int_{blue,farred,red,white}_vip, t_grow_min_vip, t_rail_min_vip,
#                 water_sup_intervals_vip_min
# 모든 actuator state/output/event, setpoint event는 그대로 유지.

# 순수 환경 sensor (5개) — 미기상 타깃 3개 + 보조 2개
SENSOR_ONLY = [
    'Tair', 'Rhair', 'CO2air', 'HumDef', 'Tot_PAR',
]

# 외부 기상 (6개)
WEATHER = [
    'Tout', 'Rhout', 'Iglob', 'PARout', 'Rain', 'Windsp',
]

# Actuator state (5개, 0-100% 연속값)
ACTUATOR_STATE_RAW = [
    'VentLee', 'Ventwind', 'AssimLight', 'EnScr', 'BlackScr',
]

# Actuator의 결과/누적 (6개) — heating, lamps, irrigation, CO2 dosing
ACTUATOR_OUTPUT = [
    'PipeLow', 'PipeGrow',
    'Tot_PAR_Lamps',
    'Cum_irr', 'water_sup',
    'co2_dos',
]

# Actuator event flags (9개, binary 0/1) — event-window 응답 t=0 마커
ACTUATOR_EVENT_FLAGS = [
    'VentLee_up', 'VentLee_down',
    'Ventwind_up', 'Ventwind_down',
    'BlackScr_up', 'BlackScr_down',
    'EnScr_up', 'EnScr_down',
    'co2_dos_on',          # rise event (0 → positive)
]

# Setpoint (8개) — 운영자 의도
SETPOINT_COLS = [
    'co2_sp', 't_heat_sp', 't_vent_sp',
    'scr_blck_sp', 'scr_enrg_sp',
    'assim_sp', 'dx_sp',
    'window_pos_lee_sp',
]

# Setpoint change events (5개) — event-window 트리거 t=0 마커
SETPOINT_EVENT_FLAGS = [
    'co2_sp_changed', 't_heat_sp_changed', 't_vent_sp_changed',
    'scr_blck_sp_changed', 'scr_enrg_sp_changed',
]

# VIP (9개) — realized intent, 실제 적용된 control value
VIP_COLS = [
    'co2_vip', 't_heat_vip', 't_ventlee_vip', 't_ventwind_vip',
    'scr_blck_vip', 'scr_enrg_vip',
    'assim_vip', 'dx_vip',
    'window_pos_lee_vip',
]


# ---------------------------------------------------------------------------
# Full 53-feature input set (cumulative across atomic groups)

ALL_FEATURES: list[str] = (
    SENSOR_ONLY + WEATHER
    + ACTUATOR_STATE_RAW + ACTUATOR_OUTPUT
    + ACTUATOR_EVENT_FLAGS
    + SETPOINT_COLS + SETPOINT_EVENT_FLAGS
    + VIP_COLS
)
assert len(ALL_FEATURES) == 53, f"expected 53 features, got {len(ALL_FEATURES)}"


# ---------------------------------------------------------------------------
# Control source partition — used by SourceAwareEmbedding
#
# 5개 제어 출처로 53개 피쳐를 분할. 각 소스는 모델 안에서 독립 임베딩 파라미터를
# 갖고, 최종 결합 시 dynamic gate로 (B, L)별 가중치가 부여됨.
#
#   sensor    (5):  내부 환경 측정값
#   weather   (6):  외부 기상 (boundary condition)
#   actuator  (20): 상태 + 출력 + 이벤트 플래그 (체인의 중간 단)
#   setpoint  (13): sp 값 + 변경 이벤트 (체인의 시작점, 운영자 의도)
#   vip       (9):  실현 의도 (체인의 sp ↔ actuator 사이 단)

FEATURE_SOURCES: dict[str, list[str]] = {
    'sensor':   SENSOR_ONLY,
    'weather':  WEATHER,
    'actuator': ACTUATOR_STATE_RAW + ACTUATOR_OUTPUT + ACTUATOR_EVENT_FLAGS,
    'setpoint': SETPOINT_COLS + SETPOINT_EVENT_FLAGS,
    'vip':      VIP_COLS,
}

# Validate: union should equal ALL_FEATURES and partitions are disjoint
_union = sum((cols for cols in FEATURE_SOURCES.values()), [])
assert sorted(_union) == sorted(ALL_FEATURES), \
    "FEATURE_SOURCES partition does not cover ALL_FEATURES"
assert len(_union) == len(set(_union)), "FEATURE_SOURCES partitions overlap"
del _union

SOURCE_ORDER: tuple[str, ...] = tuple(FEATURE_SOURCES.keys())


def source_indices(feature_cols: Sequence[str]) -> dict[str, list[int]]:
    """For each source, return indices into `feature_cols` of its features.

    Sources with no matching column in `feature_cols` are omitted.

    Used by SourceAwareEmbedding to slice the input tensor per source.
    """
    feature_to_idx = {c: i for i, c in enumerate(feature_cols)}
    out = {}
    for name in SOURCE_ORDER:
        idxs = [feature_to_idx[c] for c in FEATURE_SOURCES[name] if c in feature_to_idx]
        if idxs:
            out[name] = idxs
    return out


# ---------------------------------------------------------------------------
# Helper

def get_feature_cols(available_cols: Sequence[str] | None = None) -> list[str]:
    """Return ALL_FEATURES, filtered to columns present in `available_cols` if given.

    Parameters
    ----------
    available_cols : df.columns (없으면 전체 53개 반환).

    Returns
    -------
    list of column names (정의 순서 유지).
    """
    if available_cols is None:
        return list(ALL_FEATURES)
    avail = set(available_cols)
    return [c for c in ALL_FEATURES if c in avail]
