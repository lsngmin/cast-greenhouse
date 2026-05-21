# 연구계획서 (최종본)

## Control-Aware Spatio-Temporal Forecasting for Multi-Horizon Greenhouse Microclimate Prediction: Event-Based Evaluation on the Wageningen Autonomous Greenhouse Dataset

---

## 0. 변경 요약 (v1 → v2 → v3 → v3.1 EDA 반영본)

본 최종본은 데이터 검증 결과 및 EDA(2026-05-19 완료) 결과를 반영하여 다음을 수정함:

1. **데이터 전략 단일화**: 다중 데이터셋(Wageningen + Korean) 구조 → **Wageningen Autonomous Greenhouse Challenge 단일 데이터셋** 집중
2. **Wageningen Autonomous Greenhouse Challenge 2nd Edition 확보 완료**. 5분 자동 로깅, setpoint vs realized(VIP) 분리, 6 compartment, 166일 cherry tomato — 실측 검증됨
3. **Cross-compartment / cross-season generalization을 일반화 검증의 축으로 채택** (동일 데이터셋 내 견고성 검증)
4. **연구 일정 단축**: Week 1 데이터 확보·검증·EDA 단계가 완료됨 (Section 4.6 참조)
5. **위험 요소 축소**: actuator 로그 품질 위험(원안 매우 높음) → 대폭 완화. 외부 데이터 승인 지연 위험 완전 제거
6. **(v3.1) Setpoint signal 표현 정정**: "sparse event log" → "piecewise-constant control target + change event" (EDA에서 setpoint signal 자체가 95%+ obs임을 확인)
7. **(v3.1) Event 정의 방향 분리**: actuator state transition을 `_up` / `_down`으로 분리 (open과 close가 섞이지 않도록)
8. **(v3.1) Lookback 24h 채택 근거 명시**: 4개 메트릭(ACF, STL, FFT, hour-of-day ratio)로 일주기 구조 정량 확인. 다만 최종 lookback은 sensitivity 실험으로 결정
9. **(v3.1) Event-based delayed response 정량 결과 추가** (Section 4.6.2): co2_dos_on +76.5 ppm @ 15분, t_heat_sp_up +4.18 °C @ 355분 등 — multi-horizon 평가의 필요성을 정당화

---

## 1. 연구 배경 및 필요성

스마트팜 온실은 작물 생산성을 결정하는 미기상(microclimate)을 인위적으로 조절하는 시설이다. 내부 온도, 습도, CO₂ 농도, 일사량 등의 환경 변수는 환기, 난방, 차광, 관수, CO₂ 공급 등 다양한 액추에이터(actuator)의 작동에 의해 직간접적으로 영향을 받는다.

미기상 예측은 다음 두 가지 측면에서 중요하다.

첫째, 정확한 미기상 예측은 액추에이터의 선제적 제어(predictive control)를 가능하게 하여 에너지 효율과 작물 품질을 동시에 향상시킨다.

둘째, 환경 변수 간 상호작용과 액추에이터의 지연 효과(예: 환기 작동 후 수십 분 뒤에 나타나는 습도 변화)를 함께 고려해야만 실용적 예측이 가능하다.

기존 연구는 다음과 같은 한계를 가진다.

**한계 1**: 기존 온실 예측 연구는 STGNN을 통해 환경 변수 간 방향성 관계를 모델링하거나, actuator operation state를 활용한 단기 예측을 수행했으나, actuator event 이후의 지연 반응(delayed response)을 multi-horizon 관점에서 구조적으로 분석한 연구는 제한적이다.

**한계 2**: 대부분의 평가는 전체 평균 오차(MAE, RMSE) 중심으로 이루어져, actuator event 구간에서의 예측 품질과 응답 지연(response lag)을 구분해 평가하지 못한다.

**한계 3**: 대부분 단기(5-15분) 예측에 집중되어 있어 농가 운영에 필요한 중장기(1-24시간) 의사결정 지원이 부족하다.

**한계 4**: Setpoint(requested) → Realized(VIP) → Actuator state → 환경 변수로 이어지는 제어 인과 체인에서, setpoint과 realized 신호를 **분리해 활용**한 forecasting 연구가 드물다. 대부분 actuator state(또는 ON/OFF)만 활용한다.

본 연구는 위 한계를 해결하기 위해 control-aware spatio-temporal forecasting 프레임워크를 제안하고, event-based analysis와 lag-aware metric을 통해 actuator event 이후의 지연 반응을 정량화한다. **네덜란드 Wageningen 고기술 온실(Autonomous Greenhouse Challenge) 데이터를 활용**하여 multi-horizon 예측 성능, setpoint/realized 분리 활용의 가치, 동일 데이터셋 내 cross-compartment / cross-season 일반화 가능성을 체계적으로 분석한다.

---

## 2. 연구 문제

### 핵심 문제 정의

> **스마트팜 온실 미기상 예측에서 actuator 제어 로그(상태값, setpoint, realized setpoint)와 환경 변수 간 방향성 의존성을 동시에 반영하고, actuator event 이후의 delayed response를 event-window 기반으로 명시적으로 분석하는 control-aware multi-horizon forecasting 프레임워크를 제안하고, Wageningen Autonomous Greenhouse Challenge 2nd Edition 데이터셋(166일, 6 compartment)에서 검증한다.**

### 연구 질문

| RQ | 질문 |
|---|---|
| RQ1 | Sensor, weather, actuator state, setpoint, realized VIP를 단계적으로 추가할 때, 성능 변화는 target / horizon / event window별로 어떻게 달라지는가? |
| RQ2 | LSTM, Transformer, Mamba temporal encoder는 24h multi-horizon trajectory forecasting에서 어떤 target-specific 및 event-specific inductive bias를 보이는가? |
| RQ3 | Domain-informed prior, learned graph, prior-initialized learned graph는 변수 간 방향성 의존성 모델링에서 각각 어떤 조건부 이득과 한계를 보이는가? |
| RQ4 | Actuator event-window 평가는 전체 평균 MAE/RMSE로는 보이지 않는 delayed response, lag error, target별 실패 양상을 드러내는가? |
| RQ5 | Cross-compartment / cross-season shift에서 deep forecasting model은 persistence 대비 어떤 강점과 한계를 보이며, 어떤 입력 source와 구조가 더 견고한가? |
| RQ6 | Horizon-aware, target-specific decoder는 flat direct MLP decoder 대비 24h trajectory의 horizon별/target별 병목을 완화하는가? |

---

## 3. 연구 가설

| 가설 | 검증 방법 |
|---|---|
| **H1**: 제어 source의 추가 효과는 전체 평균에서 항상 단조 증가하지 않고, target / horizon / event type에 따라 달라진다 | 5 feature group ablation, horizon summary, event-window MAE |
| **H2**: Mamba는 장기 trajectory와 일부 event response에서 경쟁력을 보이지만, LSTM/Transformer 대비 우위는 target과 split 조건에 의존한다 | Backbone ablation, inference time, persistence-relative metric |
| **H3**: Learned graph fusion은 일부 target-event regime에서 이득을 줄 수 있으나, fixed prior graph는 부정확한 edge 또는 계절/compartment shift에서 성능을 저하시킬 수 있다 | No-graph / prior / learned / prior-learned graph ablation |
| **H4**: Event-window metric은 전체 평균 MAE/RMSE와 다른 모델 순위를 만들며, actuator delayed response 예측의 실패 양상을 더 직접적으로 드러낸다 | Event-window MAE, lag error, peak timing error |
| **H5**: Cross-compartment / cross-season 평가에서는 모든 deep model이 persistence와의 격차가 줄어들며, robustness는 target과 control source에 따라 다르게 나타난다 | Cross-compartment 6-fold, train winter → test spring, MASE |
| **H6**: Horizon-aware target-specific decoder는 flat MLP decoder보다 horizon별 출력 구조를 명시적으로 분리하여 target별 trajectory 품질을 개선할 가능성이 있다 | MLPDecoder vs HorizonQueryDecoder ablation |

---

## 4. 해결 방법

### 4.1 데이터

본 연구는 **Wageningen Autonomous Greenhouse Challenge — 2nd Edition (2019)** 단일 데이터셋을 사용한다.

#### Wageningen Autonomous Greenhouse Challenge — 2nd Edition (2019) ✅ 확보 완료

| 항목 | 내용 |
|---|---|
| 출처 | Wageningen University & Research, Bleiswijk, Netherlands |
| 라이센스 | 4TU.ResearchData (open, academic use) |
| 작물 | Cherry tomato (cv. Axiany) |
| 기간 | 2019-12-16 ~ 2020-05-30 (166일, ~5.5개월) |
| Compartment | 6개 (5 AI 팀 + 1 Reference 인간 운영자) |
| 시간 해상도 | **5분 자동 로깅** (median delta 4분 59.808초, 실측 확인) |
| 행 수 | 47,809 rows × 6 compartment |
| 결측률 | sensor/actuator state/VIP: **~0.15%**, setpoint: **95~99% 관측** (piecewise-constant signal, 변경 시점이 event) |

**핵심 변수 (GreenhouseClimate.csv, 50 columns, 47,809 rows)**:

- **센서 (16개)**: `Tair`, `Rhair`, `CO2air`, `HumDef`(VPD 대용), `PipeLow`/`PipeGrow`(난방관 온도), `Tot_PAR`, `Tot_PAR_Lamps`, `EC_drain_PC`, `pH_drain_PC`, `Cum_irr`(누적 관수), `water_sup`(누적 관수시간), `co2_dos`(CO₂ 공급량)
- **Actuator state (5개, 0~100% 연속값)**: `VentLee`(풍하 환기), `Ventwind`(풍상 환기), `AssimLight`(HPS 보광), `EnScr`(에너지 스크린), `BlackScr`(차광막)
- **Setpoints (14개, sp suffix)**: `co2_sp`, `dx_sp`, `t_heat_sp`, `t_vent_sp`, `scr_blck_sp`, `scr_enrg_sp`, `window_pos_lee_sp`, `water_sup_intervals_sp_min`, `t_rail_min_sp`, `t_grow_min_sp`, `Assim_sp`, `int_blue/red/farred/white_sp`. **Piecewise-constant control target signal** (95% 이상 관측, Reference 기준 `t_heat_sp` 6,709회, `t_vent_sp` 7,220회 변경) — 변경 시점 자체를 event signal로 활용
- **Realized VIPs (14개, vip suffix)**: 모든 setpoint과 1:1 매칭되는 실제 적용값 (99.85% obs, dense)

**보조 파일**:
- `Weather.csv` (11 cols, 5분): `Tout`, `Rhout`, `Iglob`, `Windsp`, `Winddir`, `RadSum`, `Rain`, `PARout`, `Pyrgeo`, `AbsHumOut`
- `GrodanSens.csv` (7 cols, 5분): 근권/배지 EC, WC, 온도 × 2 위치
- `Resources.csv` (일별): 난방·전기·CO₂·관수·배액 소비
- `Production`, `CropParameters`, `TomQuality`, `LabAnalysis` (수확/생장/품질)

#### 단일 데이터셋 사용의 정당성

- **5분 자동 로깅 + setpoint/VIP 1:1 분리**는 event-based 분석에 필수적인 시간 해상도 보장
- 6 compartment(5 AI + 1 Reference)를 활용한 **cross-compartment generalization** 평가 가능
- 166일의 winter→spring 기간으로 **cross-season 분할** 평가 가능
- 원안의 actuator 로그 품질 위험 및 외부 데이터 승인 지연 위험이 **완전히 해소됨**

#### 데이터 분할 전략

```
Wageningen 단일 데이터셋:
  Train (시간 순):              처음 70% (약 116일)
  Validation:                   중간 15% (약 25일)
  Test:                         마지막 15% (약 25일)
  Cross-compartment evaluation: 6개 compartment 중 5개 학습 + 1개 hold-out (6-fold)
  Cross-season evaluation:      train winter → test spring (단일 데이터셋 내)
  Rolling-origin evaluation:    미래 누설 방지
```

### 4.2 제안 프레임워크

**Control-Aware Spatio-Temporal Forecasting**:

```
입력:
- 환경 센서 시계열         X_env  (T × N_env)
- 외부 기상 시계열         X_ext  (T × N_ext)
- 액추에이터 state 시계열   X_act  (T × N_act)       — 0~100% 연속값
- Setpoint(requested) 시계열 X_sp   (T × N_sp)        — piecewise-constant, change-event flag 채널 동반
- Realized(VIP) 시계열      X_vip  (T × N_vip)       — dense (99.85% obs)
- 변수 간 방향성 그래프      G      (N × N, directed)
   · Prior graph (도메인 지식)
   · Learned graph (data-driven)

처리:
1. Temporal Encoder
   - 입력: X_env, X_ext, X_act, X_sp, X_vip 결합
   - 시간 의존성 잠재 표현 학습
   - 후보: Mamba (제안), LSTM, Transformer

2. Spatial Encoder (Directed Graph Module)
   - 입력: Temporal 출력 + 그래프 G
   - 변수 간 방향성 의존성 모델링
   - 출력: 변수 관계 반영 표현

3. Multi-Horizon Decoder
   - 입력: 통합 표현
   - 출력: 미래 H개 horizon 예측

출력:
- 1시간, 6시간, 24시간 후 내부 환경 변수 (Tair, Rhair, HumDef/VPD, CO2air)
```

**Directed Dependency Graph 정의 (Prior + Learned 비교)**:

도메인 지식 기반 prior graph 예시 (Wageningen 변수 기준):
```
Iglob (외부 일사)     → Tair, Tot_PAR
Tout (외기온)        → Tair
VentLee, Ventwind    → Tair, Rhair, CO2air, HumDef
BlackScr, EnScr      → Tot_PAR, Tair
co2_dos              → CO2air
PipeLow, PipeGrow    → Tair
water_sup, Cum_irr   → Rhair, HumDef, EC_drain_PC
co2_sp → co2_vip → co2_dos → CO2air   (setpoint → realized → actuator → 환경 인과 체인)
t_heat_sp → t_heat_vip → PipeLow/PipeGrow → Tair
```

**용어**: "Causal graph"가 아닌 "domain-informed directed dependency graph"로 표현. 인과 주장은 회피하되, control signal → physical effect의 도메인 지식은 명시적으로 활용.

### 4.3 Baseline 모델 (1개월 timeline 고려 축소)

**필수 baseline**:

| 분류 | 모델 | 비고 |
|---|---|---|
| 전통 baseline | Persistence (직전 값 유지) | sanity check |
| Tabular ML | LightGBM (lag features 포함) | strong tabular baseline |
| Recurrent | LSTM | 선행 연구 표준 |
| 최신 시계열 | PatchTST 또는 iTransformer 중 1개 | SOTA Transformer-based |
| 그래프 | STGNN only | spatial baseline |
| 시계열 (제안) | Mamba only | temporal proposal |
| **제안 모델** | **Control-Aware Mamba-STGNN** | full method |

**시간 허용 시 추가**:
- Transformer (vanilla)
- TCN
- Informer / Autoformer

**제외**:
- ARIMA (다변량+제어변수 구현 대비 기여 낮음)
- GRU (LSTM과 유사)

### 4.4 Ablation Study

| Ablation | 변경 |
|---|---|
| **Control source ablation** | (a) no control / (b) state only / (c) state+setpoint / (d) state+setpoint+VIP |
| **Graph ablation** | No-graph / Static graph / Domain-informed prior / Learned directed graph |
| **Temporal encoder ablation** | LSTM / Transformer / PatchTST / Mamba 교체 |
| **Horizon별 분석** | 1h / 6h / 24h 각각의 contribution |
| **Generalization ablation** | Time split / Cross-compartment 6-fold / Cross-season (winter→spring) |

### 4.5 평가 지표

**기본 오차**:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- **NMAE, NRMSE** (변수별 scale 정규화)
- **MASE** (Mean Absolute Scaled Error, naive baseline 대비)

**Multi-output 종합**:
- 변수별 metric 계산 → macro-average
- R² (변수별)

**Multi-horizon 분석**:
- 1h / 6h / 24h 각각 metric
- Horizon별 성능 degradation curve

**Event-based 분석 (본 논문 핵심)**:

Wageningen 데이터는 actuator state(0~100% 연속값)와 setpoint 두 가지 event source를 제공하므로, 다음 두 종류의 event를 정의한다.

**Event 정의 (1) — Actuator state transition**:
| Actuator event | Window | 분석 metric |
|---|---|---|
| 환기 (VentLee, Ventwind) opening transition (≥10%p 증가) | 0~3시간 | event-window MAE, peak error, lag error |
| 차광막/에너지 스크린 (BlackScr, EnScr) state change ≥20%p | 0~6시간 | event-window MAE |
| 관수 ON (water_sup 증가) | 0~6시간 | event-window MAE |
| CO₂ 공급 ON (co2_dos > threshold) | 0~2시간 | event-window MAE |

**Event 정의 (2) — Setpoint change (본 데이터셋 차별점)**:
| Setpoint event | Window | 분석 metric |
|---|---|---|
| `co2_sp` 변경 | 0~2시간 | response delay, peak timing error |
| `t_heat_sp` 변경 | 0~6시간 | 추적 정확도 |
| `t_vent_sp` 변경 | 0~3시간 | 추적 정확도 |
| `scr_blck_sp` / `scr_enrg_sp` 변경 | 0~6시간 | 추적 정확도 |
| `water_sup_intervals_sp_min` 변경 | 0~6시간 | 누적 관수 응답 |

**핵심 insight**: Setpoint signal 자체는 dense (95%+ obs)하지만 값이 거의 일정하게 유지되는 **piecewise-constant** 신호이므로, 값의 변경 시점(change event)을 명시적으로 검출하여 자연스러운 event timestamp로 활용한다. Wageningen은 분 단위 정확도(5분)로 event 분석 가능. EDA에서 Reference 기준 `t_heat_sp`는 6,709회, `t_vent_sp`는 7,220회 변경이 검출되어 통계적으로 충분한 event sample이 확보됨 (`02_event_response_quant.ipynb` 참조).

**Lag-aware metric (지연 효과 정량)**:
- Peak timing error (예측 peak 시점과 실제 peak 시점 차이, 분 단위)
- Response delay error (event 발생 시점부터 X% 응답까지 걸린 시간 차이)
- Cross-correlation lag

**MAPE 사용 제한**: 0 근처 값(VPD, 일사량 야간 등)에서 왜곡 위험 — 내부 온도 등 일부 변수에만 한정 사용.

**Generalization 분석**:
- **Cross-compartment** (6 compartment 중 1개 hold-out, 6-fold)
- **Cross-season** (train winter → test spring)
- 시간대별 (낮/밤) 성능
- Train-Test gap 측정

**해석 분석**:
- Prior graph vs Learned graph 비교
- Learned graph 시각화 (어떤 edge가 강화되었나?)
- Feature/Variable importance (특히 setpoint vs realized vs state 중 어느 것이 가장 영향력 있는지)
- AI 운영 compartment (5팀) vs Reference (인간 운영) 비교 — 부가 분석

### 4.6 EDA 검증 결과 (2026-05-19, 본 연구 viability 판정)

본 연구의 핵심 가설(actuator event 이후 delayed response가 데이터에서 검출 가능하고, 24h 일주기 구조가 입력 lookback 설계의 근거가 된다)을 3개의 EDA notebook으로 검증함.

#### 4.6.1 데이터 viability ([notebooks/eda/01_full_eda.ipynb](notebooks/eda/01_full_eda.ipynb))

- 6 compartment 모두 47,809 rows × 50 cols 동일 스키마, 동일 timestamp
- 5분 grid 안정 (Δt mean 5.000분, std 0.006분, 중복·결측 0)
- target (Tair, Rhair, CO2air) 결측률 0.15%
- 24h lookback + 24h horizon stride=1 sliding window 시 compartment당 47,234개 sample 가능
- target NaN 없는 clean window (stride=1h) 96.6% — cross-compartment 6-fold 학습에 충분

#### 4.6.2 Event-based delayed response 정량 검증 ([notebooks/eda/02_event_response_quant.ipynb](notebooks/eda/02_event_response_quant.ipynb))

Event를 **방향별로 분리** (`_up` / `_down`) 후 ISOLATED-only (±30분 내 다른 event 없음) response curve로 정량.

| Event | Target | Δ@30min | Δ@1h | Δ@3h | Δ@6h | peak Δ | peak time |
|---|---|---:|---:|---:|---:|---:|---:|
| `co2_dos_on` | CO2air | +48.0 | +39.4 | +28.3 | -4.3 | **+76.5 ppm** | **15 min** ⚡ |
| `co2_sp_up` | CO2air | +45.8 | +46.0 | +9.0 | -13.5 | **+62.5** | 40 min |
| `co2_sp_down` | CO2air | -86.6 | -133.2 | -169.6 | -155.9 | **-180.7** | 220 min |
| `VentLee_up` | CO2air | -19.6 | -25.3 | -16.5 | -3.0 | **-32.7** | 95 min |
| `VentLee_down` | CO2air | +9.8 | +14.6 | +31.5 | +78.0 | **+78.0** | 360 min |
| `t_heat_sp_up` | Tair | +0.85 | +1.19 | +2.92 | +4.16 | **+4.18 °C** | **355 min** 🔥 |
| `t_heat_sp_down` | Tair | -0.81 | -1.69 | -3.19 | -2.01 | **-3.28** | 230 min |

**해석**:
- 즉각 반응(co2_dos_on, peak 15분)과 지연 반응(t_heat_sp_up, peak 355분)이 **수십 분 ~ 6시간**의 다양한 시간 스케일로 공존 — multi-horizon (1h/6h/24h) 평가의 필요성 정당화
- 18개 directional pair 중 11개에서 up/down peak 부호 반전(sign flip) 확인 — event 방향 분리가 의미 있음
- → **계획서의 "actuator event 이후 delayed response 분석" 핵심 기여가 데이터에서 정량적으로 입증됨**

#### 4.6.3 Temporal structure (lookback 24h 근거) ([notebooks/eda/03_temporal_structure.ipynb](notebooks/eda/03_temporal_structure.ipynb))

6 compartment 평균 기준 4개 메트릭:

| Target | within-day ratio | ACF@12h | **ACF@24h** | ACF@48h | STL seasonal | FFT pow 24h/12h |
|---|---:|---:|---:|---:|---:|---:|
| **Tair** | **0.76** | -0.27 | **0.92** | 0.88 | **0.92** | **7.7×** |
| Rhair | 0.51 | 0.03 | **0.78** | 0.72 | 0.75 | **13.2×** |
| CO2air | 0.53 | -0.03 | **0.78** | 0.72 | 0.79 | **6.1×** |

**해석**:
- 전체 변동의 51~76%가 hour-of-day 평균 패턴으로 설명됨 (Tair에서 가장 강함)
- ACF@12h 음수 + ACF@24h 강한 양수 = **교과서적인 일주기 구조** (12시간 후 낮↔밤 반전, 24시간 후 강한 복원)
- STL seasonal strength 0.75~0.92 → 시계열의 대부분이 24h 주기로 분해됨
- FFT periodogram에서 24h peak가 12h peak보다 6~13배 강함 → 24h 주기가 dominant

→ **입력 lookback의 기본 후보를 24시간(288 step × 5분)으로 채택**. 최종 lookback 길이는 6h/12h/24h/48h sensitivity 실험(Week 3 후반 또는 Week 4)으로 결정한다. **Output horizon은 event-window delayed response 평가를 위해 24h로 고정**한다.

#### 4.6.4 EDA 결과에 따른 설계 보강

- Setpoint signal 표현을 "sparse event log"에서 **"piecewise-constant control target + change-event"** 로 일관 정정
- Event 정의에 방향 분리(`_up` / `_down`) 추가 — Section 4.5 평가 지표 적용
- 24h horizon 고정 + lookback sensitivity 명시
- 이상치 처리 정책은 별도 모듈(`src/preprocessing.py`)로 분리 예정 (Week 2)

### 4.7 통계적 유의성 (시계열 자기상관 고려)

- **Seeds**: 3 random seeds (시간 허용 시 5)
- **검정**:
  - **Diebold-Mariano test** (시계열 forecast 비교 표준)
  - **Block bootstrap / Moving block bootstrap** (자기상관 고려)
  - **Rolling-origin evaluation** (미래 누설 방지)
- **신뢰구간**: 95% CI (block bootstrap)
- **Effect size**: 평균 loss differential 보고
- **Horizon별, event-window별, compartment별 loss differential** 비교

---

## 5. 연구 일정 (1개월, 데이터 확보 완료 반영)

### Week 1: 환경 구축 및 Wageningen EDA

```
□ 지도교수님 1차 보고 (계획서 공유)  ← 최우선
☑ Wageningen 데이터 EDA (Section 4.6 결과 반영)
   ☑ 데이터 viability (01_full_eda.ipynb)
   ☑ Event-based delayed response 정량 (02_event_response_quant.ipynb)
   ☑ Temporal structure / lookback 24h 근거 (03_temporal_structure.ipynb)
□ Mamba 환경 세팅 시도 (mamba-ssm, CUDA) — 1~2일
   · 실패 시 Plan B: Transformer/PatchTST로 대체
□ PyTorch Geometric (STGNN) 세팅
□ 시계열 forecasting pipeline 코드 (LSTM baseline)
```

### Week 2: 전처리, 평가 코드, Baseline 학습

```
□ Wageningen 데이터 전처리
   · Setpoint change-event flag 추가 (signal은 이미 dense하지만 change 시점이 핵심) + 잔여 NaN ffill
   · 결측 처리, 정규화, 시간 정렬
□ Train/Val/Test split (시간 순 + cross-compartment 6-fold)
□ Rolling-origin evaluation setup
□ 평가 코드 검증:
   · NMAE, NRMSE, MASE
   · Event detection (actuator state transition + setpoint change)
   · Lag error metric, peak timing error 구현
   · Block bootstrap, Diebold-Mariano 검정 코드
□ Persistence baseline
□ LightGBM baseline (lag features)
□ LSTM baseline 학습
□ PatchTST 또는 iTransformer baseline 학습
```

### Week 3: 제안 모델 및 Ablation

```
□ STGNN only 학습 (prior + learned graph)
□ Mamba only 학습
□ Control-Aware Mamba-STGNN 학습 (제안)
□ Ablation:
  · Control source (no / state / state+sp / state+sp+VIP)
  · Graph 변형 (no / static / prior / learned)
  · Temporal encoder 변형 (LSTM / Transformer / Mamba)
□ Multi-horizon 결과 정리 (1h, 6h, 24h)
□ Event-window 분석 시작 (state transition + setpoint change 양쪽)
```

### Week 4: 분석, 통계 검정, 작성

```
□ Cross-compartment 6-fold 평가
□ Cross-season 평가 (winter → spring)
□ Event-based 분석 (actuator + setpoint 양쪽)
□ Lag error / peak timing error 계산
□ Learned graph 시각화
□ AI 운영 vs Reference 부가 분석
□ 통계 검정 (Diebold-Mariano, block bootstrap)
□ LaTeX 작성 (Introduction, Methods, Results, Discussion)
□ Figures, Tables 정리
□ 지도교수님 review
□ 제출
```

---

## 6. 예상 기여

### 학술적 기여

1. **Control-aware multi-horizon greenhouse forecasting 문제 정의**: 센서, 외부 기상, actuator state, setpoint, realized(VIP) 등 다층적 제어 정보를 결합한 통합 forecasting 프레임워크.

2. **Spatio-temporal 결합 architecture**: Temporal encoder(Mamba 등)와 directed graph module을 결합하여 장기 의존성과 변수 간 방향성 관계를 동시에 모델링.

3. **체계적 ablation 분석**: 제어 source(state/setpoint/VIP), graph 구조, temporal encoder, horizon별 각 구성요소의 기여 정량 검증.

4. **Event-based evaluation framework**: 전체 평균 성능뿐 아니라 (a) actuator state transition과 (b) setpoint change 두 종류의 event 직후 예측 오차와 응답 지연(lag error, peak timing error)을 명시적으로 분석.

5. **Setpoint vs Realized 분리 활용의 가치 정량화**: 두 신호를 모두 활용한 모델이 actuator state 단독 대비 응답 추적에서 갖는 우위 분석 (Wageningen VIP 변수 활용).

6. **Cross-compartment / cross-season generalization 분석**: 6 compartment 6-fold 평가와 winter→spring season 분할을 통해 동일 데이터셋 내 견고성 정량 제시.

### 실용적 기여

1. 스마트팜 운영자를 위한 multi-horizon 의사결정 지원 가능성 제시
2. Predictive control 또는 digital twin의 입력으로 활용 가능성 분석 (실제 closed-loop 제어는 본 연구 범위 외)
3. 학습된 directed dependency graph 시각화를 통한 인사이트
4. 고기술 온실(Wageningen)에서 검증된 forecasting 모델

---

## 7. 제한 사항 및 향후 연구

### 본 연구 제한 사항

- 단일 작물 (cherry tomato)
- 단일 데이터셋 (Wageningen 166일, 6 compartment 동일 시설 내 실험 데이터) → 진정한 cross-facility 평가에는 한계
- Uncertainty quantification 본 연구 범위 외
- 실제 closed-loop control 검증 없음 (예측까지만)
- 단일 시설 유형(네덜란드 고기술 온실)에 한정 — 한국 시설농 등 다른 운영 환경으로의 transfer는 향후 과제

### 향후 연구

1. 다른 작물 (파프리카, 오이 등) 적용 및 cross-crop 일반화
2. 한국 시설농(AI Hub) 또는 혁신밸리 데이터를 활용한 cross-dataset transfer 연구
3. Probabilistic forecasting (uncertainty quantification)
4. Closed-loop control simulation
5. Counterfactual reasoning ("환기 안 했다면?")
6. Vision-augmented multi-modal forecasting
7. LLM/RAG 기반 운영자 설명 인터페이스

---

## 8. 목표 저널

JCR/Scopus 최신 지표 기준 재확인 필요.

| 저널 | 적합도 | 비고 |
|---|---|---|
| Smart Agricultural Technology (Elsevier) | ★★★★★ | 주제 정확 일치, 1차 후보 |
| Computers and Electronics in Agriculture (Elsevier) | ★★★★ | 제어/예측 통합 기여 |
| Biosystems Engineering (Elsevier) | ★★★★ | 제어/시스템 관점 강조 시 |
| Sensors (MDPI) | ★★★★ | Wageningen 데이터 활용 선행 논문 다수 게재 |
| Agriculture / Agronomy (MDPI) | ★★★★ | 응용 논문으로 가능 |

**1차 목표**: Smart Agricultural Technology
**2차 목표** (1차 reject 시): MDPI Sensors 또는 Agronomy

---

## 9. 위험 요소 및 대응 (대폭 축소)

| 위험 | 심각도 | 대응 |
|---|---|---|
| ~~Actuator 로그 quality 낮음~~ (원안 최대 위험) | **낮음 (해결)** | Wageningen 데이터에서 5분 자동 로깅 + setpoint/VIP 분리 검증 완료 |
| ~~외부 데이터 승인 지연~~ | **제거됨** | 단일 데이터셋(Wageningen)으로 전환, 외부 신청 불필요 |
| Mamba 환경 세팅 실패 | 중 | Transformer 또는 PatchTST로 대체 |
| Setpoint event 정의 모호 | 낮음 | Setpoint signal은 piecewise-constant (95%+ obs)이며 EDA에서 변경 시점 검출이 명확히 가능함을 확인 (`02_event_response_quant.ipynb`) |
| Multi-horizon 학습 시간 폭발 | 중상 | Baseline 축소 (3-4개), horizon 축소 가능 |
| 결과가 기존 baseline과 큰 차이 없음 | 중상 | Event-based + setpoint 분석에 차별점 집중 |
| Cross-compartment 결과 분산 큼 | 중 | 6-fold 평균 + 분산 보고. 분산 자체도 분석 결과로 의미 있음 |
| 시간 부족 | 중 | Q2 저널로 목표 조정 |

---

## 10. 결론

본 연구는 actuator 제어 로그(state, setpoint, realized/VIP)와 환경 변수 간 방향성 의존성을 동시에 반영하는 control-aware multi-horizon forecasting 프레임워크를 제안하고, **네덜란드 Wageningen Autonomous Greenhouse Challenge 2nd Edition 데이터셋(166일, 6 compartment)**에서 검증한다.

단순 모델 조합이 아니라 다음을 핵심 기여로 한다:

1. Actuator event(state transition + setpoint change) 이후의 delayed response를 event-window 기반으로 명시적으로 분석하는 평가 방법론
2. Lag error, peak timing error 등 지연 효과 정량 metric
3. Setpoint vs Realized 신호 분리 활용의 가치 정량화
4. Domain-informed prior graph와 learned graph 비교를 통한 변수 관계 분석
5. Multi-horizon (1h, 6h, 24h) 통합 평가
6. Cross-compartment 6-fold 및 cross-season generalization을 통한 동일 데이터셋 내 견고성 검증

체계적 ablation, 시계열 특성을 고려한 통계 검정(Diebold-Mariano, block bootstrap)을 통해 스마트팜 미기상 예측 분야에 incremental하면서도 정량적이고 실용적인 기여를 제공한다.

---

## 부록 A: 즉시 진행 체크리스트

```
□ 지도교수님 1차 보고 (계획서 공유)
☑ Wageningen 데이터 EDA (notebooks/eda/01~03 완료)
□ Mamba 환경 세팅 시도 (1-2일)
□ PyTorch Geometric 세팅
□ LSTM baseline 코드 작성
□ NMAE, NRMSE, MASE 평가 코드
□ Event detection 코드 (state transition `_up`/`_down` + setpoint change 양쪽)
□ Lag error metric, peak timing error 코드
□ Block bootstrap, Diebold-Mariano 검정 코드
□ Outlier 처리 모듈 (src/preprocessing.py) — Reference 음수 8행 등
□ Lookback sensitivity 실험 코드 (6h/12h/24h/48h)
□ LaTeX template 준비
```

## 부록 C: EDA notebook 산출물 (2026-05-19 완료)

| Notebook | 핵심 산출 | 본문 인용 위치 |
|---|---|---|
| [`notebooks/eda/01_full_eda.ipynb`](notebooks/eda/01_full_eda.ipynb) | 스키마·timestamp·결측·sample 가능성 7단계 | Section 4.6.1, 부록 B |
| [`notebooks/eda/02_event_response_quant.ipynb`](notebooks/eda/02_event_response_quant.ipynb) | 13 event(방향 분리) × 3 target response curve + 정량표 + sign symmetry | Section 4.5 Event 정의, Section 4.6.2 |
| [`notebooks/eda/03_temporal_structure.ipynb`](notebooks/eda/03_temporal_structure.ipynb) | hour-of-day · ACF · STL · FFT (lookback 24h 근거) | Section 4.6.3 |

각 notebook은 `agc` conda 환경에서 재현 가능 (`environment.yml`). 데이터는 `data/raw/<compartment>/`에 압축 해제하여 사용.

## 부록 B: Wageningen 데이터 실측 검증 결과 (완료)

```
✅ Timestamp 해상도: 5분 (mean Δt 5.000분, std 0.006분, 중복 0, 결측 0)
✅ 기간: 2019-12-16 ~ 2020-05-30 (166일)
✅ 행 수: 47,809 rows/compartment × 6 compartment (스키마·timestamp 완전 동일)
✅ Actuator state: 5개, 0~100% 연속값
✅ Setpoint(sp): 14개, **piecewise-constant (95~99% 관측)** — 변경 시점이 event signal
✅ Realized(VIP): 14개, sp와 1:1 매칭, **99.85% obs (dense)**
✅ 결측률: sensor/state/vip = 0.15%, sp = 1~5% (변경 직전 누락)
✅ 외부 기상 동시 제공 (Weather.csv, 5분, 11 cols)
✅ 근권 센서 동시 제공 (GrodanSens.csv, 5분, 7 cols)
✅ ReadMe.pdf 공식 변수 명세 확보
✅ Event-based delayed response 정량 검증 완료 (02 notebook)
✅ Lookback 24h EDA 근거 확보 (03 notebook: ACF@24h 0.78~0.92, STL seasonal 0.75~0.92, FFT 24h peak 6~13×)
```

---

*최종본 작성일: 2026-05-19*
*다음 단계: 지도교수님 보고 + Wageningen EDA 시작 + 환경 세팅*
