# 연구계획서 요약본 (지도교수님 보고용)

## Control-Aware Spatio-Temporal Forecasting for Multi-Horizon Greenhouse Microclimate Prediction

---

## 1. 한 줄 요약

> Wageningen Autonomous Greenhouse Challenge 데이터를 활용해, actuator 제어 로그(state + setpoint + realized VIP)를 분해해 입력 효과를 분석하고, 방향성 graph fusion과 horizon-aware decoder를 포함한 **control-aware multi-horizon forecasting** 프레임워크를 평가하며, **event-based evaluation**으로 actuator event 직후의 지연 반응(delayed response)을 정량 분석한다.

---

## 2. 연구 배경 및 문제

**기존 한계**
1. Actuator event 이후의 **delayed response를 multi-horizon 관점에서 구조적으로 분석한 연구 부재**
2. 평가가 전체 평균 오차(MAE/RMSE) 중심 — event 구간 품질 및 응답 지연 측정 불가
3. 대부분 단기(5–15분) 예측에 집중, 농가 의사결정에 필요한 중장기(1–24h) 지원 부족

**핵심 문제 정의**

Setpoint(requested) → Realized(VIP) → Actuator state → 환경 변수로 이어지는 제어 인과 체인을 모델 입력으로 통합하고, actuator event(state transition + setpoint change) 이후의 lag error / peak timing error를 event-window 기반으로 명시적으로 분석한다.

---

## 3. 데이터 확보 결과 (핵심 변경 사항) ✅

본 연구는 **Wageningen Autonomous Greenhouse Challenge — 2nd Edition (2019)** 데이터셋을 단독 사용한다. 원안의 한국 데이터셋(AI Hub 534) 의존을 제거하고, **품질이 검증된 단일 외국 고기술 온실 데이터에 집중**한다.

### Wageningen 데이터 실측 검증 결과

| 항목 | 내용 |
|---|---|
| 출처 | Wageningen University & Research, Bleiswijk, Netherlands |
| 라이센스 | 4TU.ResearchData (open, academic use) |
| 작물 | Cherry tomato (cv. Axiany) |
| 기간 | 2019-12-16 ~ 2020-05-30 (**166일, ~5.5개월**) |
| Compartment | **6개** (5 AI 팀 + 1 Reference 인간 운영자) |
| 시간 해상도 | **5분 자동 로깅** (median delta 4분 59.808초, 실측) |
| 행 수 | 47,809 rows × 6 compartment |
| 결측률 | sensor/state/VIP: **0.15%**, setpoint: **95~99% 관측** (piecewise-constant signal, 변경 시점이 event) |

### 핵심 변수 구조

- **센서 16개**: Tair, Rhair, CO2air, HumDef(VPD 대용), PipeLow/Grow, Tot_PAR, EC/pH_drain, Cum_irr, water_sup, co2_dos
- **Actuator state 5개** (0–100% 연속값): VentLee, Ventwind, AssimLight, EnScr, BlackScr
- **Setpoint 14개** (sp suffix): co2_sp, t_heat_sp, t_vent_sp, scr_*_sp 등 — **piecewise-constant control target** (95% 이상 관측되는 dense signal, 변경 시점을 event signal로 활용)
- **Realized VIP 14개** (vip suffix): 모든 setpoint과 **1:1 매칭** — 실제 적용값 (99.85% obs)
- **외부 기상** (Weather.csv, 5분, 11 cols)
- **근권 센서** (GrodanSens.csv, 5분, 7 cols)
- 일별 자원 소비 (Resources), 수확/품질 (Production, TomQuality 등)

### EDA 검증 결과 (2026-05-19 완료)

3개 notebook으로 viability·event·temporal 구조 검증.

**(1) 데이터 viability** ([notebooks/eda/01_full_eda.ipynb](notebooks/eda/01_full_eda.ipynb))
- 6 compartment 모두 47,809행 / 50컬럼 / 동일 schema, 동일 timestamp (5분 grid, std 0.006분)
- target 결측률 0.15%, 24h/24h sliding window stride=1로 47,234개 가능, clean window 96.6%

**(2) Event-based delayed response 정량 검증** ([notebooks/eda/02_event_response_quant.ipynb](notebooks/eda/02_event_response_quant.ipynb))
- event를 **방향별로 분리** (`_up` / `_down`) 후 ISOLATED-only response curve로 정량
- 핵심 결과 (Δ from t=0, peak time):
  - `co2_dos_on` → CO2air **+76.5 ppm @ 15분** (즉시 반응)
  - `co2_sp_up` → CO2air **+62.5 @ 40분**, `co2_sp_down` → **-180.7 @ 220분**
  - `t_heat_sp_up` → Tair **+4.18 °C @ 355분** (long thermal inertia 확인)
  - `VentLee_up` → CO2air **-32.7 ppm**, `VentLee_down` → **+78.0** (부호 반대)
- 18 directional pair 중 **11개에서 up/down sign flip** ✓ (대부분 confounding으로 설명 가능)

→ **계획서의 "actuator event 이후 delayed response" 가설이 데이터에서 정량적으로 입증됨**

**(3) Temporal structure (lookback 24h EDA 근거)** ([notebooks/eda/03_temporal_structure.ipynb](notebooks/eda/03_temporal_structure.ipynb))
- 6 compartment 평균:

| Target | within-day ratio | ACF@12h | **ACF@24h** | STL seasonal | FFT 24h/12h |
|---|---:|---:|---:|---:|---:|
| Tair | 0.76 | -0.27 | **0.92** | 0.92 | **7.7×** |
| Rhair | 0.51 | 0.03 | **0.78** | 0.75 | 13.2× |
| CO2air | 0.53 | -0.03 | **0.78** | 0.79 | 6.1× |

→ ACF@12h 음수 + ACF@24h 강한 양수 = 일주기 구조 명확. **입력 lookback의 기본 후보를 24h로 채택**, 최종은 6h/12h/24h/48h sensitivity로 결정.

### 단일 데이터셋 사용의 정당성

- **5분 자동 로깅 + setpoint/VIP 1:1 분리**는 event-based 분석에 필수적인 시간 해상도 보장
- 6 compartment를 활용한 **cross-compartment generalization** 평가 가능 (5개 학습 + 1개 hold-out)
- Cross-season 분할(winter → spring)도 단일 데이터셋 내에서 수행 가능
- 원안의 actuator 로그 품질 위험이 **완전히 해소됨**

---

## 4. 제안 방법

### 4.1 Control-Aware Spatio-Temporal Architecture

```
입력:
  X_env (센서)  + X_ext (외부기상) + X_act (state) + X_sp (setpoint) + X_vip (realized)
  G (변수 간 directed dependency graph: prior + learned 비교)

처리:
  1) Temporal Encoder      (Mamba 제안; LSTM/Transformer/PatchTST 비교)
  2) Optional Graph Fusion (prior / learned / prior-initialized learned 비교)
  3) Multi-Horizon Decoder (flat MLP vs horizon-aware target-specific decoder)

출력:
  미래 Tair, Rhair, HumDef(VPD), CO2air
```

### 4.2 Directed Dependency Graph (인과 주장 회피, 도메인 지식 활용)

```
co2_sp → co2_vip → co2_dos → CO2air
t_heat_sp → t_heat_vip → PipeLow/Grow → Tair
VentLee/Ventwind → Tair, Rhair, HumDef
BlackScr/EnScr → Tot_PAR, Tair
Iglob (외부 일사) → Tair, Tot_PAR
```

### 4.3 Baseline (timeline 고려 축소)

Persistence · LightGBM(lag) · LSTM · PatchTST(또는 iTransformer) · STGNN only · Mamba only · **제안 모델 (Control-Aware Mamba-STGNN)**

### 4.4 Ablation

| Ablation | 변형 |
|---|---|
| Control source | no / state / state+sp / **state+sp+VIP** |
| Graph | no / prior / learned / prior-initialized learned |
| Temporal encoder | LSTM / Transformer / PatchTST / **Mamba** |
| Decoder | flat MLP / horizon-aware target-specific |
| Horizon | 1h / 6h / 24h 각 contribution |

---

## 5. 평가 방법

**기본 오차**: MAE, RMSE, **NMAE/NRMSE** (변수별 정규화), **MASE** (naive 대비)

**Event-based 분석 (본 논문 핵심)**

- (1) **Actuator state transition** event: 환기 ≥10%p, 스크린 ≥20%p, 관수 ON, CO₂ ON
- (2) **Setpoint change** event (Wageningen 강점): co2_sp / t_heat_sp / t_vent_sp / scr_*_sp 변경 시점

**Lag-aware metric**: peak timing error, response delay error, cross-correlation lag

**Generalization**: Cross-compartment (6 compartment 중 1 hold-out), Cross-season (winter → spring), Rolling-origin

**통계 검정**: Diebold-Mariano, Block bootstrap (자기상관 고려), 3 seeds (시간 허용 시 5)

---

## 6. 연구 일정 (1개월)

| Week | 주요 활동 |
|---|---|
| **W1** | 지도교수님 보고 · Wageningen EDA · Mamba/PyG 환경 세팅 · LSTM baseline 코드 |
| **W2** | 전처리 (setpoint change-event flag + ffill로 dense화) · 평가 코드 (event detection, lag metric, DM test) · Persistence/LightGBM/LSTM/PatchTST 학습 |
| **W3** | STGNN only · Mamba only · **제안 모델** 학습 · Ablation (control/graph/encoder) · multi-horizon 결과 |
| **W4** | Cross-compartment 평가 · Event-window 분석 · Learned graph 시각화 · 통계 검정 · LaTeX 작성 · 제출 |

---

## 7. 예상 기여

1. **Control-aware multi-horizon greenhouse forecasting 문제 정의** — sensor + actuator state + setpoint + realized(VIP) 통합
2. **조건부 spatio-temporal 분석** — Mamba/LSTM/Transformer와 directed graph fusion이 target·event·split별로 보이는 강점과 한계 정량화
3. **체계적 ablation** — 제어 source / graph 구조 / temporal encoder / decoder / horizon별 기여 정량
4. **Event-based evaluation framework** — actuator state transition + setpoint change 양쪽 event에서 lag error, peak timing error 정량
5. **Setpoint vs Realized 분리 활용의 가치 정량화** — Wageningen VIP 변수 활용의 차별점
6. **Cross-compartment / cross-season generalization** — 동일 데이터셋 내 견고성 검증

---

## 8. 위험 요소 및 대응

| 위험 | 심각도 | 대응 |
|---|---|---|
| ~~Actuator 로그 품질~~ | **해결** | Wageningen 5분 자동 로깅 + sp/VIP 분리 검증 완료 |
| Mamba 환경 세팅 실패 | 중 | PatchTST/Transformer로 대체 |
| Multi-horizon 학습 시간 폭발 | 중상 | Baseline 3–4개로 축소, horizon 축소 가능 |
| 결과가 baseline과 큰 차이 없음 | 중상 | Event-based + setpoint 분석에 차별점 집중 |
| 시간 부족 | 중 | Q2 저널로 목표 조정 |

---

## 9. 목표 저널

- **1차**: Smart Agricultural Technology (Elsevier) — 주제 정확 일치
- **2차**: MDPI Sensors (Wageningen 활용 선행 논문 다수 게재) 또는 Agronomy

---

## 10. 다음 단계

- [ ] 지도교수님 보고 및 피드백 수렴
- [x] Wageningen 데이터 EDA (viability + event response + temporal structure 3 notebook 완료)
- [ ] Mamba (mamba-ssm + CUDA) 환경 세팅 시도
- [ ] PyTorch Geometric (STGNN) 세팅
- [ ] LSTM baseline + 평가 코드 (NMAE, MASE, event detection, DM test) 작성
- [ ] Lookback sensitivity 실험 (6h/12h/24h/48h, baseline 1~2개로)
