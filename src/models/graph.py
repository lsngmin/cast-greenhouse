"""Directed Graph Module — 논문 8-layer 구조의 Layer 4.

────────────────────────────────────────────────────────────────────────────
역할 (한 줄)
    변수 간 directed dependency를 명시적으로 모델링.
    X: (B, L, F) → graph_context: (B, D)

────────────────────────────────────────────────────────────────────────────
왜 필요한가?

D 단계 (event-window evaluation) 결과:
  VentLee_up × CO2air 같은 환기-CO2 event에서 모든 vanilla backbone
  (LSTM/Transformer/Mamba)이 평균 persistence보다 나쁨 (rel_MAE ≥ 1.0).

해석: implicit attention/recurrence만으로는 actuator → target의 인과 chain
을 학습 불충분. 명시적 directed edge (예: VentLee → CO2air) 가 필요.

본 모듈이 그 명시적 edge를 도메인 지식 (Prior) 또는 data-driven (Learned)
방식으로 제공.

────────────────────────────────────────────────────────────────────────────
설계 결정

- **Variable-level graph (각 input feature = node):**
    F=8~74 (feature_group 따라). 도메인 edge가 명확 (예: co2_sp → co2_vip).
    feature_group ablation 시에도 자연스러움 (없는 변수의 edge는 자동 제외).

- **Single-layer message passing:**
    1 hop. 깊은 GNN은 baseline overkill. Prior chain (`co2_sp → co2_vip
    → co2_dos → CO2air`) 같은 multi-hop은 변수 순서대로 직접 edge로 표현하면
    1-hop으로 충분.

- **Time pool 먼저 (variable-wise 평균):**
    Per-timestep graph는 (B, L, F, D) 텐서 → 메모리 폭증. baseline에서는
    time average 한 번 후 graph 적용 → (B, F, D). event-window 변동은
    backbone에서 잡고, graph는 정적 인과 구조만.

- **Prior + Learned 두 가지 모드:**
    `mode='prior'`: hard-coded 0/1 edges (도메인 지식)
    `mode='learned'`: nn.Parameter로 edge weight 학습
    `mode='prior_learned'`: prior로 init한 후 학습 가능 (hybrid)
    Ablation으로 셋 다 비교 (H3).

- **Output: (B, D) graph context:**
    Fusion layer가 temporal pool 출력 (B, D)과 결합하기 좋은 형태.

────────────────────────────────────────────────────────────────────────────
입출력 contract

    Input:  X  (B, L, F)
    Output: g  (B, D)  — graph context

────────────────────────────────────────────────────────────────────────────
논문 작성 참고

    "The directed graph module operates at the variable level: each of the
     F input features is a node, and directed edges encode the assumed
     control→target causal pathway (e.g., VentLee → CO2air, t_heat_sp →
     t_heat_vip → PipeLow → Tair). The module first averages each variable
     over the lookback window to obtain a single scalar per variable,
     projects it to d_model, then applies one round of directed message
     passing along the adjacency matrix, and reads out the result by mean
     pooling over variables. Three adjacency variants are compared:
     domain-informed Prior, fully Learned (random init), and Prior-Learned
     hybrid (Prior init, gradient updates allowed)."
"""
from __future__ import annotations
from typing import Sequence

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Prior edges — 도메인 지식 + EDA(02) 결과 기반
# (src → dst): src 변수의 값/event가 dst 변수에 영향을 준다고 가정

PRIOR_EDGES: list[tuple[str, str]] = [
    # ─── External weather → indoor environment ────────────────────────
    ('Iglob', 'Tair'),
    ('Iglob', 'Tot_PAR'),
    ('Tout', 'Tair'),
    ('Rhout', 'Rhair'),
    ('RadSum', 'Tot_PAR'),
    ('PARout', 'Tot_PAR'),

    # ─── Ventilation actuator → indoor climate ────────────────────────
    ('VentLee', 'Tair'),
    ('VentLee', 'Rhair'),
    ('VentLee', 'CO2air'),
    ('Ventwind', 'Tair'),
    ('Ventwind', 'Rhair'),
    ('Ventwind', 'CO2air'),

    # ─── Screens → light + temperature ────────────────────────────────
    ('BlackScr', 'Tot_PAR'),
    ('BlackScr', 'Tair'),
    ('EnScr', 'Tair'),

    # ─── Lights → PAR / Tair ─────────────────────────────────────────
    ('AssimLight', 'Tot_PAR_Lamps'),
    ('AssimLight', 'Tot_PAR'),
    ('Tot_PAR_Lamps', 'Tot_PAR'),
    ('Tot_PAR_Lamps', 'Tair'),

    # ─── CO2 control chain ───────────────────────────────────────────
    ('co2_sp', 'co2_vip'),
    ('co2_vip', 'co2_dos'),
    ('co2_dos', 'CO2air'),

    # ─── Heating control chain ───────────────────────────────────────
    ('t_heat_sp', 't_heat_vip'),
    ('t_heat_vip', 'PipeLow'),
    ('t_heat_vip', 'PipeGrow'),
    ('PipeLow', 'Tair'),
    ('PipeGrow', 'Tair'),

    # ─── Ventilation control chain ───────────────────────────────────
    ('t_vent_sp', 't_ventlee_vip'),
    ('t_vent_sp', 't_ventwind_vip'),
    ('t_ventlee_vip', 'VentLee'),
    ('t_ventwind_vip', 'Ventwind'),

    # ─── Screen control chain ────────────────────────────────────────
    ('scr_blck_sp', 'scr_blck_vip'),
    ('scr_enrg_sp', 'scr_enrg_vip'),
    ('scr_blck_vip', 'BlackScr'),
    ('scr_enrg_vip', 'EnScr'),

    # ─── Irrigation → humidity / drain ───────────────────────────────
    ('water_sup', 'Rhair'),
    ('water_sup', 'EC_drain_PC'),
    ('Cum_irr', 'EC_drain_PC'),

    # ─── Derived ─────────────────────────────────────────────────────
    ('Tair', 'VPD'),
    ('Rhair', 'VPD'),
    ('Tair', 'HumDef'),
    ('Rhair', 'HumDef'),

    # ─── Event flags → targets ───────────────────────────────────────
    # (event flag는 시점 표현이지만, 명시적 edge로 두면 event 직후 target
    #  변동에 더 민감해질 가능성)
    ('VentLee_up', 'Tair'),
    ('VentLee_up', 'Rhair'),
    ('VentLee_up', 'CO2air'),
    ('VentLee_down', 'Tair'),
    ('VentLee_down', 'Rhair'),
    ('VentLee_down', 'CO2air'),
    ('Ventwind_up', 'CO2air'),
    ('Ventwind_down', 'CO2air'),
    ('co2_dos_on', 'CO2air'),
    ('co2_sp_changed', 'CO2air'),
    ('t_heat_sp_changed', 'Tair'),
    ('t_vent_sp_changed', 'Tair'),
    ('t_vent_sp_changed', 'CO2air'),
    ('BlackScr_up', 'Tair'),
    ('BlackScr_down', 'Tair'),
    ('EnScr_up', 'Tair'),
    ('EnScr_down', 'Tair'),
]


def build_prior_adjacency(
    feature_cols: Sequence[str],
    edges: Sequence[tuple[str, str]] = PRIOR_EDGES,
) -> torch.Tensor:
    """feature_cols에 있는 변수만 사용해서 (F, F) adjacency 만들기.

    `adj[src, dst] = 1` if (src → dst) edge exists.

    Args:
        feature_cols: WindowDataset.feature_cols 형태 list.
        edges: (src_name, dst_name) tuple list. default PRIOR_EDGES.

    Returns:
        (F, F) float tensor with 0/1 entries.
    """
    F = len(feature_cols)
    col_to_idx = {c: i for i, c in enumerate(feature_cols)}
    adj = torch.zeros(F, F, dtype=torch.float32)
    n_added = 0
    for src, dst in edges:
        if src in col_to_idx and dst in col_to_idx:
            adj[col_to_idx[src], col_to_idx[dst]] = 1.0
            n_added += 1
    return adj


# ---------------------------------------------------------------------------
# Directed Graph Module

class DirectedGraphModule(nn.Module):
    """Variable-level directed graph with 1-hop message passing.

    Pipeline:
        X (B, L, F)
          → time avg                                    (B, F)
          → per-variable embed (Linear 1 → D)           (B, F, D)
          → directed message passing: h' = h + adj.T @ msg_proj(h)
          → LayerNorm                                   (B, F, D)
          → mean over F (readout)                       (B, D)
          → final Linear                                (B, D)

    Args:
        input_dim:     F. feature_group에 따라 8/18/38/58/74.
        d_model:       hidden dim D. default 128 (다른 layer와 일관).
        adjacency:     (F, F) float tensor. None이면 zero adjacency (== no graph,
                       sanity 비교용).
        mode:          'prior'         (frozen 0/1 adj),
                       'learned'       (nn.Parameter, random init),
                       'prior_learned' (nn.Parameter, prior init).

    Shape:
        in  : (B, L, F)
        out : (B, D)
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        adjacency: torch.Tensor | None = None,
        mode: str = 'prior',
    ):
        super().__init__()
        if mode not in ('prior', 'learned', 'prior_learned'):
            raise ValueError(f"Unknown mode={mode!r}")

        self.input_dim = input_dim
        self.d_model = d_model
        self.mode = mode

        # Adjacency
        if adjacency is None:
            adj = torch.zeros(input_dim, input_dim)
        else:
            if adjacency.shape != (input_dim, input_dim):
                raise ValueError(
                    f"adjacency shape {tuple(adjacency.shape)} != "
                    f"(input_dim={input_dim}, input_dim={input_dim})"
                )
            adj = adjacency.float()

        if mode == 'prior':
            self.register_buffer('adj', adj)
            self.adj_learnable = False
        elif mode == 'learned':
            # random init, 학습 가능
            self.adj = nn.Parameter(torch.randn(input_dim, input_dim) * 0.01)
            self.adj_learnable = True
        elif mode == 'prior_learned':
            # prior로 init, 학습 가능
            self.adj = nn.Parameter(adj.clone())
            self.adj_learnable = True

        # Per-variable embed: scalar value → d_model
        # 모든 variable이 같은 projection을 공유 (variable-specific projection은
        # F=74 × D parameter 폭발 위험)
        self.embed = nn.Linear(1, d_model, bias=False)

        # Message passing components
        self.msg_proj = nn.Linear(d_model, d_model, bias=True)
        self.norm = nn.LayerNorm(d_model)

        # Final readout
        self.readout = nn.Linear(d_model, d_model, bias=False)
        self.readout_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, F) — same input as backbone (raw scaled feature window).

        Returns:
            (B, D) graph context vector.
        """
        if x.ndim != 3:
            raise ValueError(
                f"DirectedGraphModule expected (B, L, F), got {tuple(x.shape)}."
            )
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"DirectedGraphModule expected input_dim={self.input_dim}, "
                f"got x.shape[-1]={x.shape[-1]}."
            )

        # 1. Time pool (variable-wise mean over L)
        x_avg = x.mean(dim=1)                     # (B, F)
        x_var = x_avg.unsqueeze(-1)               # (B, F, 1)

        # 2. Per-variable embed
        h = self.embed(x_var)                     # (B, F, D)

        # 3. Directed message passing (1 hop)
        msg = self.msg_proj(h)                    # (B, F, D)
        # h_new[v] = sum over u where adj[u, v] > 0 of msg[u]
        # einsum: 'uv, bud -> bvd'
        agg = torch.einsum('uv,bud->bvd', self.adj, msg)
        h_out = self.norm(h + agg)                # residual + LN

        # 4. Readout: mean over variables (F축 축약)
        z = h_out.mean(dim=1)                     # (B, D)
        return self.readout_norm(self.readout(z)) # (B, D)

    def extra_repr(self) -> str:
        n_params = sum(p.numel() for p in self.parameters())
        n_edges = int((self.adj != 0).sum().item()) if self.mode == 'prior' \
            else 'learnable'
        return (
            f"input_dim={self.input_dim}, d_model={self.d_model}, "
            f"mode={self.mode!r}, n_edges={n_edges}, params={n_params}"
        )
