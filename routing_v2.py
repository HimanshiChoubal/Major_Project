"""routing_v2.py — Uncertainty- and constraint-aware QAOA crew dispatch.

Objective per subset S of sites:  cost = -sum_i w_i*clip(mean_i + kappa*std_i, 0, 1) + mu*T(S) + lambda*max(0, T(S)-budget)^2
  T(S) = optimal depot tour length / speed + service time * |S|
kappa > 0 sends crews to sites the model is *unsure* about (optimism under uncertainty).
"""
from __future__ import annotations

import numpy as np
import pennylane as qml
import torch

import config as cfg
from quantum_models import _best_tour


def solve_risk_aware_dispatch(risk_mean, risk_std=None, kappa: float = 1.0, severity=None,
                              time_budget_h: float = 4.0, speed_kmh: float = 30.0, service_h: float = 1.0,
                              travel_cost: float = 0.05, penalty: float = 10.0, p_layers: int = cfg.QAOA.p_layers,
                              n_steps: int = 100, lr: float = 0.1, seed: int = cfg.SEED) -> dict:
    n = cfg.QAOA.n_nodes
    mean = np.clip(np.asarray(risk_mean, dtype=float).reshape(-1), 0, 1)
    std = np.zeros(n) if risk_std is None else np.asarray(risk_std, dtype=float).reshape(-1)
    w = np.ones(n) if severity is None else np.asarray(severity, dtype=float).reshape(-1)
    assert len(mean) == n == len(std) == len(w), f"Need {n} sites"
    eff = np.clip(mean + kappa * std, 0, 1) * w
    D = np.array(cfg.QAOA.distance_matrix)
    depot = cfg.QAOA.depot_index

    reward, time_h, cost = np.zeros(2 ** n), np.zeros(2 ** n), np.zeros(2 ** n)
    tours = []
    for idx in range(2 ** n):
        sel = [i for i, b in enumerate(format(idx, f"0{n}b")) if b == "1"]
        km, order = _best_tour(sel, D, depot)
        t = km / speed_kmh + service_h * len(sel)
        reward[idx], time_h[idx] = eff[sel].sum(), t
        cost[idx] = -reward[idx] + travel_cost * t + penalty * max(0.0, t - time_budget_h) ** 2
        tours.append((sel, km, order))

    c_norm = (cost - cost.min()) / (cost.max() - cost.min() + 1e-8)
    c_t = torch.tensor(c_norm, dtype=torch.float64)
    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circ(gam, bet):
        for q in range(n):
            qml.Hadamard(wires=q)
        for l in range(p_layers):
            qml.DiagonalQubitUnitary(torch.exp(-1j * gam[l] * c_t * 2 * np.pi), wires=list(range(n)))
            for q in range(n):
                qml.RX(2 * bet[l], wires=q)
        return qml.probs(wires=list(range(n)))

    g = torch.Generator().manual_seed(seed)
    gam = torch.rand(p_layers, generator=g, dtype=torch.float64).requires_grad_(True)
    bet = (torch.rand(p_layers, generator=g, dtype=torch.float64) * np.pi / 2).requires_grad_(True)
    opt = torch.optim.Adam([gam, bet], lr=lr)
    hist = []
    for _ in range(n_steps):
        opt.zero_grad()
        loss = (circ(gam, bet) * c_t).sum()
        loss.backward()
        opt.step()
        hist.append(float(loss))
    with torch.no_grad():
        probs = circ(gam, bet).numpy()

    top = np.argsort(probs)[::-1][:4]
    q_idx = int(top[np.argmin(cost[top])])
    opt_idx = int(np.argmin(cost))

    # greedy baseline: add sites by effective risk while the time budget holds
    gsel = 0
    for i in np.argsort(-eff):
        cand = gsel | (1 << (n - 1 - i))
        if time_h[cand] <= time_budget_h:
            gsel = cand

    names = cfg.QAOA.node_names

    def summarize(idx):
        sel, km, order = tours[idx]
        route = [depot] + order + [depot] if order else [depot]
        return {"sites": [names[i] for i in sel], "route": [names[i] for i in route], "route_indices": route,
                "route_km": float(km), "time_h": float(time_h[idx]), "reward": float(reward[idx]),
                "true_risk_covered": float(mean[sel].sum()), "cost": float(cost[idx])}

    return {"qaoa": summarize(q_idx), "exact": summarize(opt_idx), "greedy": summarize(gsel),
            "qaoa_matches_exact": q_idx == opt_idx, "qaoa_state_prob": float(probs[q_idx]),
            "reward_ratio_vs_exact": float(reward[q_idx] / max(reward[opt_idx], 1e-9)),
            "effective_risk": eff.tolist(), "kappa": kappa, "cost_history": hist}


if __name__ == "__main__":
    r = solve_risk_aware_dispatch([0.8, 0.3, 0.55, 0.4, 0.6], [0.02, 0.25, 0.05, 0.3, 0.05], kappa=1.0)
    print(r["qaoa"], r["exact"]["sites"], r["qaoa_matches_exact"])
