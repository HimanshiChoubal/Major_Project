"""uncertainty.py — MC-dropout uncertainty, temperature scaling, calibration metrics, selective prediction, RUL with uncertainty."""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Predictive uncertainty
# --------------------------------------------------------------------------- #
def _enable_dropout(model: nn.Module) -> None:
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()


def _bin_entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


@torch.no_grad()
def mc_dropout_predict(model: nn.Module, feats: torch.Tensor, n_passes: int = 20, bs: int = 128) -> Dict[str, np.ndarray]:
    """Returns mean prob, std, predictive entropy (total), mutual information (epistemic), aleatoric (expected entropy)."""
    _enable_dropout(model)
    P = np.zeros((n_passes, len(feats)), dtype=np.float64)
    for t in range(n_passes):
        P[t] = torch.cat([torch.sigmoid(model(feats[s:s + bs])) for s in range(0, len(feats), bs)]).numpy()
    model.eval()
    mean = P.mean(0)
    total, aleatoric = _bin_entropy(mean), _bin_entropy(P).mean(0)
    return {"mean": mean, "std": P.std(0), "entropy": total, "aleatoric": aleatoric,
            "mutual_info": np.maximum(total - aleatoric, 0.0)}


@torch.no_grad()
def predict_logits(model: nn.Module, feats: torch.Tensor, bs: int = 128) -> torch.Tensor:
    model.eval()
    return torch.cat([model(feats[s:s + bs]) for s in range(0, len(feats), bs)])


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def fit_temperature(logits: torch.Tensor, y: torch.Tensor) -> float:
    """Single-scalar temperature scaling on validation logits (minimizes BCE)."""
    logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=100)
    y = y.float()

    def closure():
        opt.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(logits / logT.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(logT.exp())


def ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 10):
    """Expected calibration error (confidence of predicted class). Returns (ece, bin_conf, bin_acc, bin_count)."""
    pred = (probs >= 0.5).astype(int)
    conf = np.where(pred == 1, probs, 1 - probs)
    correct = (pred == y).astype(float)
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    e, bc, ba, bn = 0.0, [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0.5 else (conf >= lo) & (conf <= hi)
        if m.sum() == 0:
            bc.append(np.nan); ba.append(np.nan); bn.append(0)
            continue
        bc.append(conf[m].mean()); ba.append(correct[m].mean()); bn.append(int(m.sum()))
        e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e), np.array(bc), np.array(ba), np.array(bn)


def brier(probs: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((probs - y) ** 2))


def risk_coverage(probs: np.ndarray, y: np.ndarray, uncertainty: np.ndarray):
    """Selective prediction: keep the most-certain fraction. Returns (coverage, selective_error, AURC)."""
    order = np.argsort(uncertainty)
    err = ((probs >= 0.5).astype(int) != y).astype(float)[order]
    n = len(err)
    cov = np.arange(1, n + 1) / n
    sel_err = np.cumsum(err) / np.arange(1, n + 1)
    return cov, sel_err, float(sel_err.mean())


# --------------------------------------------------------------------------- #
# Remaining useful life with uncertainty
# --------------------------------------------------------------------------- #
def rul_from_score(p_mean: float, p_std: float = 0.0, limit: float = 40.0, n: int = 4000, seed: int = 0) -> dict:
    """HEURISTIC single-inspection RUL: decay rate k = 0.02 + 0.12*p (per month); p sampled from N(p_mean, p_std)."""
    rng = np.random.default_rng(seed)
    p = np.clip(rng.normal(p_mean, p_std, n), 0.0, 1.0)
    hi = 100.0 * (1 - p)
    k = 0.02 + 0.12 * p
    rul = np.where(hi > limit, np.log(np.maximum(hi, 1e-6) / limit) / k, 0.0)
    return {"median": float(np.median(rul)), "p05": float(np.percentile(rul, 5)), "p95": float(np.percentile(rul, 95)),
            "samples": rul}


def fit_degradation(times, health, limit: float = 40.0, n_boot: int = 1000, seed: int = 0) -> dict:
    """Learned RUL from repeated inspections: log-linear fit h(t)=h0*exp(-k t) + residual bootstrap -> RUL distribution (months).
    `times` in months, `health` = Health Index (0-100) at each inspection."""
    t = np.asarray(times, dtype=float)
    h = np.clip(np.asarray(health, dtype=float), 1e-3, None)
    X = np.c_[np.ones_like(t), t]
    y = np.log(h)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    rng = np.random.default_rng(seed)
    ruls = np.empty(n_boot)
    for i in range(n_boot):
        yb = X @ beta + rng.choice(res, len(res), replace=True)
        b, *_ = np.linalg.lstsq(X, yb, rcond=None)
        k, h_now = -b[1], np.exp(b[0] + b[1] * t[-1])
        ruls[i] = 0.0 if h_now <= limit else (np.inf if k <= 1e-9 else np.log(h_now / limit) / k)
    fin = ruls[np.isfinite(ruls)]
    return {"decay_rate": float(-beta[1]), "median": float(np.median(ruls)),
            "p05": float(np.percentile(ruls, 5)), "p95": float(np.percentile(ruls, 95)),
            "p_no_decline": float(1 - len(fin) / n_boot), "samples": ruls}


def simulate_inspection_history(k_true: float = 0.08, h0: float = 95.0, n: int = 8, noise: float = 0.05, seed: int = 0):
    """Demo-only synthetic monthly inspection history (no real multi-date data in SDNET2018)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    return t, h0 * np.exp(-k_true * t) * np.exp(rng.normal(0, noise, n))
