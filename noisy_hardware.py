"""noisy_hardware.py — NISQ evaluation: layer-level depolarizing noise (density-matrix sim) + zero-noise extrapolation (ZNE).

Re-implements the trained circuit on `default.mixed`, inserting a depolarizing channel on every qubit after each
circuit stage, then compares: ideal, noisy, and ZNE-mitigated accuracy. Slow by design (density matrices): use few images.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pennylane as qml
import torch
from sklearn.metrics import roc_auc_score

import quantum_models as qm


def make_noisy_qnode(use_aqfs: bool, p: float, shots=None):
    dev = qml.device("default.mixed", wires=4, shots=shots)

    def noise():
        if p > 0:
            for w in range(4):
                qml.DepolarizingChannel(p, wires=w)

    def circuit(inputs, conv1, conv2, pool, conv3, aqfs_scale=None, aqfs_bias=None, aqfs_ring=None):
        qm._feature_map(inputs); noise()
        qm._conv_layer(conv1, [(0, 1), (2, 3)]); noise()
        qm._conv_layer(conv2, [(1, 2), (3, 0)]); noise()
        if use_aqfs:
            qm._aqfs_reupload_ring(inputs, aqfs_scale, aqfs_bias, aqfs_ring); noise()
        qm._pool_layer(pool, [(0, 1), (2, 3)]); noise()
        qm._su4_conv(conv3, 1, 3); noise()
        return qm._readout()

    return qml.QNode(circuit, dev, interface="torch", diff_method=None)


def _weights(model) -> Dict[str, torch.Tensor]:
    w = getattr(model.q, "qnode_weights", None)
    if w is None:
        w = dict(model.q.named_parameters())
    return {k: v.detach().double() for k, v in w.items()}


@torch.no_grad()
def noisy_quantum_features(model, angles: torch.Tensor, p: float, shots=None, chunk: int = 256) -> torch.Tensor:
    """angles [B,T,4] -> noisy expectation features [B,T,4]."""
    qnode = make_noisy_qnode(model.mode == "aqfs", p, shots)
    w = _weights(model)
    B, T, _ = angles.shape
    flat = angles.reshape(B * T, 4).double()
    out = []
    for s in range(0, len(flat), chunk):
        res = qnode(inputs=flat[s:s + chunk], **w)
        out.append(torch.stack([torch.as_tensor(r) for r in res], dim=-1))
    return torch.cat(out).float().reshape(B, T, 4)


def richardson(q1, q2, q3):
    """Quadratic Richardson extrapolation to zero noise from noise scales 1,2,3."""
    return 3 * q1 - 3 * q2 + q3


@torch.no_grad()
def evaluate_noise_sweep(model, feats, y, p_list: List[float] = (0.0, 0.002, 0.005, 0.01, 0.02),
                         n_images: int = 40, seed: int = 0) -> List[dict]:
    """Accuracy/AUC vs depolarizing probability: ideal-sim reference, noisy, and ZNE-mitigated."""
    assert model.mode in ("qcnn", "aqfs"), "Noise sweep applies to quantum models"
    model.eval()
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(feats), size=min(n_images, len(feats)), replace=False)
    f, yy = feats[idx], y[idx].numpy().astype(int)
    ang = model.encode(f)

    def metrics(q):
        pr = torch.sigmoid(model.classify(q)[0]).numpy()
        auc = roc_auc_score(yy, pr) if len(np.unique(yy)) > 1 else float("nan")
        return float(((pr >= 0.5).astype(int) == yy).mean()), float(auc)

    rows = []
    for p in p_list:
        q1 = noisy_quantum_features(model, ang, p)
        acc, auc = metrics(q1)
        row = {"p": p, "acc_noisy": acc, "auc_noisy": auc, "acc_zne": acc, "auc_zne": auc}
        if p > 0:
            q0 = richardson(q1, noisy_quantum_features(model, ang, 2 * p), noisy_quantum_features(model, ang, 3 * p))
            row["acc_zne"], row["auc_zne"] = metrics(q0)
        rows.append(row)
        print(f"  p={p:<6} noisy acc {row['acc_noisy']:.3f} | ZNE acc {row['acc_zne']:.3f}")
    return rows
