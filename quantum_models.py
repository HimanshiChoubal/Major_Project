"""quantum_models.py — AQFS-QCNN core, hybrid PyTorch network, and QAOA maintenance-crew dispatch."""
from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Sequence

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn

import config as cfg

try:  # Qiskit is used only for hardware-portability diagnostics
    from qiskit import QuantumCircuit as _QiskitCircuit
except ImportError:  # pragma: no cover
    _QiskitCircuit = None

N_Q = cfg.QUANTUM.n_qubits  # 4


# --------------------------------------------------------------------------- #
# Circuit building blocks
# --------------------------------------------------------------------------- #
def _feature_map(x) -> None:
    """Dense angle encoding: RY/RZ per qubit + ZZ-style data-dependent entanglement ring."""
    for i in range(N_Q):
        qml.RY(x[..., i], wires=i)
        qml.RZ(x[..., i], wires=i)
    for i in range(N_Q):
        j = (i + 1) % N_Q
        qml.CNOT(wires=[i, j])
        qml.RZ(x[..., i] * x[..., j], wires=j)
        qml.CNOT(wires=[i, j])


def _su4_conv(p, a: int, b: int) -> None:
    """General SU(4) two-qubit unitary: 3 CNOTs + single-qubit rotations (15 parameters)."""
    qml.Rot(p[0], p[1], p[2], wires=a)
    qml.Rot(p[3], p[4], p[5], wires=b)
    qml.CNOT(wires=[a, b])
    qml.RY(p[6], wires=a)
    qml.RZ(p[7], wires=b)
    qml.CNOT(wires=[b, a])
    qml.RY(p[8], wires=a)
    qml.CNOT(wires=[a, b])
    qml.Rot(p[9], p[10], p[11], wires=a)
    qml.Rot(p[12], p[13], p[14], wires=b)


def _conv_layer(p, pairs) -> None:
    """Translation-invariant conv: one shared SU(4) applied to every pair."""
    for a, b in pairs:
        _su4_conv(p, a, b)


def _pool_layer(p, pairs) -> None:
    """Controlled-rotation pooling: source qubit conditions rotations on the kept qubit."""
    for src, dst in pairs:
        qml.CRZ(p[0], wires=[src, dst])
        qml.PauliX(wires=src)
        qml.CRX(p[1], wires=[src, dst])
        qml.PauliX(wires=src)


def _aqfs_reupload_ring(x, scale, bias, ring) -> None:
    """AQFS: learnable data re-uploading + data-conditioned entanglement ring across all qubits."""
    for i in range(N_Q):
        qml.RY(scale[i] * x[..., i] + bias[i], wires=i)
    for i in range(N_Q):
        j = (i + 1) % N_Q
        qml.CNOT(wires=[i, j])
        qml.RZ(ring[i] * x[..., j], wires=j)


def _readout():
    return [
        qml.expval(qml.PauliZ(1)),
        qml.expval(qml.PauliZ(3)),
        qml.expval(qml.PauliX(1)),
        qml.expval(qml.PauliX(3)),
    ]


# --------------------------------------------------------------------------- #
# 1. Quantum core
# --------------------------------------------------------------------------- #
class AQFSQCNNCore:
    """4-qubit QCNN with optional AQFS re-uploading ring. Exposes `.qnode` and `.weight_shapes`."""

    N_OUT = 4

    def __init__(self, use_aqfs: bool = cfg.QUANTUM.use_aqfs):
        assert cfg.QUANTUM.n_qubits == 4, "Core is hard-wired for 4 qubits"
        self.use_aqfs = use_aqfs
        self.dev = qml.device(cfg.QUANTUM.device, wires=N_Q)

        shapes: Dict[str, tuple] = {"conv1": (15,), "conv2": (15,), "pool": (2,), "conv3": (15,)}

        if use_aqfs:
            shapes.update({"aqfs_scale": (N_Q,), "aqfs_bias": (N_Q,), "aqfs_ring": (N_Q,)})

            def circuit(inputs, conv1, conv2, pool, conv3, aqfs_scale, aqfs_bias, aqfs_ring):
                _feature_map(inputs)
                _conv_layer(conv1, [(0, 1), (2, 3)])
                _conv_layer(conv2, [(1, 2), (3, 0)])
                _aqfs_reupload_ring(inputs, aqfs_scale, aqfs_bias, aqfs_ring)
                _pool_layer(pool, [(0, 1), (2, 3)])
                _su4_conv(conv3, 1, 3)
                return _readout()
        else:
            def circuit(inputs, conv1, conv2, pool, conv3):
                _feature_map(inputs)
                _conv_layer(conv1, [(0, 1), (2, 3)])
                _conv_layer(conv2, [(1, 2), (3, 0)])
                _pool_layer(pool, [(0, 1), (2, 3)])
                _su4_conv(conv3, 1, 3)
                return _readout()

        self.weight_shapes = shapes
        self.qnode = qml.QNode(circuit, self.dev, interface="torch", diff_method=cfg.QUANTUM.diff_method)

    # -- diagnostics ------------------------------------------------------- #
    def _dummy_kwargs(self) -> dict:
        kw = {"inputs": torch.rand(N_Q) * np.pi}
        kw.update({k: torch.rand(s) for k, s in self.weight_shapes.items()})
        return kw

    def draw(self) -> str:
        return qml.draw(self.qnode)(**self._dummy_kwargs())

    def n_params(self) -> int:
        return int(sum(int(np.prod(s)) for s in self.weight_shapes.values()))

    def qiskit_report(self) -> Optional[dict]:
        """Export to OpenQASM -> Qiskit and report depth / gate counts (NISQ portability check)."""
        if _QiskitCircuit is None:
            return None
        try:
            qasm = qml.to_openqasm(self.qnode)(**self._dummy_kwargs())
            qc = _QiskitCircuit.from_qasm_str(qasm)
            return {"depth": qc.depth(), "size": qc.size(), "ops": dict(qc.count_ops())}
        except Exception as exc:  # unsupported gate in exporter, version mismatch, etc.
            return {"error": str(exc)}


# --------------------------------------------------------------------------- #
# 2. Hybrid network
# --------------------------------------------------------------------------- #
class HybridQCNNNet(nn.Module):
    """Windows [B,T,4] -> TorchLayer QCNN per window -> softmax window attention -> FC head -> logit."""

    def __init__(self, use_aqfs: bool = cfg.QUANTUM.use_aqfs, hidden: int = cfg.QUANTUM.head_hidden):
        super().__init__()
        self.use_aqfs = use_aqfs
        self.core = AQFSQCNNCore(use_aqfs=use_aqfs)
        init = {k: (lambda w: nn.init.normal_(w, std=0.5)) for k in self.core.weight_shapes}
        self.qlayer = qml.qnn.TorchLayer(self.core.qnode, self.core.weight_shapes, init_method=init)

        self.attn = nn.Linear(AQFSQCNNCore.N_OUT, 1) if use_aqfs else None
        self.head = nn.Sequential(
            nn.Linear(AQFSQCNNCore.N_OUT, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.last_attention: Optional[torch.Tensor] = None

    def quantum_features(self, x: torch.Tensor) -> torch.Tensor:
        B, T, F = x.shape
        flat = x.reshape(B * T, F)
        out = self.qlayer(flat)
        if out.dim() == 1:  # guard against flattened multi-measurement stacking
            out = out.reshape(AQFSQCNNCore.N_OUT, B * T).T
        return out.float().reshape(B, T, AQFSQCNNCore.N_OUT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.quantum_features(x)                                   # [B,T,4]
        if self.use_aqfs:
            a = torch.softmax(self.attn(q).squeeze(-1), dim=1)         # [B,T] softmax over windows
            self.last_attention = a.detach()
            pooled = (a.unsqueeze(-1) * q).sum(dim=1)
        else:
            pooled = q.mean(dim=1)
        return self.head(pooled).squeeze(-1)                           # logits [B]

    @torch.no_grad()
    def scores(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        was_training = self.training
        self.eval()
        p = torch.sigmoid(self(x))
        self.train(was_training)
        return {"anomaly": p, "health_index": cfg.APP.health_index_scale * (1.0 - p)}

    def window_saliency(self, x: torch.Tensor) -> torch.Tensor:
        """Gradient saliency per window -> [B, gh, gw] (grid from config)."""
        self.eval()
        xi = x.clone().requires_grad_(True)
        self(xi).sum().backward()
        g = cfg.IMAGE.windows_per_axis
        return xi.grad.abs().sum(-1).reshape(-1, g, g)


# --------------------------------------------------------------------------- #
# 3. QAOA maintenance routing
# --------------------------------------------------------------------------- #
def _to_numpy(a) -> np.ndarray:
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    return np.asarray(a, dtype=np.float64).reshape(-1)


def _best_tour(nodes: Sequence[int], D: np.ndarray, depot: int):
    """Exact shortest depot -> nodes -> depot tour (<=5 nodes, brute force)."""
    nodes = [n for n in nodes if n != depot]
    if not nodes:
        return 0.0, []
    best_len, best_order = np.inf, None
    for perm in itertools.permutations(nodes):
        path = (depot, *perm, depot)
        length = sum(D[path[i], path[i + 1]] for i in range(len(path) - 1))
        if length < best_len:
            best_len, best_order = length, list(perm)
    return float(best_len), best_order


def solve_qaoa_maintenance_route(
    risk_scores,
    max_sites: int = 3,
    p_layers: int = cfg.QAOA.p_layers,
    n_steps: int = cfg.QAOA.n_steps,
    step_size: float = cfg.QAOA.step_size,
    risk_weight: float = 1.0,
    travel_weight: float = 0.15,
    seed: int = cfg.SEED,
) -> dict:
    """QAOA (5 qubits, 1 per site) selects the dispatch set maximizing covered risk minus travel,
    subject to a crew capacity `max_sites`; an exact tour then orders the selected stops."""
    n = cfg.QAOA.n_nodes
    risks = np.clip(_to_numpy(risk_scores), 0.0, 1.0)
    if risks.shape[0] != n:
        raise ValueError(f"Expected {n} risk scores, got {risks.shape[0]}")
    D = np.array(cfg.QAOA.distance_matrix, dtype=np.float64)
    depot = cfg.QAOA.depot_index

    # Classical cost table over all 2^n selections (wire 0 = most significant bit)
    costs = np.zeros(2 ** n)
    tours: List[tuple] = []
    for idx in range(2 ** n):
        sel = [i for i, b in enumerate(format(idx, f"0{n}b")) if b == "1"]
        length, order = _best_tour(sel, D, depot)
        overflow = max(0, len(sel) - max_sites)
        costs[idx] = (
            -risk_weight * risks[sel].sum()
            + travel_weight * length
            + cfg.QAOA.penalty * overflow ** 2
        )
        tours.append((sel, length, order))
    c_norm = (costs - costs.min()) / (costs.max() - costs.min() + 1e-8)
    c_t = torch.tensor(c_norm, dtype=torch.float64)

    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def qaoa_circuit(gammas, betas):
        for w in range(n):
            qml.Hadamard(wires=w)
        for layer in range(p_layers):
            phase = torch.exp(-1j * gammas[layer] * c_t * 2 * np.pi)
            qml.DiagonalQubitUnitary(phase, wires=list(range(n)))
            for w in range(n):
                qml.RX(2 * betas[layer], wires=w)
        return qml.probs(wires=list(range(n)))

    g = torch.Generator().manual_seed(seed)
    gammas = (torch.rand(p_layers, generator=g, dtype=torch.float64)).requires_grad_(True)
    betas = (torch.rand(p_layers, generator=g, dtype=torch.float64) * np.pi / 2).requires_grad_(True)
    opt = torch.optim.Adam([gammas, betas], lr=step_size)

    history: List[float] = []
    for _ in range(n_steps):
        opt.zero_grad()
        loss = (qaoa_circuit(gammas, betas) * c_t).sum()
        loss.backward()
        opt.step()
        history.append(float(loss))

    with torch.no_grad():
        probs = qaoa_circuit(gammas, betas).numpy()

    top = np.argsort(probs)[::-1][:4]
    best_idx = int(top[np.argmin(costs[top])])          # lowest-cost among most probable states
    opt_idx = int(np.argmin(costs))                     # classical optimum for verification
    sel, length, order = tours[best_idx]

    names = cfg.QAOA.node_names
    route_names = [names[depot]] + [names[i] for i in order] + [names[depot]] if order else [names[depot]]
    return {
        "selected_sites": [names[i] for i in sel],
        "selected_indices": sel,
        "route": route_names,
        "route_indices": [depot] + order + [depot] if order else [depot],
        "route_length_km": length,
        "covered_risk": float(risks[sel].sum()),
        "qaoa_state": format(best_idx, f"0{n}b"),
        "qaoa_state_probability": float(probs[best_idx]),
        "classical_optimum_state": format(opt_idx, f"0{n}b"),
        "matches_classical_optimum": best_idx == opt_idx,
        "cost_history": history,
        "gammas": gammas.detach().numpy().tolist(),
        "betas": betas.detach().numpy().tolist(),
    }


if __name__ == "__main__":
    core = AQFSQCNNCore(use_aqfs=True)
    print(core.draw())
    print("Core params:", core.n_params(), "| Qiskit:", core.qiskit_report())
    net = HybridQCNNNet(use_aqfs=True)
    xb = torch.rand(2, cfg.IMAGE.n_windows, N_Q) * np.pi
    print("logits:", net(xb).shape, "| attention:", net.last_attention.shape)
    print(solve_qaoa_maintenance_route([0.9, 0.2, 0.7, 0.4, 0.8]))