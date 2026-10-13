"""config.py — Central configuration for the Quantum-Enhanced Infrastructure Monitoring Streamlit app."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# --------------------------------------------------------------------------- #
# Global
# --------------------------------------------------------------------------- #
SEED: int = 42


# --------------------------------------------------------------------------- #
# Imagery / preprocessing
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ImageConfig:
    patch_size: int = 32                       # 32x32 pixel satellite patch
    bands: Tuple[str, ...] = ("R", "G", "B", "NIR")
    window_size: int = 8                       # 8x8 sliding window (non-overlapping)
    window_stride: int = 8

    @property
    def n_bands(self) -> int:
        return len(self.bands)                 # 4

    @property
    def windows_per_axis(self) -> int:
        return self.patch_size // self.window_stride   # 4

    @property
    def n_windows(self) -> int:
        return self.windows_per_axis ** 2      # 16

    @property
    def window_feature_dim(self) -> int:
        return self.window_size * self.window_size * self.n_bands   # 256

    @property
    def patch_shape(self) -> Tuple[int, int, int]:
        return (self.n_bands, self.patch_size, self.patch_size)     # (C, H, W)


@dataclass(frozen=True)
class PCAConfig:
    n_components: int = 4                      # one component per qubit
    whiten: bool = False
    angle_range: Tuple[float, float] = (0.0, math.pi)   # rescaled for RY/RZ encoding


# --------------------------------------------------------------------------- #
# Quantum model
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class QuantumConfig:
    n_qubits: int = 4                          # == PCAConfig.n_components
    device: str = "default.qubit"
    diff_method: str = "backprop"
    use_aqfs: bool = True                      # Adaptive Quantum Feature Saliency block
    head_hidden: int = 16


# --------------------------------------------------------------------------- #
# Classes
# --------------------------------------------------------------------------- #
ASSET_CLASSES: Dict[int, str] = {
    0: "Healthy",
    1: "Fissured",
}
CLASS_COLORS: Dict[int, str] = {0: "#2e7d32", 1: "#c62828"}


# --------------------------------------------------------------------------- #
# QAOA maintenance routing (5 asset sites)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class QAOAConfig:
    # (name, (x_km, y_km))
    sites: Tuple[Tuple[str, Tuple[float, float]], ...] = (
        ("Bridge-A", (0.0, 0.0)),
        ("Dam-B", (4.0, 1.5)),
        ("Highway-C", (2.5, 5.0)),
        ("Viaduct-D", (7.0, 4.0)),
        ("Tunnel-E", (5.5, 8.0)),
    )
    depot_index: int = 0                       # route starts/ends here
    p_layers: int = 2                          # QAOA depth
    n_steps: int = 80
    step_size: float = 0.1
    shots: int | None = None                   # None -> analytic
    penalty: float = 10.0                      # TSP constraint penalty weight

    @property
    def n_nodes(self) -> int:
        return len(self.sites)                 # 5

    @property
    def node_names(self) -> List[str]:
        return [s[0] for s in self.sites]

    @property
    def coords(self) -> List[Tuple[float, float]]:
        return [s[1] for s in self.sites]

    @property
    def distance_matrix(self) -> List[List[float]]:
        c = self.coords
        return [[math.dist(a, b) for b in c] for a in c]


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TrainConfig:
    n_samples: int = 1000
    train_frac: float = 0.70
    val_frac: float = 0.15
    batch_size: int = 32
    epochs: int = 10
    lr: float = 0.02
    noise_sigma: float = 0.08
    cloud_prob: float = 0.4


# --------------------------------------------------------------------------- #
# Streamlit UI
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AppConfig:
    page_title: str = "Quantum Infrastructure Monitor"
    page_icon: str = "🛰️"
    layout: str = "wide"
    anomaly_threshold: float = 0.5
    health_index_scale: float = 100.0


# --------------------------------------------------------------------------- #
# Singletons
# --------------------------------------------------------------------------- #
IMAGE = ImageConfig()
PCA = PCAConfig()
QUANTUM = QuantumConfig()
QAOA = QAOAConfig()
TRAIN = TrainConfig()
APP = AppConfig()

# Consistency checks
assert IMAGE.n_windows == 16, "Expected 16 windows (4x4) per 32x32 patch"
assert IMAGE.n_bands == 4, "Expected RGB+NIR = 4 bands"
assert PCA.n_components == QUANTUM.n_qubits == 4, "PCA components must equal qubit count"
assert QAOA.n_nodes == 5, "Expected 5 maintenance asset sites"