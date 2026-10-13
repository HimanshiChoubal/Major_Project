"""data_generator.py — Synthetic RGB+NIR infrastructure patches, windowing, and PCA preprocessing."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA as SkPCA

import config as cfg


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _smooth_field(rng: np.random.Generator, size: int, sigma: float) -> np.ndarray:
    """Zero-mean, unit-std low-frequency random field."""
    f = gaussian_filter(rng.standard_normal((size, size)), sigma)
    return (f - f.mean()) / (f.std() + 1e-8)


def _random_walk_crack(rng: np.random.Generator, size: int) -> np.ndarray:
    """Binary random-walk crack mask (size x size), guaranteed non-trivial length."""
    for _ in range(20):
        mask = np.zeros((size, size), dtype=np.float32)
        r, c = rng.uniform(4, size - 4, 2)
        ang = rng.uniform(0, 2 * np.pi)
        for _ in range(int(rng.integers(25, 50))):
            ang += rng.normal(0.0, 0.45)
            r += np.sin(ang)
            c += np.cos(ang)
            ri, ci = int(round(r)), int(round(c))
            if 0 <= ri < size and 0 <= ci < size:
                mask[ri, ci] = 1.0
        if mask.sum() >= 12:
            return mask
    return mask


# --------------------------------------------------------------------------- #
# 1. Patch generator
# --------------------------------------------------------------------------- #
def generate_synthetic_patch(
    label: int,
    rng: np.random.Generator,
    noise_sigma: float = cfg.TRAIN.noise_sigma,
    cloud_prob: float = cfg.TRAIN.cloud_prob,
    return_mask: bool = False,
):
    """Return a (4, S, S) float32 patch [R, G, B, NIR]; label 1 adds random-walk cracks."""
    S = cfg.IMAGE.patch_size
    yy, xx = np.mgrid[0:S, 0:S] / (S - 1)

    # Concrete deck: mid-grey base + low-frequency texture + expansion-joint bands
    lum = 0.55 + 0.06 * _smooth_field(rng, S, 4.0) + 0.02 * _smooth_field(rng, S, 1.0)
    for pos in rng.uniform(0.2, 0.8, int(rng.integers(1, 3))):
        lum -= 0.05 * np.exp(-((yy - pos) / 0.012) ** 2)

    img = np.stack([lum * 1.00, lum * 0.97, lum * 0.92, 0.30 + 0.45 * lum])  # NIR: concrete moderately reflective

    # Vegetation patches: low red, high green, very high NIR
    veg = np.zeros((S, S), dtype=np.float32)
    for _ in range(int(rng.integers(1, 4))):
        v = np.zeros((S, S), dtype=np.float32)
        v[rng.integers(2, S - 2), rng.integers(2, S - 2)] = 1.0
        v = gaussian_filter(v, rng.uniform(1.5, 3.0))
        veg = np.maximum(veg, v / (v.max() + 1e-8))
    veg *= rng.uniform(0.5, 0.9)
    img[0] += veg * (0.30 - img[0])
    img[1] += veg * (0.55 - img[1])
    img[2] += veg * (0.25 - img[2])
    img[3] += veg * (0.85 - img[3])

    # Distractor shadows (RGB only, no NIR signature) in both classes
    for _ in range(int(rng.integers(0, 3))):
        b = np.zeros((S, S), dtype=np.float32)
        b[rng.integers(3, S - 3), rng.integers(3, S - 3)] = 1.0
        b = gaussian_filter(b, rng.uniform(1.5, 2.5))
        img[:3] -= 0.08 * b / (b.max() + 1e-8)

    # Cracks: dark in RGB, NIR drop (moisture ingress)
    mask = np.zeros((S, S), dtype=np.float32)
    if label == 1:
        raw = _random_walk_crack(rng, S)
        soft = gaussian_filter(raw, 0.5)
        soft /= soft.max() + 1e-8
        img[:3] -= 0.18 * soft
        img[3] -= 0.12 * soft
        mask = raw

    # Clouds: bright, low-contrast semi-transparent overlay
    if rng.random() < cloud_prob:
        a = gaussian_filter(rng.random((S, S)), 4.0)
        a = (a - a.min()) / (a.max() - a.min() + 1e-8)
        alpha = rng.uniform(0.3, 0.7) * a
        cloud = np.array([0.92, 0.93, 0.95, 0.85], dtype=np.float32)[:, None, None]
        img = (1 - alpha) * img + alpha * cloud

    # Sensor noise
    img = img + rng.normal(0.0, noise_sigma, img.shape)
    img = img.astype(np.float32)
    return (img, mask) if return_mask else img


# --------------------------------------------------------------------------- #
# 2. Dataset
# --------------------------------------------------------------------------- #
def generate_dataset(
    n_samples: int = cfg.TRAIN.n_samples,
    seed: int = cfg.SEED,
    noise_sigma: float = cfg.TRAIN.noise_sigma,
    cloud_prob: float = cfg.TRAIN.cloud_prob,
    return_masks: bool = False,
):
    """Balanced dataset. Returns (X: [N,4,S,S] float32, y: [N] float32); optionally masks [N,S,S]."""
    rng = np.random.default_rng(seed)
    labels = np.arange(n_samples) % 2
    rng.shuffle(labels)

    X = np.empty((n_samples, cfg.IMAGE.n_bands, cfg.IMAGE.patch_size, cfg.IMAGE.patch_size), dtype=np.float32)
    M = np.zeros((n_samples, cfg.IMAGE.patch_size, cfg.IMAGE.patch_size), dtype=np.float32)
    for i, lab in enumerate(labels):
        X[i], M[i] = generate_synthetic_patch(int(lab), rng, noise_sigma, cloud_prob, return_mask=True)

    Xt = torch.from_numpy(X)
    yt = torch.from_numpy(labels.astype(np.float32))
    return (Xt, yt, torch.from_numpy(M)) if return_masks else (Xt, yt)


# --------------------------------------------------------------------------- #
# 3. Window extraction
# --------------------------------------------------------------------------- #
def extract_windows(X, win: int = cfg.IMAGE.window_size) -> np.ndarray:
    """[N,C,H,W] -> [N, n_windows, C*win*win] non-overlapping windows (row-major order)."""
    X = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else np.asarray(X)
    N, C, H, W = X.shape
    assert H % win == 0 and W % win == 0, "Patch size must be divisible by window size"
    w = X.reshape(N, C, H // win, win, W // win, win)
    w = w.transpose(0, 2, 4, 1, 3, 5)                       # N, gh, gw, C, win, win
    return w.reshape(N, (H // win) * (W // win), C * win * win).astype(np.float32)


# --------------------------------------------------------------------------- #
# 4. PCA preprocessor
# --------------------------------------------------------------------------- #
@dataclass
class PCAPreprocessor:
    win: int
    n_components: int
    mu: np.ndarray            # per-channel mean  [1,C,1,1]
    sd: np.ndarray            # per-channel std   [1,C,1,1]
    pca: SkPCA
    lo: np.ndarray            # per-component 1st percentile
    hi: np.ndarray            # per-component 99th percentile
    angle_range: Tuple[float, float] = cfg.PCA.angle_range

    def _normalize(self, X) -> np.ndarray:
        X = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else np.asarray(X)
        return ((X - self.mu) / self.sd).astype(np.float32)

    def normalize_images(self, X) -> torch.Tensor:
        """Z-score images (for the classical CNN baseline)."""
        return torch.from_numpy(self._normalize(X))

    def transform(self, X) -> torch.Tensor:
        """[N,C,H,W] -> [N, n_windows, n_components] angles in [0, pi]."""
        wins = extract_windows(self._normalize(X), self.win)
        N, T, D = wins.shape
        Z = self.pca.transform(wins.reshape(-1, D))
        a, b = self.angle_range
        Z = np.clip((Z - self.lo) / (self.hi - self.lo + 1e-8), 0.0, 1.0) * (b - a) + a
        return torch.from_numpy(Z.reshape(N, T, self.n_components).astype(np.float32))


def fit_pca_preprocessor(
    X_train,
    win: int = cfg.IMAGE.window_size,
    n_components: int = cfg.PCA.n_components,
) -> PCAPreprocessor:
    """Fit z-score + PCA on TRAIN windows only; scale components to [0, pi]."""
    X = X_train.detach().cpu().numpy() if isinstance(X_train, torch.Tensor) else np.asarray(X_train)
    mu = X.mean(axis=(0, 2, 3), keepdims=True)
    sd = X.std(axis=(0, 2, 3), keepdims=True) + 1e-8
    Xn = ((X - mu) / sd).astype(np.float32)

    wins = extract_windows(Xn, win)
    flat = wins.reshape(-1, wins.shape[-1])
    pca = SkPCA(n_components=n_components, random_state=cfg.SEED).fit(flat)

    Z = pca.transform(flat)
    lo, hi = np.percentile(Z, [1, 99], axis=0)
    return PCAPreprocessor(win=win, n_components=n_components, mu=mu, sd=sd, pca=pca, lo=lo, hi=hi)


if __name__ == "__main__":
    X, y = generate_dataset(64)
    pre = fit_pca_preprocessor(X[:48])
    F = pre.transform(X[48:])
    print(X.shape, y.shape, F.shape, float(F.min()), float(F.max()))