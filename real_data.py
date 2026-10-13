"""real_data.py — SDNET2018 (real concrete crack images) loader, splits, atmospheric degradation, edge channel.

Dataset: SDNET2018 (Utah State Univ.), https://doi.org/10.15142/T3TD19
Expected layout after unzipping:  <root>/D/{CD,UD}  <root>/P/{CP,UP}  <root>/W/{CW,UW}
  D = bridge decks, P = pavements, W = walls;  C* = cracked (label 1), U* = uncracked (label 0).
SDNET2018 is RGB only (no NIR), so the 4th channel is a crack-aware Sobel edge map.
"""
from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

GROUPS = {"D": ("CD", "UD"), "P": ("CP", "UP"), "W": ("CW", "UW")}
GROUP_NAMES = ["Bridge deck", "Pavement", "Wall"]
IMG_EXT = (".jpg", ".jpeg", ".png")
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _find_root(root: str) -> str:
    base_depth = root.rstrip("/\\").count(os.sep)
    for dp, dn, _ in os.walk(root):
        if os.path.isdir(os.path.join(dp, "D", "CD")):
            return dp
        if dp.count(os.sep) - base_depth >= 2:
            dn[:] = []
    raise FileNotFoundError(
        f"SDNET2018 not found under '{root}'. Download from https://doi.org/10.15142/T3TD19 and unzip "
        "so that <root>/D/CD, <root>/D/UD, <root>/P/CP ... exist."
    )


def load_sdnet(root: str = "data/SDNET2018", per_class: int = 300, size: int = 128, seed: int = 42,
               cache_dir: str = "data/cache", use_cache: bool = True):
    """Balanced subset: per_class images per (structure type x cracked/uncracked).
    Returns X uint8 [N,3,size,size], y int [N] (1=cracked), g int [N] (0 deck,1 pavement,2 wall)."""
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f"sdnet_{size}_{per_class}_{seed}.npz")
    if use_cache and os.path.exists(cache):
        z = np.load(cache)
        return z["X"], z["y"], z["g"]

    root = _find_root(root)
    rng = np.random.default_rng(seed)
    Xs, ys, gs = [], [], []
    for gi, (key, (cdir, udir)) in enumerate(GROUPS.items()):
        for lab, sub in ((1, cdir), (0, udir)):
            d = os.path.join(root, key, sub)
            files = sorted(f for f in os.listdir(d) if f.lower().endswith(IMG_EXT))
            pick = rng.choice(len(files), size=min(per_class, len(files)), replace=False)
            for i in pick:
                im = Image.open(os.path.join(d, files[i])).convert("RGB").resize((size, size), Image.LANCZOS)
                Xs.append(np.asarray(im, dtype=np.uint8).transpose(2, 0, 1))
                ys.append(lab)
                gs.append(gi)
            print(f"  loaded {len(pick):4d} from {key}/{sub}")
    X, y, g = np.stack(Xs), np.array(ys, dtype=np.int64), np.array(gs, dtype=np.int64)
    np.savez_compressed(cache, X=X, y=y, g=g)
    return X, y, g


def make_splits(y: np.ndarray, g: np.ndarray, seed: int = 42, frac: Tuple[float, float] = (0.70, 0.15)):
    """Stratified (label x structure) train/val/test index arrays."""
    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for s in np.unique(y * 3 + g):
        idx = np.where(y * 3 + g == s)[0]
        rng.shuffle(idx)
        a, b = int(frac[0] * len(idx)), int((frac[0] + frac[1]) * len(idx))
        tr += list(idx[:a]); va += list(idx[a:b]); te += list(idx[b:])
    return np.array(tr), np.array(va), np.array(te)


# --------------------------------------------------------------------------- #
# Degradation + edge channel (torch, batched)
# --------------------------------------------------------------------------- #
def degrade(x: torch.Tensor, sigma: float = 0.0, cloud: float = 0.0, gen: torch.Generator | None = None):
    """x in [0,1], [B,3,H,W]. Semi-transparent cloud overlay (peak alpha=cloud) + Gaussian sensor noise."""
    B, _, H, W = x.shape
    if cloud > 0:
        a = F.interpolate(torch.rand(B, 1, 6, 6, generator=gen), size=(H, W), mode="bicubic", align_corners=False)
        a = a - a.amin((2, 3), keepdim=True)
        a = a / (a.amax((2, 3), keepdim=True) + 1e-8)
        alpha = cloud * a
        x = (1 - alpha) * x + alpha * 0.93
    if sigma > 0:
        x = x + sigma * torch.randn(x.shape, generator=gen)
    return x.clamp(0, 1)


_SX = torch.tensor([[-1.0, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)


def add_edge_channel(rgb: torch.Tensor) -> torch.Tensor:
    """[B,3,H,W] in [0,1] -> [B,4,H,W]; 4th channel = per-image standardized Sobel gradient magnitude."""
    lum = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
    gx = F.conv2d(lum, _SX, padding=1)
    gy = F.conv2d(lum, _SX.transpose(2, 3), padding=1)
    mag = F.avg_pool2d(torch.sqrt(gx ** 2 + gy ** 2 + 1e-8), 3, 1, 1)
    e = (mag - mag.mean((1, 2, 3), keepdim=True)) / (mag.std((1, 2, 3), keepdim=True) + 1e-6)
    return torch.cat([rgb, e], dim=1)


def prepare_batch(x_u8: torch.Tensor, sigma: float = 0.0, cloud: float = 0.0, gen=None) -> torch.Tensor:
    """uint8 [B,3,H,W] -> degraded, edge-augmented, ImageNet-normalized float [B,4,H,W]."""
    x = degrade(x_u8.float() / 255.0, sigma, cloud, gen)
    x = add_edge_channel(x)
    x[:, :3] = (x[:, :3] - IMAGENET_MEAN) / IMAGENET_STD
    return x
