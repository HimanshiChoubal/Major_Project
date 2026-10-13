"""pipeline.py — Training loop, single-patch inference with saliency, and atmospheric stress-testing."""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

import config as cfg
from data_generator import PCAPreprocessor, generate_dataset, fit_pca_preprocessor
from quantum_models import HybridQCNNNet

# Preprocessor registry so `run_inference_with_saliency(model, raw_patch)` works with the spec'd signature
_PREPROCESSOR: Optional[PCAPreprocessor] = None


def set_preprocessor(pre: PCAPreprocessor) -> None:
    global _PREPROCESSOR
    _PREPROCESSOR = pre


def get_preprocessor() -> Optional[PCAPreprocessor]:
    return _PREPROCESSOR


def get_device_free_seed(seed: int = cfg.SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #
def build_loaders(
    n_samples: int = cfg.TRAIN.n_samples,
    batch_size: int = cfg.TRAIN.batch_size,
    seed: int = cfg.SEED,
) -> Tuple[DataLoader, DataLoader, DataLoader, PCAPreprocessor]:
    """Generate data, fit PCA on train only, and return (train, val, test) window-feature loaders."""
    get_device_free_seed(seed)
    X, y = generate_dataset(n_samples, seed=seed)
    n_tr = int(cfg.TRAIN.train_frac * n_samples)
    n_va = int(cfg.TRAIN.val_frac * n_samples)

    pre = fit_pca_preprocessor(X[:n_tr])
    set_preprocessor(pre)

    def mk(lo: int, hi: int, shuffle: bool) -> DataLoader:
        ds = TensorDataset(pre.transform(X[lo:hi]), y[lo:hi])
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    return mk(0, n_tr, True), mk(n_tr, n_tr + n_va, False), mk(n_tr + n_va, n_samples, False), pre


# --------------------------------------------------------------------------- #
# 1. Training
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader) -> Dict[str, float]:
    model.eval()
    crit = nn.BCEWithLogitsLoss(reduction="sum")
    total, loss_sum, preds, labels = 0, 0.0, [], []
    for xb, yb in loader:
        logits = model(xb)
        loss_sum += float(crit(logits, yb))
        preds.append((torch.sigmoid(logits) >= cfg.APP.anomaly_threshold).long())
        labels.append(yb.long())
        total += len(yb)
    p, t = torch.cat(preds).numpy(), torch.cat(labels).numpy()
    return {
        "loss": loss_sum / max(total, 1),
        "acc": float((p == t).mean()),
        "f1": float(f1_score(t, p, zero_division=0)),
    }


def train_hybrid_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = cfg.TRAIN.epochs,
    lr: float = cfg.TRAIN.lr,
    verbose: bool = True,
) -> Dict[str, list]:
    """BCE + Adam loop. Returns history: train/val loss, acc, f1 and per-epoch wall time."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.BCEWithLogitsLoss()
    hist: Dict[str, list] = defaultdict(list)

    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        loss_sum, n, preds, labels = 0.0, 0, [], []
        for xb, yb in train_loader:
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            loss_sum += float(loss) * len(yb)
            n += len(yb)
            preds.append((torch.sigmoid(logits.detach()) >= cfg.APP.anomaly_threshold).long())
            labels.append(yb.long())
        epoch_time = time.time() - t0

        p, t = torch.cat(preds).numpy(), torch.cat(labels).numpy()
        val = evaluate_model(model, val_loader)

        hist["train_loss"].append(loss_sum / n)
        hist["train_acc"].append(float((p == t).mean()))
        hist["train_f1"].append(float(f1_score(t, p, zero_division=0)))
        hist["val_loss"].append(val["loss"])
        hist["val_acc"].append(val["acc"])
        hist["val_f1"].append(val["f1"])
        hist["epoch_time_s"].append(epoch_time)

        if verbose:
            print(
                f"Epoch {ep:02d}/{epochs} | loss {hist['train_loss'][-1]:.4f} "
                f"acc {hist['train_acc'][-1]:.3f} f1 {hist['train_f1'][-1]:.3f} | "
                f"val loss {val['loss']:.4f} acc {val['acc']:.3f} f1 {val['f1']:.3f} | {epoch_time:.1f}s"
            )
    return dict(hist)


# --------------------------------------------------------------------------- #
# 3. Atmospheric degradation
# --------------------------------------------------------------------------- #
_CLOUD_RGBN = np.array([0.92, 0.93, 0.95, 0.85], dtype=np.float32)[:, None, None]


def inject_atmospheric_degradation(
    patch,
    noise_sigma: float = 0.15,
    cloud_opacity: float = 0.5,
    seed: Optional[int] = None,
):
    """Overlay a smooth cloud layer (peak alpha = cloud_opacity) and Gaussian sensor noise.
    Accepts [C,H,W] or [N,C,H,W]; numpy in -> numpy out, tensor in -> tensor out."""
    is_tensor = isinstance(patch, torch.Tensor)
    arr = patch.detach().cpu().numpy() if is_tensor else np.asarray(patch)
    arr = arr.astype(np.float32)
    squeeze = arr.ndim == 3
    if squeeze:
        arr = arr[None]

    rng = np.random.default_rng(seed)
    N, C, H, W = arr.shape
    out = np.empty_like(arr)
    for i in range(N):
        a = gaussian_filter(rng.random((H, W)), max(H / 8, 1.0))
        a = (a - a.min()) / (a.max() - a.min() + 1e-8)
        alpha = np.clip(cloud_opacity, 0.0, 1.0) * a
        img = (1.0 - alpha) * arr[i] + alpha * _CLOUD_RGBN[:C]
        out[i] = img + rng.normal(0.0, noise_sigma, img.shape).astype(np.float32)
    out = out[0] if squeeze else out
    return torch.from_numpy(out) if is_tensor else out


# --------------------------------------------------------------------------- #
# 2. Inference + saliency
# --------------------------------------------------------------------------- #
def _norm01(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    return (a - a.min()) / (a.max() - a.min() + 1e-8)


def upsample_map(m: np.ndarray, size: int = cfg.IMAGE.patch_size) -> np.ndarray:
    """Bilinear upsample a [g,g] window map to [size,size] for overlay plots."""
    t = torch.from_numpy(np.asarray(m, dtype=np.float32))[None, None]
    return F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)[0, 0].numpy()


def run_inference_with_saliency(
    model: HybridQCNNNet,
    raw_patch,
    preprocessor: Optional[PCAPreprocessor] = None,
) -> Dict[str, object]:
    """Single raw patch [4,S,S] -> failure score, health index, AQFS attention heatmap, gradient saliency."""
    pre = preprocessor or _PREPROCESSOR
    if pre is None:
        raise RuntimeError("No preprocessor: call set_preprocessor(pre) or pass preprocessor=...")

    arr = raw_patch.detach().cpu().numpy() if isinstance(raw_patch, torch.Tensor) else np.asarray(raw_patch)
    feats = pre.transform(arr[None].astype(np.float32)).clone().requires_grad_(True)   # [1,T,4]

    g = cfg.IMAGE.windows_per_axis
    model.eval()
    with torch.enable_grad():
        model.zero_grad(set_to_none=True)
        logit = model(feats)[0]
        logit.backward()
    sal = feats.grad.abs().sum(-1).reshape(g, g).numpy()

    prob = float(torch.sigmoid(logit.detach()))
    attn = None
    if getattr(model, "use_aqfs", False) and model.last_attention is not None:
        attn = model.last_attention[0].reshape(g, g).cpu().numpy()

    return {
        "failure_score": prob,
        "health_index": cfg.APP.health_index_scale * (1.0 - prob),
        "label": int(prob >= cfg.APP.anomaly_threshold),
        "label_name": cfg.ASSET_CLASSES[int(prob >= cfg.APP.anomaly_threshold)],
        "attention_map": None if attn is None else _norm01(attn),            # [g,g]
        "attention_map_full": None if attn is None else upsample_map(_norm01(attn)),
        "saliency_map": _norm01(sal),                                         # [g,g]
        "saliency_map_full": upsample_map(_norm01(sal)),
    }


if __name__ == "__main__":
    tr, va, te, pre = build_loaders(n_samples=96, batch_size=16)
    net = HybridQCNNNet(use_aqfs=True)
    hist = train_hybrid_model(net, tr, va, epochs=1)
    Xs, ys = generate_dataset(4, seed=7)
    noisy = inject_atmospheric_degradation(Xs[0], noise_sigma=0.2, cloud_opacity=0.6, seed=0)
    res = run_inference_with_saliency(net, noisy)
    print({k: (v.shape if isinstance(v, np.ndarray) else v) for k, v in res.items()})