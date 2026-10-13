"""hybrid_v2.py — Frozen ResNet-18 features -> supervised 4-qubit projection -> (AQFS-)QCNN -> attention -> head.

Models (all share backbone, projection, pooling and head so only the middle block differs):
  aqfs          : AQFS-QCNN (learnable re-uploading entanglement ring + softmax token attention)
  qcnn          : standard QCNN, mean pooling
  mlp           : classical control (tiny tanh MLP replacing the quantum block)
  linear_probe  : frozen features -> GAP -> linear  (strong classical reference)
"""
from __future__ import annotations

import warnings

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
import torchvision

import real_data as rd
from quantum_models import AQFSQCNNCore


class Backbone(nn.Module):
    """ResNet-18 up to layer3 (stride 16, 256 ch), 4-channel input (RGB + edge), frozen."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        try:
            w = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
            m = torchvision.models.resnet18(weights=w)
        except Exception as exc:  # offline / blocked download
            warnings.warn(f"Pretrained weights unavailable ({exc}); using random init.")
            m = torchvision.models.resnet18(weights=None)
        conv = nn.Conv2d(4, 64, 7, 2, 3, bias=False)
        with torch.no_grad():
            conv.weight[:, :3] = m.conv1.weight
            conv.weight[:, 3:] = m.conv1.weight.mean(1, keepdim=True)
        self.net = nn.Sequential(conv, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3)
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def extract_features(backbone: Backbone, x_u8: torch.Tensor, sigma=0.0, cloud=0.0, seed=0, bs=128):
    """uint8 images -> cached float16 feature maps [N,256,h,w]; degradation is seeded and reproducible."""
    out = []
    for s in range(0, len(x_u8), bs):
        gen = torch.Generator().manual_seed(seed * 100003 + s)
        out.append(backbone(rd.prepare_batch(x_u8[s:s + bs], sigma, cloud, gen)).half())
    return torch.cat(out)


class LinearProbe(nn.Module):
    mode = "linear_probe"

    def __init__(self, in_ch: int = 256):
        super().__init__()
        self.fc = nn.Sequential(nn.BatchNorm1d(in_ch), nn.Linear(in_ch, 1))

    def forward(self, f):
        return self.fc(f.float().mean((2, 3))).squeeze(-1)


class HybridV2(nn.Module):
    def __init__(self, mode: str = "aqfs", dropout: float = 0.2, hidden: int = 16, in_ch: int = 256):
        super().__init__()
        assert mode in ("aqfs", "qcnn", "mlp")
        self.mode = mode
        self.proj = nn.Sequential(nn.BatchNorm2d(in_ch), nn.Dropout(dropout), nn.Conv2d(in_ch, 4, 1))
        if mode == "mlp":
            self.core = None
            self.q = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 4), nn.Tanh())
        else:
            self.core = AQFSQCNNCore(use_aqfs=(mode == "aqfs"))
            init = {k: (lambda w: nn.init.normal_(w, std=0.5)) for k in self.core.weight_shapes}
            self.q = qml.qnn.TorchLayer(self.core.qnode, self.core.weight_shapes, init_method=init)
        self.attn = nn.Linear(4, 1) if mode == "aqfs" else None
        self.head = nn.Sequential(nn.Linear(4, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    # ---- stages ---------------------------------------------------------- #
    def encode(self, f: torch.Tensor) -> torch.Tensor:
        """features [B,256,h,w] -> token angles [B,T,4] in [0,pi] (learned, supervised reduction)."""
        a = torch.pi * torch.sigmoid(self.proj(f.float()))
        return a.flatten(2).transpose(1, 2)

    def quantum(self, angles: torch.Tensor) -> torch.Tensor:
        B, T, D = angles.shape
        out = self.q(angles.reshape(B * T, D))
        if out.dim() == 1:
            out = out.reshape(4, B * T).T
        return out.float().reshape(B, T, 4)

    def classify(self, q: torch.Tensor, keep: torch.Tensor | None = None):
        """q [B,T,4], keep optional bool mask [B,T] (False = token removed). Returns (logit, attention|None)."""
        if self.mode == "aqfs":
            s = self.attn(q).squeeze(-1)
            if keep is not None:
                s = s.masked_fill(~keep, -1e9)
            a = torch.softmax(s, 1)
            pooled = (a.unsqueeze(-1) * q).sum(1)
        else:
            a = None
            if keep is None:
                pooled = q.mean(1)
            else:
                k = keep.unsqueeze(-1).float()
                pooled = (q * k).sum(1) / k.sum(1).clamp(min=1)
        return self.head(pooled).squeeze(-1), a

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.classify(self.quantum(self.encode(f)))[0]


def trainable_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


class SmallCNN(nn.Module):
    """Classical from-scratch baseline on the 4-channel image (~60k params)."""
    mode = "cnn_scratch"

    def __init__(self):
        super().__init__()

        def blk(i, o, k, s):
            return nn.Sequential(nn.Conv2d(i, o, k, s, k // 2, bias=False), nn.BatchNorm2d(o), nn.ReLU())

        self.net = nn.Sequential(blk(4, 16, 5, 2), blk(16, 32, 3, 2), blk(32, 64, 3, 2), blk(64, 64, 3, 2),
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)
