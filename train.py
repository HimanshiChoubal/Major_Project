"""train.py — Train the hybrid AQFS-QCNN and save weights to qcnn_weights.pt."""
from __future__ import annotations

import inspect

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import config
import data_generator
import pipeline
import quantum_models

WEIGHTS_PATH = "qcnn_weights.pt"


def to_angles(pre, X_raw: torch.Tensor) -> torch.Tensor:
    """Raw patches [N,C,H,W] -> quantum feature angles [N, n_windows, n_qubits] in [0, pi]."""
    if hasattr(pre, "transform"):  # PCAPreprocessor object (z-score + PCA + [0, pi] scaling)
        return pre.transform(X_raw)
    # dict preprocessor: {"pca": pca, "lo": lo, "hi": hi}
    wins = data_generator.extract_windows(X_raw, config.IMAGE.window_size)      # (N, T, C*win*win)
    N, T, D = wins.shape
    Z = pre["pca"].transform(wins.reshape(-1, D))
    Z = np.clip((Z - pre["lo"]) / (pre["hi"] - pre["lo"] + 1e-8), 0.0, 1.0) * np.pi
    return torch.from_numpy(Z.reshape(N, T, -1).astype(np.float32))


def main() -> None:
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    # 1. Raw dataset
    X_raw, y_raw = data_generator.generate_dataset(config.TRAIN.n_samples)
    y_raw = y_raw.float()
    n = len(X_raw)
    n_tr = int(config.TRAIN.train_frac * n)
    n_va = int(getattr(config.TRAIN, "val_frac", 0.15) * n)

    # 2. PCA fit on TRAIN windows only
    pre = data_generator.fit_pca_preprocessor(
        X_raw[:n_tr], config.IMAGE.window_size, config.QUANTUM.n_qubits
    )

    # 3. Quantum feature angles + splits
    F_all = to_angles(pre, X_raw)
    data_dict = {
        "train": (F_all[:n_tr], y_raw[:n_tr]),
        "val": (F_all[n_tr:n_tr + n_va], y_raw[n_tr:n_tr + n_va]),
        "test": (F_all[n_tr + n_va:], y_raw[n_tr + n_va:]),
    }
    print({k: tuple(v[0].shape) for k, v in data_dict.items()})

    # 4. Model + optimizer
    model = quantum_models.HybridQCNNNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.TRAIN.lr)

    # 5. Train — dispatch on the actual pipeline.train_hybrid_model signature
    params = inspect.signature(pipeline.train_hybrid_model).parameters
    if "train_loader" in params:
        bs = config.TRAIN.batch_size
        train_l = DataLoader(TensorDataset(*data_dict["train"]), batch_size=bs, shuffle=True)
        val_l = DataLoader(TensorDataset(*data_dict["val"]), batch_size=bs)
        test_l = DataLoader(TensorDataset(*data_dict["test"]), batch_size=bs)
        history = pipeline.train_hybrid_model(model, train_l, val_l, epochs=config.TRAIN.epochs)
        if hasattr(pipeline, "evaluate_model"):
            print("Test:", pipeline.evaluate_model(model, test_l))
    else:
        history = pipeline.train_hybrid_model(
            model,
            "AQFS-QCNN",
            data_dict,
            optimizer,
            config.TRAIN.epochs,
            config.TRAIN.batch_size,
        )

    # 6. Save weights
    torch.save(model.state_dict(), WEIGHTS_PATH)
    print(f"Saved weights to {WEIGHTS_PATH}")
    return history


if __name__ == "__main__":
    main()