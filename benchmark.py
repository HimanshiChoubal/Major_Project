"""benchmark.py — End-to-end publishable experiment on real SDNET2018 data.

Compares  linear_probe | mlp_ctrl | qcnn | aqfs | cnn_scratch  over several seeds, under atmospheric degradation, then runs:
  uncertainty/calibration (MC-dropout, temperature scaling, risk-coverage), faithfulness (deletion test),
  noisy-hardware + ZNE, and uncertainty-aware QAOA dispatch.  Outputs -> results/ (JSON, CSV, PNG).

Run:  python benchmark.py --root data/SDNET2018 --per-class 300 --seeds 3 --epochs 10
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score

import real_data as rd
import uncertainty as unc
from hybrid_v2 import Backbone, HybridV2, LinearProbe, SmallCNN, extract_features, trainable_params
from noisy_hardware import evaluate_noise_sweep
from routing_v2 import solve_risk_aware_dispatch

CONDITIONS = [  # (name, sensor sigma, cloud opacity)
    ("clean", 0.0, 0.0), ("s0.05", 0.05, 0.0), ("s0.10", 0.10, 0.0), ("s0.20", 0.20, 0.0), ("s0.30", 0.30, 0.0),
    ("c0.30", 0.0, 0.3), ("c0.60", 0.0, 0.6), ("c0.90", 0.0, 0.9), ("mixed", 0.15, 0.5),
]
MODELS = ["linear_probe", "mlp_ctrl", "qcnn", "aqfs", "cnn_scratch"]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def build(name: str):
    if name == "linear_probe":
        return LinearProbe()
    if name == "mlp_ctrl":
        return HybridV2("mlp")
    if name == "cnn_scratch":
        return SmallCNN()
    return HybridV2(name)


@torch.no_grad()
def probs_feats(model, F, bs=128):
    model.eval()
    return torch.cat([torch.sigmoid(model(F[s:s + bs])) for s in range(0, len(F), bs)]).numpy()


@torch.no_grad()
def probs_images(model, x_u8, sigma, cloud, seed=0, bs=128):
    model.eval()
    out = []
    for s in range(0, len(x_u8), bs):
        gen = torch.Generator().manual_seed(seed * 100003 + s)
        out.append(torch.sigmoid(model(rd.prepare_batch(x_u8[s:s + bs], sigma, cloud, gen))))
    return torch.cat(out).numpy()


def metrics(p, y):
    pred = (p >= 0.5).astype(int)
    auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")
    return {"acc": float((pred == y).mean()), "f1": float(f1_score(y, pred, zero_division=0)), "auc": float(auc)}


def train(model, kind, tr, va, yt, yv, epochs, lr, bs, wd=1e-4):
    """kind='feat' -> tr/va are cached feature tensors; kind='img' -> uint8 image tensors (on-the-fly prep)."""
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    steps = epochs * int(np.ceil(len(tr) / bs))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    crit = nn.BCEWithLogitsLoss()
    best, best_state, t0 = -1.0, None, time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(tr))
        for s in range(0, len(tr), bs):
            idx = perm[s:s + bs]
            xb = tr[idx] if kind == "feat" else rd.prepare_batch(tr[idx])
            opt.zero_grad()
            loss = crit(model(xb), yt[idx].float())
            loss.backward()
            nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            sched.step()
        pv = probs_feats(model, va) if kind == "feat" else probs_images(model, va, 0.0, 0.0)
        m = metrics(pv, yv.numpy())
        score = m["f1"] + m["acc"]
        if score > best:
            best, best_state = score, {k: v.clone() for k, v in model.state_dict().items()}
        print(f"    ep {ep + 1:02d}/{epochs} val acc {m['acc']:.3f} f1 {m['f1']:.3f}")
    model.load_state_dict(best_state)
    return time.time() - t0


# --------------------------------------------------------------------------- #
# Faithfulness: deletion test (no pixel masks exist in SDNET2018)
# --------------------------------------------------------------------------- #
def deletion_curve(model, F, y, rank: str, ks, seed=0):
    model.eval()
    Fc = F[y == 1]
    ang = model.encode(Fc).detach()
    B, T, _ = ang.shape
    with torch.no_grad():
        q = model.quantum(ang)
    if rank == "attention":
        with torch.no_grad():
            imp = model.classify(q)[1]
        if imp is None:
            return None
    elif rank == "gradient":
        a = ang.clone().requires_grad_(True)
        model.classify(model.quantum(a))[0].sum().backward()
        imp = a.grad.abs().sum(-1)
    else:
        imp = torch.rand(B, T, generator=torch.Generator().manual_seed(seed))
    order = imp.argsort(1, descending=True)
    curve = []
    with torch.no_grad():
        for k in ks:
            keep = torch.ones(B, T, dtype=torch.bool)
            keep.scatter_(1, order[:, :k], False)
            curve.append(float(torch.sigmoid(model.classify(q, keep)[0]).mean()))
    return curve


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_robustness(agg, out):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for ax, names, xs, xl in ((axs[0], ["clean", "s0.05", "s0.10", "s0.20", "s0.30"], [0, .05, .1, .2, .3], "Sensor noise σ"),
                              (axs[1], ["clean", "c0.30", "c0.60", "c0.90"], [0, .3, .6, .9], "Cloud opacity")):
        for m in MODELS:
            mu = [agg[m][c]["acc"][0] for c in names]
            sd = [agg[m][c]["acc"][1] for c in names]
            ax.errorbar(xs, mu, yerr=sd, marker="o", capsize=3, label=m)
        ax.set_xlabel(xl); ax.set_ylabel("Accuracy"); ax.grid(alpha=.3)
    axs[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, "robustness.png"), dpi=200); plt.close(fig)


def plot_uncertainty(cal, out):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    axs[0].plot([0.5, 1], [0.5, 1], "k--", lw=1)
    for k, lab in (("raw", "uncalibrated"), ("cal", "temp-scaled")):
        axs[0].plot(cal[k]["bin_conf"], cal[k]["bin_acc"], "o-", label=f"{lab} (ECE {cal[k]['ece']:.3f})")
    axs[0].set_xlabel("Confidence"); axs[0].set_ylabel("Accuracy"); axs[0].set_title("Reliability"); axs[0].legend(); axs[0].grid(alpha=.3)
    for name, (cov, err) in cal["risk_cov"].items():
        axs[1].plot(cov, err, label=name)
    axs[1].set_xlabel("Coverage"); axs[1].set_ylabel("Selective error"); axs[1].set_title("Risk–coverage"); axs[1].legend(); axs[1].grid(alpha=.3)
    axs[2].hist(cal["mi_correct"], bins=20, alpha=.6, label="correct", density=True)
    axs[2].hist(cal["mi_wrong"], bins=20, alpha=.6, label="wrong", density=True)
    axs[2].set_xlabel("Epistemic uncertainty (MI)"); axs[2].set_title("Uncertainty vs correctness"); axs[2].legend()
    fig.tight_layout(); fig.savefig(os.path.join(out, "uncertainty.png"), dpi=200); plt.close(fig)


def plot_deletion(dele, ks, out):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for k, c in dele.items():
        if c is not None:
            ax.plot(ks, c, marker="o", label=f"{k} (AUC {np.mean(c):.3f})")
    ax.set_xlabel("Tokens removed (most salient first)"); ax.set_ylabel("Mean P(cracked)")
    ax.set_title("Deletion test (lower = more faithful)"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(out, "deletion.png"), dpi=200); plt.close(fig)


def plot_hardware(rows, out):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    p = [r["p"] for r in rows]
    ax.plot(p, [r["acc_noisy"] for r in rows], "o-", label="noisy")
    ax.plot(p, [r["acc_zne"] for r in rows], "s-", label="ZNE-mitigated")
    ax.set_xlabel("Depolarizing probability per stage"); ax.set_ylabel("Accuracy"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(out, "hardware_noise.png"), dpi=200); plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/SDNET2018")
    ap.add_argument("--per-class", type=int, default=300)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--mc", type=int, default=20)
    ap.add_argument("--hw-images", type=int, default=40)
    ap.add_argument("--out", default="results")
    ap.add_argument("--no-pretrained", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    X, y, g = rd.load_sdnet(a.root, a.per_class, a.size)
    tr, va, te = rd.make_splits(y, g)
    Xt, Xv, Xe = (torch.from_numpy(X[i]) for i in (tr, va, te))
    yt, yv, ye = (torch.from_numpy(y[i]) for i in (tr, va, te))
    print(f"Data: train {len(tr)} | val {len(va)} | test {len(te)} | cracked frac {y.mean():.2f}")

    backbone = Backbone(pretrained=not a.no_pretrained)
    print("Extracting frozen backbone features...")
    Ftr, Fva = extract_features(backbone, Xt), extract_features(backbone, Xv)
    Fte = {c: extract_features(backbone, Xe, s, cl, seed=1) for c, s, cl in CONDITIONS}

    res = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    times, nparams, kept = defaultdict(list), {}, {}
    for seed in range(a.seeds):
        for name in MODELS:
            print(f"[seed {seed}] training {name}")
            torch.manual_seed(seed); np.random.seed(seed)
            m = build(name)
            if name == "cnn_scratch":
                dt = train(m, "img", Xt, Xv, yt, yv, a.epochs, 3e-3, a.bs)
            else:
                dt = train(m, "feat", Ftr, Fva, yt, yv, a.epochs, a.lr, a.bs)
            times[name].append(dt)
            nparams[name] = trainable_params(m)
            for c, s, cl in CONDITIONS:
                p = probs_images(m, Xe, s, cl, seed=1) if name == "cnn_scratch" else probs_feats(m, Fte[c])
                for k, v in metrics(p, ye.numpy()).items():
                    res[name][c][k].append(v)
            if seed == 0:
                kept[name] = m
            print(f"    clean acc {res[name]['clean']['acc'][-1]:.3f} | mixed acc {res[name]['mixed']['acc'][-1]:.3f} | {dt:.0f}s")

    # ---- aggregate + save ----
    agg = {m: {c: {k: (float(np.mean(v)), float(np.std(v))) for k, v in res[m][c].items()} for c in res[m]} for m in res}
    with open(os.path.join(a.out, "benchmark.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "trainable_params", "train_time_s", "condition", "acc_mean", "acc_std", "f1_mean", "f1_std", "auc_mean", "auc_std"])
        for m in MODELS:
            for c, _, _ in CONDITIONS:
                r = agg[m][c]
                w.writerow([m, nparams[m], f"{np.mean(times[m]):.1f}", c, *r["acc"], *r["f1"], *r["auc"]])
    plot_robustness(agg, a.out)
    print("\n=== Accuracy (mean±std over seeds) ===")
    print(f"{'model':<14}{'params':>8}" + "".join(f"{c:>12}" for c, _, _ in CONDITIONS))
    for m in MODELS:
        print(f"{m:<14}{nparams[m]:>8}" + "".join(f"{agg[m][c]['acc'][0]:>7.3f}±{agg[m][c]['acc'][1]:.2f}" for c, _, _ in CONDITIONS))

    extras = {}
    aq = kept["aqfs"]

    # ---- uncertainty / calibration (AQFS, seed 0) ----
    print("\nUncertainty + calibration...")
    val_logits = unc.predict_logits(aq, Fva)
    T = unc.fit_temperature(val_logits, yv)
    te_logits = unc.predict_logits(aq, Fte["clean"])
    p_raw, p_cal = torch.sigmoid(te_logits).numpy(), torch.sigmoid(te_logits / T).numpy()
    mc = unc.mc_dropout_predict(aq, Fte["clean"], a.mc)
    yn = ye.numpy()
    e_raw, bc_r, ba_r, _ = unc.ece(p_raw, yn)
    e_cal, bc_c, ba_c, _ = unc.ece(p_cal, yn)
    correct = ((mc["mean"] >= 0.5).astype(int) == yn)
    cal = {"temperature": T, "raw": {"ece": e_raw, "bin_conf": bc_r, "bin_acc": ba_r, "brier": unc.brier(p_raw, yn)},
           "cal": {"ece": e_cal, "bin_conf": bc_c, "bin_acc": ba_c, "brier": unc.brier(p_cal, yn)},
           "risk_cov": {}, "mi_correct": mc["mutual_info"][correct], "mi_wrong": mc["mutual_info"][~correct]}
    aurc = {}
    for name, u in (("MC entropy", mc["entropy"]), ("MC epistemic (MI)", mc["mutual_info"]),
                    ("random", np.random.default_rng(0).random(len(yn)))):
        cov, err, ar = unc.risk_coverage(mc["mean"], yn, u)
        cal["risk_cov"][name] = (cov, err); aurc[name] = ar
    plot_uncertainty(cal, a.out)
    extras["calibration"] = {"T": T, "ece_raw": e_raw, "ece_cal": e_cal, "brier_raw": cal["raw"]["brier"],
                             "brier_cal": cal["cal"]["brier"], "AURC": aurc}
    print(f"  T={T:.2f} | ECE {e_raw:.3f} -> {e_cal:.3f} | AURC {aurc}")

    # ---- faithfulness ----
    print("Deletion test...")
    ks = list(range(0, 65, 8))
    dele = {r: deletion_curve(aq, Fte["clean"], ye, r, ks) for r in ("attention", "gradient", "random")}
    plot_deletion(dele, ks, a.out)
    extras["deletion_auc"] = {k: float(np.mean(v)) for k, v in dele.items() if v is not None}

    # ---- noisy hardware ----
    print("Noisy hardware sweep (slow)...")
    hw = evaluate_noise_sweep(aq, Fte["clean"], ye, n_images=a.hw_images)
    plot_hardware(hw, a.out)
    extras["hardware"] = hw
    try:
        extras["qiskit"] = aq.core.qiskit_report()
    except Exception as exc:
        extras["qiskit"] = {"error": str(exc)}

    # ---- uncertainty-aware dispatch demo (5 test images as 5 sites) ----
    sites = np.random.default_rng(0).choice(len(yn), 5, replace=False)
    rm, rs = mc["mean"][sites], mc["std"][sites]
    extras["dispatch"] = {f"kappa={k}": solve_risk_aware_dispatch(rm, rs, kappa=k) for k in (0.0, 2.0)}
    for k, v in extras["dispatch"].items():
        print(f"  {k}: QAOA {v['qaoa']['sites']} | exact {v['exact']['sites']} | greedy {v['greedy']['sites']}")

    # ---- RUL demo with uncertainty ----
    t, h = unc.simulate_inspection_history()
    extras["rul"] = {"learned": {k: v for k, v in unc.fit_degradation(t, h).items() if k != "samples"},
                     "single_score": {k: v for k, v in unc.rul_from_score(float(mc["mean"][0]), float(mc["std"][0])).items() if k != "samples"}}

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o

    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump(clean({"aggregate": agg, "params": nparams, "train_time_s": {k: float(np.mean(v)) for k, v in times.items()},
                         "extras": extras}), f, indent=2)
    print(f"\nSaved everything to {a.out}/")


if __name__ == "__main__":
    main()
