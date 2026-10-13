"""app.py — Streamlit Decision Support System: Quantum-Enhanced Infrastructure Monitoring (AQFS-QCNN + QAOA)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

import config as cfg
from data_generator import generate_dataset, generate_synthetic_patch
from pipeline import (
    build_loaders,
    evaluate_model,
    inject_atmospheric_degradation,
    run_inference_with_saliency,
    set_preprocessor,
    train_hybrid_model,
)
from quantum_models import HybridQCNNNet, solve_qaoa_maintenance_route

st.set_page_config(page_title=cfg.APP.page_title, page_icon=cfg.APP.page_icon, layout=cfg.APP.layout)


# --------------------------------------------------------------------------- #
# Cached resources
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Training hybrid QCNN on synthetic satellite data...")
def get_system(n_samples: int, epochs: int, use_aqfs: bool):
    train_l, val_l, test_l, pre = build_loaders(n_samples=n_samples, batch_size=cfg.TRAIN.batch_size)
    net = HybridQCNNNet(use_aqfs=use_aqfs)
    hist = train_hybrid_model(net, train_l, val_l, epochs=epochs, verbose=False)
    test = evaluate_model(net, test_l)
    return net, pre, hist, test


@st.cache_data(show_spinner=False)
def sample_bank(n: int = 12, seed: int = cfg.SEED + 101):
    X, y, M = generate_dataset(n, seed=seed, return_masks=True)
    return X.numpy(), y.numpy(), M.numpy()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def load_uploaded(file) -> np.ndarray:
    """Return a (4,S,S) float32 patch from .npy (4xHxW or HxWx4) or an RGB image (NIR approximated)."""
    S = cfg.IMAGE.patch_size
    name = file.name.lower()
    if name.endswith(".npy"):
        arr = np.load(file, allow_pickle=False).astype(np.float32)
        if arr.ndim == 3 and arr.shape[-1] == cfg.IMAGE.n_bands and arr.shape[0] != cfg.IMAGE.n_bands:
            arr = arr.transpose(2, 0, 1)
        if arr.ndim != 3 or arr.shape[0] != cfg.IMAGE.n_bands:
            raise ValueError(f"Expected (4,H,W) or (H,W,4) array, got {arr.shape}")
    else:
        from PIL import Image

        img = np.asarray(Image.open(file).convert("RGB").resize((S, S)), dtype=np.float32) / 255.0
        rgb = img.transpose(2, 0, 1)
        lum = rgb.mean(0, keepdims=True)
        arr = np.concatenate([rgb, 0.30 + 0.5 * lum], axis=0).astype(np.float32)
    if arr.shape[-1] != S or arr.shape[-2] != S:
        arr = F.interpolate(torch.from_numpy(arr)[None], size=(S, S), mode="bilinear", align_corners=False)[0].numpy()
    return arr


def alert_card(score: float, thr: float) -> None:
    if score >= thr:
        color, title, msg = "#c62828", "HAZARD ALERT", "Probable micro-fissuring detected. Schedule priority field inspection."
    elif score >= 0.8 * thr:
        color, title, msg = "#ef6c00", "WATCH", "Elevated failure score near alert threshold. Increase monitoring cadence."
    else:
        color, title, msg = "#2e7d32", "NOMINAL", "No significant degradation signature detected."
    st.markdown(
        f"""<div style="background:{color};padding:14px 18px;border-radius:10px;color:white;">
        <div style="font-size:1.25rem;font-weight:700;">{title}</div>
        <div style="font-size:0.95rem;">Failure score <b>{score:.3f}</b> vs threshold <b>{thr:.2f}</b></div>
        <div style="font-size:0.9rem;margin-top:4px;">{msg}</div></div>""",
        unsafe_allow_html=True,
    )
def show_table(df: pd.DataFrame, index: bool = True) -> None:
    """Render a DataFrame as HTML (avoids pyarrow, which st.dataframe requires)."""
    st.markdown(
        df.to_html(index=index, float_format=lambda v: f"{v:.3f}", border=0),
        unsafe_allow_html=True,
    )    


def stretch_rgb(patch: np.ndarray) -> np.ndarray:
    rgb = patch[:3].transpose(1, 2, 0)
    lo, hi = np.percentile(rgb, [1, 99])
    return np.clip((rgb - lo) / (hi - lo + 1e-8), 0, 1)


def overlay(ax, base, heat, title: str) -> None:
    ax.imshow(base)
    if heat is None:
        ax.text(0.5, 0.5, "n/a\n(standard QCNN)", ha="center", va="center", transform=ax.transAxes, color="white",
                bbox=dict(facecolor="black", alpha=0.5))
    else:
        ax.imshow(heat, cmap="inferno", alpha=0.55, vmin=0, vmax=1)
        w = cfg.IMAGE.window_size
        for k in range(1, cfg.IMAGE.windows_per_axis):
            ax.axhline(k * w - 0.5, color="white", lw=0.4, alpha=0.35)
            ax.axvline(k * w - 0.5, color="white", lw=0.4, alpha=0.35)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def health_gauge(hi: float, limit: float):
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    zones = [(0, limit, "#c62828"), (limit, min(limit + 30, 100), "#ef6c00"), (min(limit + 30, 100), 100, "#2e7d32")]
    for lo, hi_z, col in zones:
        ax.add_patch(Wedge((0, 0), 1.0, 180 - hi_z * 1.8, 180 - lo * 1.8, width=0.3, color=col, alpha=0.9))
    ang = np.deg2rad(180 - np.clip(hi, 0, 100) * 1.8)
    ax.plot([0, 0.82 * np.cos(ang)], [0, 0.82 * np.sin(ang)], color="black", lw=3)
    ax.add_patch(plt.Circle((0, 0), 0.05, color="black"))
    ax.text(0, -0.28, f"{hi:.1f}", ha="center", fontsize=22, fontweight="bold")
    ax.text(0, -0.52, "Health Index", ha="center", fontsize=10)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-0.7, 1.1)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig


def degradation_projection(hi0: float, score: float, limit: float, horizon: int):
    k = 0.02 + 0.12 * score                       # monthly decay rate (heuristic mapping from failure score)
    t_hist = np.arange(-12, 1)
    hist_hi = np.minimum(100.0, hi0 * np.exp(k * (-t_hist)))
    t_f = np.linspace(0, horizon, 120)
    mid, fast, slow = (hi0 * np.exp(-kk * t_f) for kk in (k, 1.3 * k, 0.7 * k))

    def rul(kk: float) -> float:
        return 0.0 if hi0 <= limit else float(np.log(hi0 / limit) / kk)

    r_mid, r_lo, r_hi = rul(k), rul(1.3 * k), rul(0.7 * k)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(t_hist, hist_hi, color="gray", ls="--", label="Reconstructed history")
    ax.plot(t_f, mid, color="#1565c0", lw=2.2, label="Projected Health Index")
    ax.fill_between(t_f, fast, slow, color="#1565c0", alpha=0.18, label="Uncertainty band (±30% decay)")
    ax.axhline(limit, color="#c62828", ls=":", lw=1.5, label=f"Critical limit ({limit:.0f})")
    if r_mid <= horizon:
        ax.axvline(r_mid, color="#c62828", alpha=0.6)
        ax.text(r_mid, 102, f" RUL≈{r_mid:.1f} mo", color="#c62828", fontsize=9, va="bottom")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_ylim(0, 108)
    ax.set_xlabel("Months from now")
    ax.set_ylabel("Health Index")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left", fontsize=8)
    return fig, r_mid, r_lo, r_hi


def route_figure(risks: np.ndarray, res: dict):
    names, coords = cfg.QAOA.node_names, np.array(cfg.QAOA.coords)
    sel = set(res["selected_indices"])
    route = res["route_indices"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.3, 1]})

    n = len(names)
    for i in range(n):
        for j in range(i + 1, n):
            ax.plot(*zip(coords[i], coords[j]), color="gray", alpha=0.15, lw=0.8)
    for a, b in zip(route[:-1], route[1:]):
        ax.annotate("", xy=coords[b], xytext=coords[a],
                    arrowprops=dict(arrowstyle="-|>", color="#1565c0", lw=2.4, shrinkA=14, shrinkB=14))
    sc = ax.scatter(coords[:, 0], coords[:, 1], s=350 + 900 * risks, c=risks, cmap="RdYlGn_r", vmin=0, vmax=1,
                    edgecolors=["black" if i in sel else "#bdbdbd" for i in range(n)],
                    linewidths=[3 if i in sel else 1 for i in range(n)], zorder=3)
    for i, nm in enumerate(names):
        ax.text(coords[i, 0], coords[i, 1] - 0.55, f"{nm}\nrisk {risks[i]:.2f}", ha="center", va="top", fontsize=8)
    ax.scatter(*coords[cfg.QAOA.depot_index], marker="*", s=260, c="gold", edgecolors="black", zorder=4)
    for order, idx in enumerate(route[1:-1], start=1):
        ax.text(coords[idx, 0] + 0.15, coords[idx, 1] + 0.25, f"#{order}", fontsize=11, fontweight="bold",
                color="#1565c0", zorder=5)
    ax.set_title(f"Dispatch map  (route {res['route_length_km']:.1f} km, ★ = depot)")
    ax.set_xlabel("km")
    ax.set_ylabel("km")
    ax.margins(0.2)
    ax.grid(alpha=0.2)
    plt.colorbar(sc, ax=ax, label="Failure risk", fraction=0.04)

    order_idx = np.argsort(risks)[::-1]
    stop_no = {idx: k for k, idx in enumerate(route[1:-1], start=1)}
    colors = ["#1565c0" if i in sel else "#bdbdbd" for i in order_idx]
    ax2.barh([names[i] for i in order_idx][::-1], risks[order_idx][::-1], color=colors[::-1])
    for y_pos, i in enumerate(order_idx[::-1]):
        lab = f"stop #{stop_no[i]}" if i in stop_no else ("depot" if i in sel else "")
        ax2.text(risks[i] + 0.01, y_pos, lab, va="center", fontsize=9)
    ax2.set_xlim(0, 1.15)
    ax2.set_title("Priority graph (blue = dispatched)")
    ax2.set_xlabel("Failure risk")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
st.title(f"{cfg.APP.page_icon} {cfg.APP.page_title}")
st.caption("Hybrid quantum-classical decision support: AQFS-QCNN inspection, predictive maintenance, QAOA crew dispatch.")

with st.sidebar:
    st.header("Input")
    source = st.radio("Patch source", ["Sample bank", "Upload"], horizontal=True)
    Xb, yb, Mb = sample_bank()
    patch_raw, truth = None, None
    if source == "Sample bank":
        opts = [f"Sample {i:02d}" for i in range(len(Xb))]
        pick = st.selectbox("Select patch", range(len(opts)), format_func=lambda i: opts[i])
        patch_raw, truth = Xb[pick].copy(), int(yb[pick])
    else:
        up = st.file_uploader("Upload .npy (4×H×W) or RGB image", type=["npy", "png", "jpg", "jpeg"])
        if up is not None:
            try:
                patch_raw = load_uploaded(up)
            except Exception as exc:
                st.error(f"Could not read file: {exc}")
        if patch_raw is None:
            st.info("No valid upload — using Sample 00.")
            patch_raw, truth = Xb[0].copy(), int(yb[0])

    st.header("Atmospheric stress")
    cloud_opacity = st.slider("Cloud opacity", 0.0, 1.0, 0.0, 0.05)
    noise_sigma = st.slider("Sensor noise σ", 0.0, 0.4, 0.0, 0.01)
    deg_seed = st.number_input("Degradation seed", 0, 9999, 0, 1)

    st.header("Alerting")
    threshold = st.slider("Hazard alert threshold", 0.1, 0.9, float(cfg.APP.anomaly_threshold), 0.05)

    with st.expander("Model settings"):
        use_aqfs = st.checkbox("Use AQFS block", value=cfg.QUANTUM.use_aqfs)
        n_train = st.slider("Synthetic training samples", 200, 1000, 400, 100)
        n_epochs = st.slider("Training epochs", 2, 10, 5, 1)

model, pre, hist, test_metrics = get_system(n_train, n_epochs, use_aqfs)
set_preprocessor(pre)
with st.sidebar:
    st.caption(f"Model test acc {test_metrics['acc']:.3f} · F1 {test_metrics['f1']:.3f}")

patch_deg = inject_atmospheric_degradation(patch_raw, noise_sigma, cloud_opacity, seed=int(deg_seed))
res = run_inference_with_saliency(model, patch_deg, pre)
score, hi0 = res["failure_score"], res["health_index"]

tab1, tab2, tab3 = st.tabs([
    "Live Inspection & QCNN Analytics",
    "Predictive Maintenance & RUL",
    "Quantum QAOA Route Optimization",
])

# --------------------------------------------------------------------------- #
# Tab 1
# --------------------------------------------------------------------------- #
with tab1:
    c1, c2, c3, c4 = st.columns([1.6, 1, 1, 1])
    with c1:
        alert_card(score, threshold)
    c2.metric("Failure score", f"{score:.3f}")
    c3.metric("Health Index", f"{hi0:.1f}")
    c4.metric("Predicted class", "Fissured" if score >= threshold else "Healthy",
              delta=None if truth is None else f"truth: {cfg.ASSET_CLASSES[truth]}", delta_color="off")

    base = stretch_rgb(patch_deg)
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.8))
    axs[0].imshow(base)
    axs[0].set_title("RGB (degraded)", fontsize=10)
    axs[0].axis("off")
    im = axs[1].imshow(patch_deg[3], cmap="viridis")
    axs[1].set_title("NIR band", fontsize=10)
    axs[1].axis("off")
    overlay(axs[2], base, res["attention_map_full"], "AQFS quantum attention")
    overlay(axs[3], base, res["saliency_map_full"], "Gradient saliency")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    with st.expander("Window-level detail (4×4 grid)"):
        g = cfg.IMAGE.windows_per_axis
        df = pd.DataFrame({
            "Saliency": res["saliency_map"].reshape(-1),
            **({"AQFS attention": res["attention_map"].reshape(-1)} if res["attention_map"] is not None else {}),
        }, index=[f"W{r}{c}" for r in range(g) for c in range(g)])
        show_table(df)

# --------------------------------------------------------------------------- #
# Tab 2
# --------------------------------------------------------------------------- #
with tab2:
    cA, cB = st.columns([1, 2])
    with cA:
        limit = st.slider("Critical Health Index limit", 10, 70, 40, 5)
        horizon = st.slider("Projection horizon (months)", 6, 60, 36, 6)
        st.pyplot(health_gauge(hi0, limit), use_container_width=True)
        plt.close("all")
    fig_ts, r_mid, r_lo, r_hi = degradation_projection(hi0, score, limit, horizon)
    with cB:
        st.pyplot(fig_ts, use_container_width=True)
        plt.close(fig_ts)
    m1, m2, m3 = st.columns(3)
    m1.metric("Estimated RUL", f"{r_mid:.1f} mo")
    m2.metric("Pessimistic RUL", f"{r_lo:.1f} mo")
    m3.metric("Optimistic RUL", f"{r_hi:.1f} mo")
    st.caption("RUL uses an exponential degradation heuristic seeded by the QCNN failure score; "
               "calibrate with real inspection histories before operational use.")

# --------------------------------------------------------------------------- #
# Tab 3
# --------------------------------------------------------------------------- #
with tab3:
    n_sites = cfg.QAOA.n_nodes
    names = cfg.QAOA.node_names
    left, right = st.columns([1, 2])

    with left:
        mode = st.radio("Site risk source", ["Model-scored (synthetic site patches)", "Manual"])
        if mode.startswith("Model"):
            scen_seed = st.number_input("Scenario seed", 0, 9999, 7, 1)
            key = (int(scen_seed), cloud_opacity, noise_sigma, n_train, n_epochs, use_aqfs)
            if st.session_state.get("site_key") != key:
                with st.spinner("Scoring site patches..."):
                    rng = np.random.default_rng(int(scen_seed))
                    labels = (rng.random(n_sites) < 0.5).astype(int)
                    vals = []
                    for i, lab in enumerate(labels):
                        p = generate_synthetic_patch(int(lab), rng)
                        p = inject_atmospheric_degradation(p, noise_sigma, cloud_opacity, seed=int(scen_seed) + i)
                        vals.append(run_inference_with_saliency(model, p, pre)["failure_score"])
                st.session_state["site_key"], st.session_state["site_risks"] = key, np.array(vals)
            risks = st.session_state["site_risks"].copy()
            if st.checkbox(f"Use current inspection as {names[0]}", value=True):
                risks[0] = score
        else:
            risks = np.array([st.slider(nm, 0.0, 1.0, float(v), 0.01)
                              for nm, v in zip(names, [0.8, 0.25, 0.6, 0.35, 0.7])])

        with st.expander("QAOA settings", expanded=True):
            max_sites = st.slider("Crew capacity (sites)", 1, n_sites, 3)
            risk_w = st.slider("Risk weight", 0.1, 3.0, 1.0, 0.1)
            travel_w = st.slider("Travel weight", 0.0, 1.0, 0.15, 0.05)
            p_layers = st.slider("QAOA layers (p)", 1, 3, cfg.QAOA.p_layers)
            steps = st.slider("Optimizer steps", 20, 150, cfg.QAOA.n_steps, 10)
        run = st.button("Run QAOA dispatch optimization", type="primary", use_container_width=True)

    if run:
        with st.spinner("Optimizing QAOA circuit..."):
            st.session_state["qaoa_res"] = solve_qaoa_maintenance_route(
                risks, max_sites=max_sites, p_layers=p_layers, n_steps=steps,
                step_size=cfg.QAOA.step_size, risk_weight=risk_w, travel_weight=travel_w,
            )
            st.session_state["qaoa_risks"] = risks.copy()

    with right:
        qres = st.session_state.get("qaoa_res")
        if qres is None:
            st.info("Set site risks and press **Run QAOA dispatch optimization**.")
        else:
            used = st.session_state["qaoa_risks"]
            if not np.allclose(used, risks):
                st.warning("Inputs changed since the last run — re-run to refresh the route.")
            if not qres["selected_indices"]:
                st.info("QAOA selected no sites (risk does not justify dispatch under current weights).")
            else:
                fig_r = route_figure(used, qres)
                st.pyplot(fig_r, use_container_width=True)
                plt.close(fig_r)

                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Route length", f"{qres['route_length_km']:.1f} km")
                k2.metric("Risk covered", f"{qres['covered_risk']:.2f}")
                k3.metric("QAOA state prob.", f"{qres['qaoa_state_probability']:.2f}")
                k4.metric("Matches classical optimum", "Yes" if qres["matches_classical_optimum"] else "No")
                st.markdown("**Dispatch route:** " + " → ".join(qres["route"]))

                stops = {idx: k for k, idx in enumerate(qres["route_indices"][1:-1], start=1)}
                D = np.array(cfg.QAOA.distance_matrix)
                table = pd.DataFrame({
                    "Site": names,
                    "Risk": used,
                    "Dispatched": [i in qres["selected_indices"] for i in range(n_sites)],
                    "Stop #": [stops.get(i, "—") for i in range(n_sites)],
                    "Dist. from depot (km)": D[cfg.QAOA.depot_index],
                }).sort_values("Risk", ascending=False)
                show_table(table, index=False)

                site = st.selectbox("Inspect site", names)
                si = names.index(site)
                st.write(f"**{site}** · risk {used[si]:.2f} · coords {cfg.QAOA.coords[si]} km · "
                         f"{'dispatched (stop ' + str(stops.get(si, 'depot')) + ')' if si in qres['selected_indices'] else 'not dispatched'}")
                fig_c, ax_c = plt.subplots(figsize=(6, 2.4))
                ax_c.plot(qres["cost_history"], color="#1565c0")
                ax_c.set_xlabel("Optimizer step")
                ax_c.set_ylabel("Expected QAOA cost")
                ax_c.grid(alpha=0.25)
                st.pyplot(fig_c)
                plt.close(fig_c)
