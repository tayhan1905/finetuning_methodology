"""
view_principal_angles.py
========================
Load and visualise principal-angle results from results_v7 or results_v8.

Usage
-----
    python view_principal_angles.py --results_dir results_v7
    python view_principal_angles.py --results_dir results_v8

Outputs (saved alongside the script, or use --out_dir to redirect)
-------
    pa_final_summary.png         — bar chart: mean angle per layer (final ckpt)
    pa_epoch_trajectory.png      — line plot: mean angle averaged across layers per epoch
    pa_per_layer_trajectory.png  — heatmap: mean angle [layer × epoch]
    pa_angle_distribution.png    — violin/box: full angle distribution per epoch
                                   (uses data from the first .npz layer found)

Dependencies: numpy, pandas, matplotlib, scipy (optional, for smoothing)
"""

import argparse
import os
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless-safe; swap to "TkAgg" if you want a window
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--results_dir", default="results_v7",
                    help="Root results directory (results_v7 or results_v8)")
parser.add_argument("--out_dir", default=None,
                    help="Where to save plots (defaults to <results_dir>/principal_angles/plots)")
parser.add_argument("--top_k", type=int, default=None,
                    help="Use only the first top_k angles from each layer npz "
                         "(None = use all saved)")
parser.add_argument("--show", action="store_true",
                    help="Display plots interactively (requires a display)")
args = parser.parse_args()

PA_DIR  = os.path.join(args.results_dir, "principal_angles")
OUT_DIR = args.out_dir or os.path.join(PA_DIR, "plots")
os.makedirs(OUT_DIR, exist_ok=True)

print(f"\n{'='*60}")
print(f"  Principal-angle viewer")
print(f"  Source : {PA_DIR}")
print(f"  Output : {OUT_DIR}")
print(f"{'='*60}\n")


# ─────────────────────────────────────────────
# 1. Final comparison summary
# ─────────────────────────────────────────────
final_csv = os.path.join(PA_DIR, "final_comparison_summary.csv")
if os.path.exists(final_csv):
    df_final = pd.read_csv(final_csv)
    print("── Final comparison (last checkpoint) ──")
    print(df_final.to_string(index=False))
    print()

    # Shorten layer names for legibility
    short_names = [
        l.replace("bert.encoder.layer.", "L")
         .replace(".attention.self.", ".")
        for l in df_final["layer"]
    ]

    fig, ax = plt.subplots(figsize=(max(10, len(df_final) * 0.5), 5))
    x = np.arange(len(df_final))
    width = 0.4
    ax.bar(x - width/2, df_final["mean_angle_deg"], width, label="Mean angle (°)", color="#4C72B0")
    ax.bar(x + width/2, df_final["max_angle_deg"],  width, label="Max angle (°)",  color="#DD8452", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Principal angle (degrees)")
    ax.set_title(f"Final checkpoint — SALTEDORA vs Full FT\n{args.results_dir}")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "pa_final_summary.png")
    plt.savefig(path, dpi=150)
    print(f"  ✓ Saved: {path}")
    if args.show:
        plt.show()
    plt.close()
else:
    print(f"  ⚠  {final_csv} not found — skipping final summary plot.")


# ─────────────────────────────────────────────
# 2. Epoch trajectory summary
# ─────────────────────────────────────────────
traj_csv = os.path.join(PA_DIR, "epoch_trajectory_summary.csv")
if os.path.exists(traj_csv):
    df_traj = pd.read_csv(traj_csv)
    print("── Epoch trajectory (head rows) ──")
    print(df_traj.head(20).to_string(index=False))
    print()

    # 2a  Mean across all layers per epoch
    grouped = df_traj.groupby("epoch")["mean_angle_deg"].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(grouped["epoch"], grouped["mean"], marker="o", color="#4C72B0", label="Mean ± 1σ across layers")
    ax.fill_between(grouped["epoch"],
                    grouped["mean"] - grouped["std"],
                    grouped["mean"] + grouped["std"],
                    alpha=0.2, color="#4C72B0")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean principal angle (°)")
    ax.set_title(f"Layer-averaged PA trajectory\n{args.results_dir}")
    ax.legend()
    ax.grid(linestyle="--", alpha=0.4)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "pa_epoch_trajectory.png")
    plt.savefig(path, dpi=150)
    print(f"  ✓ Saved: {path}")
    if args.show:
        plt.show()
    plt.close()

    # 2b  Heatmap [layer × epoch]
    layers = df_traj["layer"].unique()
    epochs = sorted(df_traj["epoch"].unique())
    mat = np.full((len(layers), len(epochs)), np.nan)
    layer_idx = {l: i for i, l in enumerate(layers)}
    epoch_idx = {e: j for j, e in enumerate(epochs)}
    for _, row in df_traj.iterrows():
        i = layer_idx[row["layer"]]
        j = epoch_idx[row["epoch"]]
        mat[i, j] = row["mean_angle_deg"]

    short_layers = [
        l.replace("bert.encoder.layer.", "L")
         .replace(".attention.self.", ".")
        for l in layers
    ]

    fig, ax = plt.subplots(figsize=(max(6, len(epochs) * 1.2), max(6, len(layers) * 0.35)))
    im = ax.imshow(mat, aspect="auto", cmap="viridis", origin="upper")
    ax.set_xticks(range(len(epochs)))
    ax.set_xticklabels([f"ep{int(e)}" for e in epochs])
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(short_layers, fontsize=7)
    plt.colorbar(im, ax=ax, label="Mean angle (°)")
    ax.set_title(f"Principal angle heatmap [layer × epoch]\n{args.results_dir}")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "pa_per_layer_trajectory.png")
    plt.savefig(path, dpi=150)
    print(f"  ✓ Saved: {path}")
    if args.show:
        plt.show()
    plt.close()
else:
    print(f"  ⚠  {traj_csv} not found — skipping trajectory plots.")


# ─────────────────────────────────────────────
# 3. Full angle distribution from .npz
#    (uses the first layer's npz found)
# ─────────────────────────────────────────────
npz_files = sorted(glob.glob(os.path.join(PA_DIR, "*_trajectory_angles.npz")))
if npz_files:
    chosen_npz = npz_files[0]
    layer_label = os.path.basename(chosen_npz).replace("_trajectory_angles.npz", "").replace("_", ".")
    print(f"── Angle distribution from: {os.path.basename(chosen_npz)} ──")
    data = np.load(chosen_npz)
    keys = list(data.keys())
    print(f"   Keys: {keys}\n")

    # Collect per-epoch angle arrays
    epoch_angle_sets = []
    epoch_labels     = []
    for k in sorted(keys):
        if "angles_deg" in k:
            arr = data[k][:args.top_k] if args.top_k else data[k]
            epoch_angle_sets.append(arr)
            epoch_labels.append(k.replace("_angles_deg", "").replace("epoch_", "ep"))

    if epoch_angle_sets:
        fig, ax = plt.subplots(figsize=(max(7, len(epoch_angle_sets) * 1.2), 5))
        colors = cm.viridis(np.linspace(0.15, 0.85, len(epoch_angle_sets)))
        bp = ax.boxplot(epoch_angle_sets, patch_artist=True, labels=epoch_labels,
                        medianprops=dict(color="black", linewidth=1.5))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.75)
        ax.set_xlabel("Checkpoint")
        ax.set_ylabel("Principal angle (°)")
        ax.set_title(f"Full angle distribution — {layer_label}\n{args.results_dir}")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()
        path = os.path.join(OUT_DIR, "pa_angle_distribution.png")
        plt.savefig(path, dpi=150)
        print(f"  ✓ Saved: {path}")
        if args.show:
            plt.show()
        plt.close()

    # Print numeric summary for each checkpoint
    print(f"\n  Layer: {layer_label}")
    print(f"  {'Checkpoint':<20} {'Min':>8} {'Mean':>8} {'Median':>8} {'Max':>8}")
    print("  " + "-"*56)
    for lbl, arr in zip(epoch_labels, epoch_angle_sets):
        print(f"  {lbl:<20} {arr.min():>8.2f} {arr.mean():>8.2f} "
              f"{np.median(arr):>8.2f} {arr.max():>8.2f}")
    print()

    # List all npz files available
    print(f"  Available .npz layer files ({len(npz_files)} total):")
    for f in npz_files:
        print(f"    {os.path.basename(f)}")
else:
    print(f"  ⚠  No *_trajectory_angles.npz files found in {PA_DIR}.")

print(f"\n✅ All plots saved to: {OUT_DIR}\n")
