"""
plot_v7_results.py
==================
Reads v7 results from results/data/ and produces publication-quality plots.

SALTEDORA r=64 data is always expected:
    results/data/results_v7/final_comparison_summary.csv
    results/data/results_v7/epoch_trajectory_summary.csv

DoRA r=64 data is optional — drop these files into results/data/results_dora/
and the script will automatically include DoRA in all comparison plots:
    results/data/results_dora/final_comparison_summary.csv
    results/data/results_dora/epoch_trajectory_summary.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR  = "/sessions/wizardly-awesome-goodall/mnt/finetuning_methodology/results"
SALT_DIR  = os.path.join(BASE_DIR, "data", "results_v7")
DORA_DIR  = os.path.join(BASE_DIR, "data", "results_dora")
OUT_DIR   = os.path.join(BASE_DIR, "plots")
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# Load SALTEDORA data (always present)
# ─────────────────────────────────────────────
df_final = pd.read_csv(os.path.join(SALT_DIR, "final_comparison_summary.csv"))
df_traj  = pd.read_csv(os.path.join(SALT_DIR, "epoch_trajectory_summary.csv"))

# ─────────────────────────────────────────────
# Load DoRA data (optional)
# ─────────────────────────────────────────────
dora_final_path = os.path.join(DORA_DIR, "final_comparison_summary.csv")
dora_traj_path  = os.path.join(DORA_DIR, "epoch_trajectory_summary.csv")
DORA_AVAILABLE  = os.path.exists(dora_final_path) and os.path.exists(dora_traj_path)

if DORA_AVAILABLE:
    df_dora_final = pd.read_csv(dora_final_path)
    df_dora_traj  = pd.read_csv(dora_traj_path)
    print("✓ DoRA data found — comparison plots will include DoRA")
else:
    print("ℹ  No DoRA data found — plotting SALTEDORA only")
    print(f"   (drop final_comparison_summary.csv and "
          f"epoch_trajectory_summary.csv into {DORA_DIR} to enable)")


# ─────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────
def parse_layer(name):
    layer = int(name.split(".layer.")[1].split(".")[0])
    head  = name.split("self.")[1]
    return layer, head

def add_parsed_cols(df):
    df[["layer_idx", "head"]] = pd.DataFrame(
        df["layer"].apply(parse_layer).tolist(), index=df.index)
    return df

df_final = add_parsed_cols(df_final)
df_traj  = add_parsed_cols(df_traj)
if DORA_AVAILABLE:
    df_dora_final = add_parsed_cols(df_dora_final)
    df_dora_traj  = add_parsed_cols(df_dora_traj)

layers     = list(range(12))
head_types = ["query", "key", "value"]
epochs     = sorted(df_traj["epoch"].unique())
colors     = {"query": "#4C72B0", "key": "#DD8452", "value": "#55A868"}
TITLE_PAD  = 10

# Line styles: SALTEDORA solid, DoRA dashed
styles = {
    "saltedora": {"linestyle": "-",  "marker": "o", "alpha": 1.0},
    "dora":      {"linestyle": "--", "marker": "s", "alpha": 0.85},
}


# ═══════════════════════════════════════════════════════════════════
# Plot 1 — Final checkpoint: mean angle per layer, Q/K/V line plot
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 5))

for head in head_types:
    sub = df_final[df_final["head"] == head].sort_values("layer_idx")
    ax.plot(sub["layer_idx"], sub["mean_angle_deg"],
            color=colors[head], label=f"{head.capitalize()} — SALTEDORA",
            linewidth=2, markersize=6, **styles["saltedora"])

    if DORA_AVAILABLE:
        sub_d = df_dora_final[df_dora_final["head"] == head].sort_values("layer_idx")
        ax.plot(sub_d["layer_idx"], sub_d["mean_angle_deg"],
                color=colors[head], label=f"{head.capitalize()} — DoRA",
                linewidth=2, markersize=6, **styles["dora"])

ax.set_xlabel("BERT Encoder Layer", fontsize=12)
ax.set_ylabel("Mean Principal Angle (°)", fontsize=12)
title_suffix = " vs DoRA" if DORA_AVAILABLE else ""
ax.set_title(f"SALTEDORA r=64{title_suffix} vs Full Fine-Tuning\n"
             f"Mean Principal Angles — Final Checkpoint",
             fontsize=13, fontweight="bold", pad=TITLE_PAD)
ax.set_xticks(layers)
ax.set_xticklabels([f"L{l}" for l in layers])
ax.set_ylim(0, 52)
ax.legend(title="Head · Method", fontsize=9, ncol=2 if DORA_AVAILABLE else 1)
ax.grid(linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "1_final_mean_angle_line.png"), dpi=150, bbox_inches="tight")
plt.close()
print("✓ Plot 1 saved")


# ═══════════════════════════════════════════════════════════════════
# Plot 2 — Final checkpoint: heatmap [head × layer]
#          If DoRA available: side-by-side heatmaps
# ═══════════════════════════════════════════════════════════════════
def make_mat(df):
    return np.array([
        [df[(df["layer_idx"] == l) & (df["head"] == h)]["mean_angle_deg"].values[0]
         for l in layers]
        for h in head_types
    ])

mat_salt = make_mat(df_final)

if DORA_AVAILABLE:
    mat_dora = make_mat(df_dora_final)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    datasets  = [("SALTEDORA r=64", mat_salt, axes[0]),
                 ("DoRA r=64",      mat_dora, axes[1])]
else:
    fig, axes = plt.subplots(1, 1, figsize=(12, 3.5))
    datasets  = [("SALTEDORA r=64", mat_salt, axes)]

for title, mat, ax in datasets:
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=50)
    plt.colorbar(im, ax=ax, label="Mean angle (°)", shrink=0.9)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([f"L{l}" for l in layers])
    ax.set_yticks(range(len(head_types)))
    ax.set_yticklabels([h.capitalize() for h in head_types], fontsize=11)
    ax.set_xlabel("BERT Encoder Layer", fontsize=11)
    ax.set_title(f"{title} vs Full FT — Final Checkpoint",
                 fontsize=11, fontweight="bold")
    for i, h in enumerate(head_types):
        for j in range(len(layers)):
            val = mat[i, j]
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    fontsize=7, color="black" if val < 30 else "white", fontweight="bold")

fig.suptitle("Principal Angle Heatmap — Final Checkpoint",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "2_final_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close()
print("✓ Plot 2 saved")


# ═══════════════════════════════════════════════════════════════════
# Plot 3 — Epoch trajectory: layer-averaged mean angle per head type
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5))

for head in head_types:
    sub = df_traj[df_traj["head"] == head].groupby("epoch")["mean_angle_deg"].agg(
        mean="mean", std="std").reset_index()
    ax.plot(sub["epoch"], sub["mean"],
            color=colors[head], label=f"{head.capitalize()} — SALTEDORA",
            linewidth=2, markersize=6, **styles["saltedora"])
    ax.fill_between(sub["epoch"],
                    sub["mean"] - sub["std"],
                    sub["mean"] + sub["std"],
                    alpha=0.12, color=colors[head])

    if DORA_AVAILABLE:
        sub_d = df_dora_traj[df_dora_traj["head"] == head].groupby("epoch")["mean_angle_deg"].agg(
            mean="mean", std="std").reset_index()
        ax.plot(sub_d["epoch"], sub_d["mean"],
                color=colors[head], label=f"{head.capitalize()} — DoRA",
                linewidth=2, markersize=6, **styles["dora"])
        ax.fill_between(sub_d["epoch"],
                        sub_d["mean"] - sub_d["std"],
                        sub_d["mean"] + sub_d["std"],
                        alpha=0.08, color=colors[head])

ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Mean Principal Angle (°)\naveraged across all layers", fontsize=12)
ax.set_title("Principal Angle Evolution Across Training\nSALTEDORA r=64 vs Full FT — SST-2",
             fontsize=13, fontweight="bold", pad=TITLE_PAD)
ax.set_xticks(epochs)
ax.set_xticklabels([f"Epoch {e}" for e in epochs])
ax.legend(title="Head · Method", fontsize=9, ncol=2 if DORA_AVAILABLE else 1)
ax.grid(linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "3_epoch_trajectory_by_head.png"), dpi=150, bbox_inches="tight")
plt.close()
print("✓ Plot 3 saved")


# ═══════════════════════════════════════════════════════════════════
# Plot 4 — Epoch trajectory: heatmap [layer × epoch] per head
#          If DoRA available: SALTEDORA left, DoRA right per head
# ═══════════════════════════════════════════════════════════════════
def epoch_mat(df, head):
    return np.array([
        [df[(df["epoch"] == e) & (df["layer_idx"] == l) & (df["head"] == head)
            ]["mean_angle_deg"].values[0]
         for e in epochs]
        for l in layers
    ])

n_cols = 2 if DORA_AVAILABLE else 1
fig, axes = plt.subplots(3, n_cols, figsize=(11 * n_cols, 9), sharex="col", sharey="row")
if n_cols == 1:
    axes = axes.reshape(-1, 1)

fig.suptitle("Principal Angle Trajectory [Layer × Epoch]",
             fontsize=13, fontweight="bold")

for row, head in enumerate(head_types):
    mat_s = epoch_mat(df_traj, head)

    # SALTEDORA column
    ax = axes[row, 0]
    im = ax.imshow(mat_s.T, aspect="auto", cmap="YlOrRd", vmin=0, vmax=50, origin="upper")
    ax.set_yticks(range(len(epochs)))
    ax.set_yticklabels([f"Ep {e}" for e in epochs], fontsize=8)
    ax.set_ylabel(f"{head.capitalize()}", fontsize=10, fontweight="bold",
                  color=colors[head])
    if row == 0:
        ax.set_title("SALTEDORA r=64", fontsize=11, fontweight="bold")
    for j, e in enumerate(epochs):
        for i in range(len(layers)):
            val = mat_s[i, j]
            ax.text(i, j, f"{val:.0f}", ha="center", va="center",
                    fontsize=6.5, color="black" if val < 30 else "white")

    # DoRA column (if available)
    if DORA_AVAILABLE:
        mat_d = epoch_mat(df_dora_traj, head)
        ax2   = axes[row, 1]
        im2   = ax2.imshow(mat_d.T, aspect="auto", cmap="YlOrRd", vmin=0, vmax=50, origin="upper")
        ax2.set_yticks(range(len(epochs)))
        ax2.set_yticklabels([f"Ep {e}" for e in epochs], fontsize=8)
        if row == 0:
            ax2.set_title("DoRA r=64", fontsize=11, fontweight="bold")
        for j, e in enumerate(epochs):
            for i in range(len(layers)):
                val = mat_d[i, j]
                ax2.text(i, j, f"{val:.0f}", ha="center", va="center",
                         fontsize=6.5, color="black" if val < 30 else "white")

for col in range(n_cols):
    axes[-1, col].set_xticks(range(len(layers)))
    axes[-1, col].set_xticklabels([f"L{l}" for l in layers])
    axes[-1, col].set_xlabel("BERT Encoder Layer", fontsize=11)

plt.colorbar(im, ax=axes, label="Mean angle (°)", shrink=0.5, pad=0.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "4_epoch_layer_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close()
print("✓ Plot 4 saved")


# ═══════════════════════════════════════════════════════════════════
# Plot 5 — Deep layers (L8–L11) trajectory per head
# ═══════════════════════════════════════════════════════════════════
deep_layers = [8, 9, 10, 11]
cmap_depth  = plt.cm.plasma

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
fig.suptitle("Deep Layer (L8–L11) Principal Angle Trajectory\nSALTEDORA r=64 vs Full FT — SST-2",
             fontsize=13, fontweight="bold")

for ax, head in zip(axes, head_types):
    for i, l in enumerate(deep_layers):
        col = cmap_depth(i / (len(deep_layers) - 1))
        sub = df_traj[(df_traj["head"] == head) & (df_traj["layer_idx"] == l)].sort_values("epoch")
        ax.plot(sub["epoch"], sub["mean_angle_deg"],
                marker="o", linewidth=2, markersize=5, color=col,
                label=f"L{l} — SALTEDORA", linestyle="-")

        if DORA_AVAILABLE:
            sub_d = df_dora_traj[(df_dora_traj["head"] == head) &
                                 (df_dora_traj["layer_idx"] == l)].sort_values("epoch")
            ax.plot(sub_d["epoch"], sub_d["mean_angle_deg"],
                    marker="s", linewidth=2, markersize=5, color=col,
                    label=f"L{l} — DoRA", linestyle="--", alpha=0.75)

    ax.set_title(head.capitalize(), fontsize=11, fontweight="bold", color=colors[head])
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_xticks(epochs)
    ax.grid(linestyle="--", alpha=0.4)
    ax.legend(fontsize=7, ncol=2 if DORA_AVAILABLE else 1)

axes[0].set_ylabel("Mean Principal Angle (°)", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "5_deep_layer_trajectory.png"), dpi=150, bbox_inches="tight")
plt.close()
print("✓ Plot 5 saved")


# ═══════════════════════════════════════════════════════════════════
# Plot 6 — Epoch 1 vs Final: delta per layer
#          If DoRA available: SALTEDORA and DoRA bars side by side
# ═══════════════════════════════════════════════════════════════════
def compute_delta(df):
    ep1   = df[df["epoch"] == df["epoch"].min()].set_index(["layer_idx", "head"])["mean_angle_deg"]
    epfin = df[df["epoch"] == df["epoch"].max()].set_index(["layer_idx", "head"])["mean_angle_deg"]
    return (epfin - ep1).reset_index().rename(columns={"mean_angle_deg": "delta"})

delta_s = compute_delta(df_traj)
if DORA_AVAILABLE:
    delta_d = compute_delta(df_dora_traj)

fig, ax = plt.subplots(figsize=(13, 4.5))

bar_width = 0.35 if DORA_AVAILABLE else 0.6
x_pos, xlabels = [], []
idx = 0

for head in head_types:
    for l in layers:
        row_s = delta_s[(delta_s["layer_idx"] == l) & (delta_s["head"] == head)]
        d_s   = row_s["delta"].values[0] if len(row_s) else 0

        offset = -bar_width / 2 if DORA_AVAILABLE else 0
        ax.bar(idx + offset, d_s, bar_width,
               color=colors[head], alpha=0.85, edgecolor="white", linewidth=0.5,
               label="SALTEDORA" if (head == "query" and l == 0) else "")

        if DORA_AVAILABLE:
            row_d = delta_d[(delta_d["layer_idx"] == l) & (delta_d["head"] == head)]
            d_d   = row_d["delta"].values[0] if len(row_d) else 0
            ax.bar(idx + bar_width / 2, d_d, bar_width,
                   color=colors[head], alpha=0.45, edgecolor="white", linewidth=0.5,
                   hatch="//",
                   label="DoRA" if (head == "query" and l == 0) else "")

        x_pos.append(idx)
        xlabels.append(f"L{l}")
        idx += 1

ax.axhline(0, color="black", linewidth=0.9)
for sep, lbl in zip([12, 24], ["Key", "Value"]):
    ax.axvline(sep - 0.5, color="gray", linewidth=0.8, linestyle="--")

ymax = max(abs(ax.get_ylim()[0]), abs(ax.get_ylim()[1]))
for pos, lbl, head in [(5.5, "Query", "query"), (17.5, "Key", "key"), (29.5, "Value", "value")]:
    ax.text(pos, ymax * 0.88, lbl, fontsize=10, ha="center",
            color=colors[head], fontweight="bold")

ax.set_xticks(x_pos[::3])
ax.set_xticklabels(xlabels[::3], fontsize=8)
ax.set_ylabel("Δ Mean Angle (°)\nEpoch 5 − Epoch 1", fontsize=11)
ax.set_title("Change in Principal Angle from Epoch 1 → Final\n"
             "Positive = divergence grew, Negative = models converged",
             fontsize=12, fontweight="bold", pad=TITLE_PAD)
ax.grid(axis="y", linestyle="--", alpha=0.4)
if DORA_AVAILABLE:
    ax.legend(["SALTEDORA (solid)", "DoRA (hatched)"], fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "6_epoch1_vs_final_delta.png"), dpi=150, bbox_inches="tight")
plt.close()
print("✓ Plot 6 saved")

print(f"\n✅ All 6 plots saved to {OUT_DIR}")
if not DORA_AVAILABLE:
    print("\n   When DoRA results are ready, place:")
    print(f"     {DORA_DIR}/final_comparison_summary.csv")
    print(f"     {DORA_DIR}/epoch_trajectory_summary.csv")
    print("   Then re-run this script — all plots will automatically include DoRA.")


# ═══════════════════════════════════════════════════════════════════
# Plot 7 — Direct SALTEDORA vs DoRA head-to-head (only if DoRA present)
# ═══════════════════════════════════════════════════════════════════
if DORA_AVAILABLE:
    import matplotlib.gridspec as gridspec

    layers     = list(range(12))
    head_types = ["query", "key", "value"]
    colors     = {"query": "#4C72B0", "key": "#DD8452", "value": "#55A868"}

    # ── 7a: Side-by-side bar chart per layer, averaged across heads ──
    salt_avg = df_final.groupby("layer_idx")["mean_angle_deg"].mean()
    dora_avg = df_dora_final.groupby("layer_idx")["mean_angle_deg"].mean()

    x      = np.arange(len(layers))
    width  = 0.35
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x - width/2, salt_avg.values, width,
           color="#4C72B0", alpha=0.85, label="SALTEDORA r=64", edgecolor="white")
    ax.bar(x + width/2, dora_avg.values,  width,
           color="#DD8452", alpha=0.85, label="DoRA r=64",      edgecolor="white")

    ax.set_xlabel("BERT Encoder Layer", fontsize=12)
    ax.set_ylabel("Mean Principal Angle (°)\naveraged across Q, K, V", fontsize=12)
    ax.set_title("SALTEDORA r=64 vs DoRA r=64 — Subspace Alignment with Full FT\n"
                 "Lower angle = closer to full fine-tuning",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in layers])
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Annotate overall means
    sm = df_final["mean_angle_deg"].mean()
    dm = df_dora_final["mean_angle_deg"].mean()
    ax.axhline(sm, color="#4C72B0", linewidth=1.2, linestyle=":", alpha=0.7)
    ax.axhline(dm, color="#DD8452", linewidth=1.2, linestyle=":", alpha=0.7)
    ax.text(11.6, sm + 0.4, f"μ={sm:.1f}°", color="#4C72B0", fontsize=8, ha="right")
    ax.text(11.6, dm + 0.4, f"μ={dm:.1f}°", color="#DD8452", fontsize=8, ha="right")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "7a_saltedora_vs_dora_bar.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ Plot 7a saved")

    # ── 7b: Per-head line comparison ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.suptitle("SALTEDORA r=64 vs DoRA r=64 — Mean Principal Angle per Head Type\n"
                 "Lower = more aligned with Full Fine-Tuning",
                 fontsize=13, fontweight="bold")

    for ax, head in zip(axes, head_types):
        s = df_final[df_final["head"] == head].sort_values("layer_idx")
        d = df_dora_final[df_dora_final["head"] == head].sort_values("layer_idx")

        ax.plot(s["layer_idx"], s["mean_angle_deg"],
                marker="o", linewidth=2, markersize=6,
                color="#4C72B0", label="SALTEDORA")
        ax.plot(d["layer_idx"], d["mean_angle_deg"],
                marker="s", linewidth=2, markersize=6,
                color="#DD8452", label="DoRA", linestyle="--")

        # Shade gap — red where SALTEDORA is further, green where it's closer
        sv = s["mean_angle_deg"].values
        dv = d["mean_angle_deg"].values
        xi = s["layer_idx"].values
        ax.fill_between(xi, sv, dv,
                        where=(sv > dv), alpha=0.15, color="red",
                        label="SALTEDORA further from FT")
        ax.fill_between(xi, sv, dv,
                        where=(sv <= dv), alpha=0.15, color="green",
                        label="SALTEDORA closer to FT")

        ax.set_title(head.capitalize(), fontsize=12, fontweight="bold",
                     color=colors[head])
        ax.set_xlabel("BERT Encoder Layer", fontsize=10)
        ax.set_xticks(layers)
        ax.set_xticklabels([f"L{l}" for l in layers], fontsize=7)
        ax.grid(linestyle="--", alpha=0.4)
        ax.legend(fontsize=7)

    axes[0].set_ylabel("Mean Principal Angle (°)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "7b_saltedora_vs_dora_per_head.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ Plot 7b saved")

    # ── 7c: Delta plot — SALTEDORA minus DoRA per layer-head ──
    merged = df_final.set_index(["layer_idx", "head"])[["mean_angle_deg"]].rename(
        columns={"mean_angle_deg": "salt"})
    merged["dora"] = df_dora_final.set_index(["layer_idx", "head"])["mean_angle_deg"]
    merged["delta"] = merged["salt"] - merged["dora"]   # positive = SALTEDORA further from FT
    merged = merged.reset_index()

    fig, ax = plt.subplots(figsize=(13, 5))
    x_pos, bar_cols, deltas, xlabels = [], [], [], []
    idx = 0
    for head in head_types:
        for l in layers:
            row = merged[(merged["layer_idx"] == l) & (merged["head"] == head)]
            d   = row["delta"].values[0] if len(row) else 0
            x_pos.append(idx)
            deltas.append(d)
            bar_cols.append(colors[head])
            xlabels.append(f"L{l}")
            idx += 1

    deltas = np.array(deltas)
    ax.bar(x_pos, deltas, color=bar_cols, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=1.0)

    for sep, label, head in zip([12, 24], ["Key", "Value"], ["key", "value"]):
        ax.axvline(sep - 0.5, color="gray", linewidth=0.8, linestyle="--")
    ymax = max(abs(deltas.min()), abs(deltas.max())) * 1.1
    for pos, lbl, head in [(5.5, "Query", "query"), (17.5, "Key", "key"), (29.5, "Value", "value")]:
        ax.text(pos, ymax * 0.88, lbl, fontsize=10, ha="center",
                color=colors[head], fontweight="bold")

    ax.set_xticks(x_pos[::3])
    ax.set_xticklabels(xlabels[::3], fontsize=8)
    ax.set_ylim(-ymax, ymax)
    ax.set_ylabel("Δ Mean Angle (°)\nSALTEDORA − DoRA", fontsize=11)
    ax.set_title("Subspace Alignment Gap: SALTEDORA vs DoRA\n"
                 "Positive = SALTEDORA is further from Full FT than DoRA",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    pct_worse = (deltas > 0).mean() * 100
    ax.text(0.99, 0.97,
            f"SALTEDORA is further from FT\nin {pct_worse:.0f}% of layer-head pairs\nOverall Δ = {deltas.mean():+.2f}°",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.85))

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "7c_saltedora_vs_dora_delta.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ Plot 7c saved")

    print(f"\n  SALTEDORA overall mean: {df_final['mean_angle_deg'].mean():.2f}°")
    print(f"  DoRA overall mean:      {df_dora_final['mean_angle_deg'].mean():.2f}°")
    print(f"  Gap:                    {df_final['mean_angle_deg'].mean() - df_dora_final['mean_angle_deg'].mean():+.2f}°")


# ═══════════════════════════════════════════════════════════════════
# Plot 8 — Per-layer epoch trajectory: 4×3 grid, one subplot per layer
#          Each subplot shows Q/K/V mean angle across epochs
#          SALTEDORA solid, DoRA dashed (if available)
# ═══════════════════════════════════════════════════════════════════
layers     = list(range(12))
head_types = ["query", "key", "value"]
colors     = {"query": "#4C72B0", "key": "#DD8452", "value": "#55A868"}
epochs     = sorted(df_traj["epoch"].unique())

fig, axes = plt.subplots(4, 3, figsize=(16, 14), sharex=True)
fig.suptitle(
    "Principal Angle Epoch Trajectory — Per Layer\n"
    "SALTEDORA r=64 (solid) vs Full FT" +
    (" | DoRA r=64 (dashed) vs Full FT" if DORA_AVAILABLE else ""),
    fontsize=14, fontweight="bold"
)

for l in layers:
    row, col = divmod(l, 3)
    ax = axes[row][col]

    for head in head_types:
        # SALTEDORA
        sub_s = df_traj[
            (df_traj["layer_idx"] == l) & (df_traj["head"] == head)
        ].sort_values("epoch")
        ax.plot(sub_s["epoch"], sub_s["mean_angle_deg"],
                marker="o", linewidth=2, markersize=4,
                color=colors[head], linestyle="-",
                label=f"{head[0].upper()} SALT")

        # DoRA
        if DORA_AVAILABLE:
            sub_d = df_dora_traj[
                (df_dora_traj["layer_idx"] == l) & (df_dora_traj["head"] == head)
            ].sort_values("epoch")
            ax.plot(sub_d["epoch"], sub_d["mean_angle_deg"],
                    marker="s", linewidth=2, markersize=4,
                    color=colors[head], linestyle="--", alpha=0.75,
                    label=f"{head[0].upper()} DoRA")

    ax.set_title(f"Layer {l}", fontsize=10, fontweight="bold")
    ax.set_xticks(epochs)
    ax.grid(linestyle="--", alpha=0.35)
    ax.set_ylim(bottom=0)

    # Only show y-label on leftmost column
    if col == 0:
        ax.set_ylabel("Mean angle (°)", fontsize=9)
    # Only show x-label on bottom row
    if row == 3:
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_xticklabels([f"Ep{e}" for e in epochs], fontsize=8)

# Shared legend in top-right corner outside the grid
handles, labels = axes[0][0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right", fontsize=9,
           ncol=2 if DORA_AVAILABLE else 1,
           title="Head · Method", title_fontsize=9,
           bbox_to_anchor=(1.0, 0.98))

plt.tight_layout(rect=[0, 0, 0.88 if DORA_AVAILABLE else 1.0, 0.95])
plt.savefig(os.path.join(OUT_DIR, "8_per_layer_epoch_trajectory.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("✓ Plot 8 saved")


# ═══════════════════════════════════════════════════════════════════
# Plot 9 — Per-layer final comparison bar chart: Q/K/V grouped bars
#          SALTEDORA vs DoRA (if available), one subplot per layer
# ═══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(4, 3, figsize=(16, 13), sharey=False)
fig.suptitle(
    "Final Checkpoint Principal Angles — Per Layer\n"
    "SALTEDORA r=64" + (" vs DoRA r=64" if DORA_AVAILABLE else "") + " vs Full FT",
    fontsize=14, fontweight="bold"
)

bar_width = 0.25 if DORA_AVAILABLE else 0.45
x_heads   = np.arange(len(head_types))

for l in layers:
    row, col = divmod(l, 3)
    ax = axes[row][col]

    s_vals = [
        df_final[(df_final["layer_idx"] == l) & (df_final["head"] == h)
                 ]["mean_angle_deg"].values[0]
        for h in head_types
    ]

    if DORA_AVAILABLE:
        d_vals = [
            df_dora_final[(df_dora_final["layer_idx"] == l) & (df_dora_final["head"] == h)
                         ]["mean_angle_deg"].values[0]
            for h in head_types
        ]
        ax.bar(x_heads - bar_width/2, s_vals, bar_width,
               color=[colors[h] for h in head_types],
               alpha=0.85, edgecolor="white", label="SALTEDORA")
        ax.bar(x_heads + bar_width/2, d_vals, bar_width,
               color=[colors[h] for h in head_types],
               alpha=0.4, edgecolor="white", hatch="//", label="DoRA")
    else:
        ax.bar(x_heads, s_vals, bar_width,
               color=[colors[h] for h in head_types],
               alpha=0.85, edgecolor="white")

    ax.set_title(f"Layer {l}", fontsize=10, fontweight="bold")
    ax.set_xticks(x_heads)
    ax.set_xticklabels(["Q", "K", "V"], fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_ylim(bottom=0)

    if col == 0:
        ax.set_ylabel("Mean angle (°)", fontsize=9)

handles2 = [
    plt.Rectangle((0,0),1,1, color="#4C72B0", alpha=0.85, label="Q SALTEDORA"),
    plt.Rectangle((0,0),1,1, color="#DD8452", alpha=0.85, label="K SALTEDORA"),
    plt.Rectangle((0,0),1,1, color="#55A868", alpha=0.85, label="V SALTEDORA"),
]
if DORA_AVAILABLE:
    handles2 += [
        plt.Rectangle((0,0),1,1, color="#4C72B0", alpha=0.4, hatch="//", label="Q DoRA"),
        plt.Rectangle((0,0),1,1, color="#DD8452", alpha=0.4, hatch="//", label="K DoRA"),
        plt.Rectangle((0,0),1,1, color="#55A868", alpha=0.4, hatch="//", label="V DoRA"),
    ]
fig.legend(handles=handles2, loc="upper right", fontsize=9,
           ncol=2 if DORA_AVAILABLE else 1,
           title="Head · Method", title_fontsize=9,
           bbox_to_anchor=(1.0, 0.98))

plt.tight_layout(rect=[0, 0, 0.88 if DORA_AVAILABLE else 1.0, 0.95])
plt.savefig(os.path.join(OUT_DIR, "9_per_layer_final_bar.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("✓ Plot 9 saved")
