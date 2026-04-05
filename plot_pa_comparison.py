"""
plot_pa_r64.py
==============
Plots final principal-angle analysis for SALTEDORA r=64 (v7)
across all BERT QKV layers.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# Data  (layer, mean_angle_deg, max_angle_deg)
# ─────────────────────────────────────────────
RAW_V7 = """bert.encoder.layer.0.attention.self.query,7.180601596832275,88.70610046386719
bert.encoder.layer.0.attention.self.key,11.175009727478027,88.04399108886719
bert.encoder.layer.0.attention.self.value,24.604942321777344,89.73992919921875
bert.encoder.layer.1.attention.self.query,3.820260763168335,12.66716194152832
bert.encoder.layer.1.attention.self.key,6.676321983337402,85.23497772216797
bert.encoder.layer.1.attention.self.value,10.755348205566406,89.94770812988281
bert.encoder.layer.2.attention.self.query,5.973355293273926,83.69325256347656
bert.encoder.layer.2.attention.self.key,11.460210800170898,89.34632110595703
bert.encoder.layer.2.attention.self.value,10.831398963928223,85.53697204589844
bert.encoder.layer.3.attention.self.query,6.627931118011475,87.98113250732422
bert.encoder.layer.3.attention.self.key,14.373397827148438,89.01393127441406
bert.encoder.layer.3.attention.self.value,20.154455184936523,89.61779022216797
bert.encoder.layer.4.attention.self.query,11.928850173950195,88.01451110839844
bert.encoder.layer.4.attention.self.key,14.357070922851562,89.73233032226562
bert.encoder.layer.4.attention.self.value,22.586917877197266,89.87248992919922
bert.encoder.layer.5.attention.self.query,15.265589714050293,89.3873062133789
bert.encoder.layer.5.attention.self.key,14.849891662597656,89.99269104003906
bert.encoder.layer.5.attention.self.value,18.003021240234375,89.74862670898438
bert.encoder.layer.6.attention.self.query,9.484840393066406,88.95098114013672
bert.encoder.layer.6.attention.self.key,19.845380783081055,89.70133209228516
bert.encoder.layer.6.attention.self.value,19.61655616760254,89.4858169555664
bert.encoder.layer.7.attention.self.query,11.774726867675781,89.59073638916016
bert.encoder.layer.7.attention.self.key,11.338708877563477,89.33402252197266
bert.encoder.layer.7.attention.self.value,32.42383575439453,89.63794708251953
bert.encoder.layer.8.attention.self.query,20.127838134765625,89.87530517578125
bert.encoder.layer.8.attention.self.key,13.855674743652344,88.36380767822266
bert.encoder.layer.8.attention.self.value,28.74332618713379,89.70783233642578
bert.encoder.layer.9.attention.self.query,17.128402709960938,89.5728759765625
bert.encoder.layer.9.attention.self.key,9.163787841796875,89.56397247314453
bert.encoder.layer.9.attention.self.value,43.302696228027344,89.88854217529297
bert.encoder.layer.10.attention.self.query,24.21025848388672,89.85807800292969
bert.encoder.layer.10.attention.self.key,30.91808319091797,89.87255859375
bert.encoder.layer.10.attention.self.value,41.33984375,89.91885375976562
bert.encoder.layer.11.attention.self.query,13.504190444946289,89.82917022705078
bert.encoder.layer.11.attention.self.key,19.61849594116211,89.94705200195312
bert.encoder.layer.11.attention.self.value,44.91111373901367,89.96045684814453"""


def parse_raw(raw: str):
    rows = {}
    for line in raw.strip().splitlines():
        parts = line.split(",")
        name  = parts[0].strip()
        mean  = float(parts[1])
        max_  = float(parts[2])
        layer = int(name.split(".layer.")[1].split(".")[0])
        head  = name.split("self.")[1]
        rows[name] = {"layer": layer, "head": head, "mean": mean, "max": max_}
    return rows

d7 = parse_raw(RAW_V7)

layers     = list(range(12))
head_types = ["query", "key", "value"]
colors     = {"query": "#4C72B0", "key": "#DD8452", "value": "#55A868"}


# ─────────────────────────────────────────────
# Figure 1: Line plot — mean angle per head
# across encoder depth
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))

for head in head_types:
    means = [d7[f"bert.encoder.layer.{l}.attention.self.{head}"]["mean"] for l in layers]
    ax.plot(layers, means, marker="o", linewidth=2, markersize=6,
            color=colors[head], label=head.capitalize())

ax.set_xlabel("BERT Encoder Layer", fontsize=11)
ax.set_ylabel("Mean principal angle (°)", fontsize=11)
ax.set_title(
    "Principal Angles — SALTEDORA r=64 vs Full Fine-Tuning\n"
    "Final Checkpoint · SST-2",
    fontsize=12, fontweight="bold"
)
ax.set_xticks(layers)
ax.set_xticklabels([f"L{l}" for l in layers])
ax.set_ylim(0, 52)
ax.legend(title="Head type", fontsize=10)
ax.grid(linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("pa_r64_line.png", dpi=150, bbox_inches="tight")
print("✓ Saved: pa_r64_line.png")
plt.close()


# ─────────────────────────────────────────────
# Figure 2: Grouped bar chart — Q / K / V
# side by side per layer
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))

x      = np.arange(len(layers))
width  = 0.25
offsets = [-width, 0, width]

for head, offset in zip(head_types, offsets):
    vals = [d7[f"bert.encoder.layer.{l}.attention.self.{head}"]["mean"] for l in layers]
    ax.bar(x + offset, vals, width, color=colors[head],
           label=head.capitalize(), alpha=0.85, edgecolor="white", linewidth=0.5)

ax.set_xlabel("BERT Encoder Layer", fontsize=11)
ax.set_ylabel("Mean principal angle (°)", fontsize=11)
ax.set_title(
    "Mean Principal Angles by Layer and Head Type — r=64 · SST-2",
    fontsize=12, fontweight="bold"
)
ax.set_xticks(x)
ax.set_xticklabels([f"L{l}" for l in layers])
ax.legend(title="Head type", fontsize=10)
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("pa_r64_grouped_bar.png", dpi=150, bbox_inches="tight")
print("✓ Saved: pa_r64_grouped_bar.png")
plt.close()


# ─────────────────────────────────────────────
# Figure 3: Heatmap — [head × layer]
# ─────────────────────────────────────────────
mat = np.array([
    [d7[f"bert.encoder.layer.{l}.attention.self.{h}"]["mean"] for l in layers]
    for h in head_types
])

fig, ax = plt.subplots(figsize=(11, 3.5))
im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=50)
plt.colorbar(im, ax=ax, label="Mean angle (°)")
ax.set_xticks(range(len(layers)))
ax.set_xticklabels([f"L{l}" for l in layers])
ax.set_yticks(range(len(head_types)))
ax.set_yticklabels([h.capitalize() for h in head_types], fontsize=11)
ax.set_xlabel("BERT Encoder Layer", fontsize=11)
ax.set_title(
    "Principal Angle Heatmap — SALTEDORA r=64 vs Full FT · SST-2",
    fontsize=12, fontweight="bold"
)

# Annotate cells with values
for i, h in enumerate(head_types):
    for j, l in enumerate(layers):
        val = mat[i, j]
        ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                fontsize=7, color="black" if val < 30 else "white")

plt.tight_layout()
plt.savefig("pa_r64_heatmap.png", dpi=150, bbox_inches="tight")
print("✓ Saved: pa_r64_heatmap.png")
plt.close()

print("\nAll plots saved.")
