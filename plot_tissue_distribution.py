#!/usr/bin/env python3
"""Recreate the tissue distribution donut chart with larger fonts."""

import json

import matplotlib.pyplot as plt
import numpy as np

# Load data
with open("tissue_distribution.json") as f:
    data = json.load(f)

dataset = data["per_dataset"]["dataset_pretraining"]
tissues = dataset["tissues"]
total_cells = dataset["total_cells"]

# Separate "rare" tissues (below ~3% threshold) from main tissues
rare_threshold = 3.0
main_tissues = {}
rare_tissues = {}

for tissue, info in tissues.items():
    if tissue == "Other":
        main_tissues[tissue] = info
    elif info["percent"] < rare_threshold:
        rare_tissues[tissue] = info
    else:
        main_tissues[tissue] = info

# Add "Rare tissues" as a combined slice
rare_total_count = sum(v["count"] for v in rare_tissues.values())
rare_total_pct = sum(v["percent"] for v in rare_tissues.values())
main_tissues["Rare tissues"] = {"count": rare_total_count, "percent": round(rare_total_pct, 1)}

# Sort main tissues by count descending, but put "Rare tissues" second-to-last and "Other" last
regular = [(k, v) for k, v in main_tissues.items() if k not in ("Rare tissues", "Other")]
regular.sort(key=lambda x: x[1]["count"], reverse=True)
main_sorted = regular + [
    ("Rare tissues", main_tissues["Rare tissues"]),
    ("Other", main_tissues["Other"]),
]
labels = [t[0] for t in main_sorted]
sizes = [t[1]["percent"] for t in main_sorted]
counts = [t[1]["count"] for t in main_sorted]

# Sort rare tissues by count descending
rare_sorted = sorted(rare_tissues.items(), key=lambda x: x[1]["count"], reverse=True)

# Colors by tissue name
color_map = {
    "Brain": "#1F77B4",
    "Breast": "#FF7F0F",
    "Ovarian": "#2BA02B",
    "Lung": "#D62727",
    "Pancreas": "#9467BD",
    "Skin": "#8C564A",
    "Hematologic": "#E377C1",
    "Neuroendocrine": "#7F7F7F",
    "Kidney": "#DBDB8D",
    "Rare tissues": "#8CA1BF",
    "Other": "#C7C7C7",
}
colors = [color_map[lab] for lab in labels]

# Font scaling factor
FONT_SCALE = 1.65

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9), gridspec_kw={"width_ratios": [1.2, 1]})

# --- Donut chart ---
wedges, texts, autotexts = ax1.pie(
    sizes,
    labels=labels,
    autopct=lambda pct: f"{pct:.1f}%",
    startangle=90,
    colors=colors,
    pctdistance=0.80,
    labeldistance=1.12,
    wedgeprops=dict(width=0.4, edgecolor="white", linewidth=1.5),
)

for t in texts:
    t.set_fontsize(11 * FONT_SCALE)
    t.set_fontweight("bold")
for t in autotexts:
    t.set_fontsize(8.5 * FONT_SCALE)
    t.set_fontweight("bold")
    t.set_color("white")

ax1.set_title(
    f"Tissue Distribution — Pretraining Dataset\n({total_cells:,} cells)",
    fontsize=13 * FONT_SCALE,
    fontweight="bold",
    pad=20,
)

# --- Rare tissues bar chart ---
rare_labels = [t[0] for t in rare_sorted]
rare_counts = [t[1]["count"] for t in rare_sorted]
rare_pcts = [t[1]["percent"] for t in rare_sorted]

bar_colors = plt.cm.Blues(np.linspace(0.3, 0.7, len(rare_labels)))[::-1]
bars = ax2.barh(rare_labels, rare_counts, color=bar_colors, edgecolor="white", height=0.6)

# Add count and percent labels on bars
for bar, count, pct in zip(bars, rare_counts, rare_pcts):
    ax2.text(
        bar.get_width() + 1500,
        bar.get_y() + bar.get_height() / 2,
        f"{count:,} ({pct}%)",
        va="center",
        fontsize=10 * FONT_SCALE,
        fontweight="bold",
    )

ax2.set_title(
    f'"Rare Tissues" Breakdown\n({rare_total_count:,} cells — {rare_total_pct:.1f}%)',
    fontsize=12 * FONT_SCALE,
    fontweight="bold",
    pad=15,
)
ax2.set_xlabel("Cells", fontsize=10 * FONT_SCALE, fontweight="bold")
ax2.tick_params(axis="both", labelsize=10 * FONT_SCALE)
for label in ax2.get_xticklabels() + ax2.get_yticklabels():
    label.set_fontweight("bold")
ax2.set_xlim(0, max(rare_counts) * 1.45)
ax2.invert_yaxis()
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

# Footnote about "Other"
fig.text(
    0.02,
    0.01,
    '*"Other" (180,147 cells) includes: Bladder (58,613), Brain-Met (48,897), '
    "ESCC (18,570), Uterine (22,414), Mesothelioma (10,500)...",
    fontsize=7 * FONT_SCALE,
    color="gray",
    fontweight="bold",
    ha="left",
)

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig(
    "tissue_distribution_donut_large_font.png", dpi=200, bbox_inches="tight", facecolor="white"
)
plt.close()
print("Saved tissue_distribution_donut_large_font.png")
