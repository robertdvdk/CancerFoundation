import matplotlib.pyplot as plt

labels = ["CancerFoundation_Old", "CancerFoundation_New"]
throughput = [247.3, 1920]
colors = ["#ff9f43", "#54a0ff"]

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(labels, throughput, color=colors, width=0.5, edgecolor="white", linewidth=1.5)

for bar, val in zip(bars, throughput):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 30,
        f"{val:.0f}",
        ha="center",
        va="bottom",
        fontsize=14,
        fontweight="bold",
    )

ax.set_ylabel("Samples / second", fontsize=13)
ax.set_title("Training Throughput", fontsize=15)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig("throughput.png", dpi=150)
print("Saved to throughput.png")
