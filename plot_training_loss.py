import matplotlib.pyplot as plt
import wandb

api = wandb.Api()
entity = "robert-vdklis"
project = "brain"
run_names = ["CancerFoundation_New", "CancerFoundation_Old"]

fig, ax = plt.subplots(figsize=(10, 6))

for run_name in run_names:
    runs = api.runs(f"{entity}/{project}", filters={"display_name": run_name})
    if not runs:
        print(f"Warning: no run found with name '{run_name}'")
        continue
    run = runs[0]
    history = run.history(keys=["train/mse", "_timestamp", "_runtime"], pandas=True)
    if history.empty or "train/mse" not in history.columns:
        print(f"Warning: no 'train/mse' data for run '{run_name}'")
        continue

    df = history.dropna(subset=["train/mse"]).copy()
    df["minutes"] = df["_runtime"] / 60.0

    ax.plot(df["minutes"], df["train/mse"], label=run_name, alpha=0.8, linewidth=1.2)

ax.set_xlabel("Time (minutes)", fontsize=13)
ax.set_ylabel("train/mse", fontsize=13)
ax.set_title("Training Loss", fontsize=15)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig("training_loss.png", dpi=150)
print("Saved to training_loss.png")
