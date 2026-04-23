import matplotlib.pyplot as plt
import wandb

api = wandb.Api()
entity = "robert-vdklis"
project = "brain"

run_ids = {
    "sweep_baseline_seed0_63390": "ajwhwhyb",
    "sweep_baseline_seed42_63391": "1gsoinic",
    "sweep_baseline_unbinned_seed0_63398": "x5zl12bi",
    "sweep_baseline_unbinned_seed42_63399": "f4esj27a",
}

short_labels = {
    "sweep_baseline_seed0_63390": "baseline (seed 0)",
    "sweep_baseline_seed42_63391": "baseline (seed 42)",
    "sweep_baseline_unbinned_seed0_63398": "baseline unbinned (seed 0)",
    "sweep_baseline_unbinned_seed42_63399": "baseline unbinned (seed 42)",
}

metrics = [
    (
        "celltype_probe/f1_macro",
        "Cell typing on Peripheral blood mononuclear cells",
        "F1 macro score",
    ),
    (
        "scib/neftel_ss2_hvg/Total",
        "Neftel Glioblastoma batch integration score",
        "scIB total score",
    ),
    (
        "scib/neftel_ss2_hvg/Batch correction",
        "Neftel Glioblastoma batch correction score",
        "scIB batch correction score",
    ),
    (
        "scib/neftel_ss2_hvg/Bio conservation",
        "Neftel Glioblastoma bio conservation score",
        "scIB bio conservation score",
    ),
]

# Fetch all run data
run_data = {}
for run_name, run_id in run_ids.items():
    run = api.run(f"{entity}/{project}/{run_id}")
    history = run.history(pandas=True, samples=5000)
    run_data[run_name] = history
    print(f"Fetched {len(history)} rows for {run_name}")

colors = ["#54a0ff", "#ff6b6b", "#1dd1a1", "#ff9f43"]
run_names = list(run_ids.keys())

for metric_key, metric_title, ylabel in metrics:
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, run_name in enumerate(run_names):
        if run_name not in run_data:
            continue
        df = run_data[run_name].dropna(subset=[metric_key]).copy()
        if df.empty:
            print(f"Warning: no '{metric_key}' data for {run_name}")
            continue
        ax.plot(
            df["epoch"],
            df[metric_key],
            label=short_labels[run_name],
            color=colors[i],
            alpha=0.85,
            linewidth=1.5,
        )

    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(metric_title, fontsize=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    fname = f"sweep_{metric_key.replace('/', '_').replace(' ', '_')}.png"
    fig.savefig(fname, dpi=150)
    print(f"Saved {fname}")
    plt.close(fig)
