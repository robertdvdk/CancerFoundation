#!/usr/bin/env python3
"""Generate SLURM submission scripts for hyperparameter sweep on bristen.

Writes individual .sh files to submits_cscs/sweep/ and a submit_all.sh launcher.

Usage:
    python submits_cscs/generate_sweep.py

    # Review:       ls submits_cscs/sweep/
    # Submit all:   bash submits_cscs/sweep/submit_all.sh
    # Submit one:   sbatch submits_cscs/sweep/baseline.sh

To add experiments, append to the EXPERIMENTS list below. Each entry is:
    (name, description, {overrides from BASELINE})

Set a value to None to remove a baseline key (e.g. gen_method=None for pcpt-only).

Additional axes to consider:
    - Mask ratio:   {"mask_ratio": 0.15}  or  {"mask_ratio": 0.6}
    - Dropout:      {"dropout": 0.1}
    - Batch size:   {"batch_size": 64, "lr": 0.0004}  (linear LR scaling)
    - Evaluation:   {"eval_every_n_epochs": 5}
"""

import stat
from pathlib import Path

# ============================================================
# Cluster configuration
# ============================================================

OUT_DIR = Path("submits_cscs/sweep")
ENV_FILE = "./bionemo_bristen.toml"
TRAIN_DIR = "/capstor/scratch/cscs/rvander/DATA/brain_processed/processed_data/train"
WANDB_PROJECT = "brain"
GPUS = 4
CPUS_PER_TASK = 32
WALL_TIME = "06:00:00"
ACCOUNT = "a132"

# ============================================================
# Baseline config (shared across all experiments)
# ============================================================

BASELINE = {
    "max_seq_len": 1200,
    "batch_size": 64,
    "nlayers": 6,
    "nheads": 8,
    "embsize": 128,
    "d_hid": 256,
    "epochs": 50,
    "lr": 0.0001,
    "warmup_ratio_or_step": 1,
    "trunc_by_sample": True,
    "loss": "mse",
    "balance_primary": "technology",
    "zero_percentages": [0.2, 0.4, 0.6],
    "strategy": "ddp",
    "seed": 0,
    "precision": "bf16-mixed",
    "compile": True,
    "log_interval": 100,
    "conditions": "technology",
    "where_condition": "end",
    "num_workers": 24,
    "training_tasks": "both",
    "gen_method": "quick",
    "input_emb_style": "theirs",
    "do_mvc": True,
    "their_init_weights": True,
    "eval_every_n_epochs": 5,
    "eval_datasets": [
        "/capstor/scratch/cscs/rvander/DATA/brain_processed/neftel_ss2.h5ad",
        "/capstor/scratch/cscs/rvander/DATA/brain_processed/ji_skin.h5ad",
        "/capstor/scratch/cscs/rvander/DATA/brain_processed/kim_lung.h5ad",
    ],
}

# ============================================================
# Experiments: (name, description, overrides)
# ============================================================

EXPERIMENTS = [
    # 2x2 grid: training_tasks (both/pcpt) x do_mvc (True/False)
    ("baseline", "Both tasks + MVC (default config)", {}),
    ("both_no_mvc", "Both tasks, no MVC", {"do_mvc": False}),
    (
        "pcpt_mvc",
        "Perception only + MVC",
        {"training_tasks": "pcpt", "gen_method": None},
    ),
    (
        "pcpt_no_mvc",
        "Perception only, no MVC",
        {"training_tasks": "pcpt", "gen_method": None, "do_mvc": False},
    ),
]

# ============================================================
# Template (@@PLACEHOLDER@@ for Python, $ for shell)
# ============================================================

# Using @@...@@ avoids conflicts with both shell ${} and Python {}
TEMPLATE = r"""#!/bin/bash -l
#SBATCH --job-name=sweep_@@NAME@@
#SBATCH --output=./%x_%j.out
#SBATCH --time=@@TIME@@
#SBATCH --partition=normal
#SBATCH --ntasks-per-node=@@GPUS@@
#SBATCH --gres=gpu:@@GPUS@@
#SBATCH --cpus-per-task=@@CPUS@@
#SBATCH --account=@@ACCOUNT@@

# @@DESC@@

set -x
ulimit -c 0

SAVE_DIR="./save/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "$SAVE_DIR"
TRAIN_DIR="@@TRAIN_DIR@@"

srun -ul --environment=@@ENV@@ bash -c "
    MASTER_ADDR=\$(scontrol show hostnames \$SLURM_JOB_NODELIST | head -n 1) \
    MASTER_PORT=29500 \
    RANK=\${SLURM_PROCID} \
    LOCAL_RANK=\${SLURM_LOCALID} \
    WORLD_SIZE=\${SLURM_NTASKS} \
    python pretrain.py \
    --gpus @@GPUS@@ \
    --save-dir \"$SAVE_DIR\" \
    --train-path \"$TRAIN_DIR\" \
    --wandb \"@@WANDB@@\" \
    --wandb-name \"${SLURM_JOB_NAME}_${SLURM_JOB_ID}\" \
@@ARGS@@
"

if [ -d "./lightning_logs/version_${SLURM_JOB_ID}" ]; then
    mv "./lightning_logs/version_${SLURM_JOB_ID}" "$SAVE_DIR/lightning_log"
fi
cp "$TRAIN_DIR/vocab.json" "$SAVE_DIR/vocab.json" 2>/dev/null || true
cp "$0" "$SAVE_DIR/run_script.sh"
mv "./${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" "$SAVE_DIR/slurm.out" 2>/dev/null || true
echo "Job finished. Outputs and logs are in $SAVE_DIR"
"""


def format_args(config: dict) -> str:
    """Format config dict into shell argument lines for inside bash -c."""
    lines = []
    for key, value in config.items():
        if value is None:
            continue
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                lines.append(f"    {flag}")
        elif isinstance(value, (list, tuple)):
            vals = " ".join(str(v) for v in value)
            lines.append(f"    {flag} {vals}")
        else:
            lines.append(f"    {flag} {value}")
    return " \\\n".join(lines)


def make_config(overrides: dict) -> dict:
    """Merge overrides into baseline. None values remove the key."""
    config = dict(BASELINE)
    for key, value in overrides.items():
        if value is None:
            config.pop(key, None)
        else:
            config[key] = value
    return config


def render(name: str, desc: str, args_str: str) -> str:
    """Fill the template with @@PLACEHOLDER@@ replacements."""
    return (
        TEMPLATE.replace("@@NAME@@", name)
        .replace("@@TIME@@", WALL_TIME)
        .replace("@@GPUS@@", str(GPUS))
        .replace("@@CPUS@@", str(CPUS_PER_TASK))
        .replace("@@ACCOUNT@@", ACCOUNT)
        .replace("@@DESC@@", desc)
        .replace("@@TRAIN_DIR@@", TRAIN_DIR)
        .replace("@@ENV@@", ENV_FILE)
        .replace("@@WANDB@@", WANDB_PROJECT)
        .replace("@@ARGS@@", args_str)
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scripts = []

    print(f"Generating {len(EXPERIMENTS)} experiment scripts in {OUT_DIR}/\n")

    for name, desc, overrides in EXPERIMENTS:
        config = make_config(overrides)
        args_str = format_args(config)
        script = render(name, desc, args_str)

        path = OUT_DIR / f"{name}.sh"
        path.write_text(script.lstrip("\n"))
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
        scripts.append(path)

        # Show what changed from baseline
        delta = ", ".join(f"{k}={v}" for k, v in overrides.items()) or "(baseline)"
        print(f"  {path.name:<30s} {delta}")

    # Generate submit_all.sh
    submit_all = OUT_DIR / "submit_all.sh"
    lines = [
        "#!/bin/bash",
        f"# Submit all {len(scripts)} sweep experiments on bristen",
        "set -e",
        "",
    ]
    for path in scripts:
        lines.append(f'echo "Submitting {path.stem}..." && sbatch {path}')
    lines.append("")
    lines.append(f'echo "\\nSubmitted {len(scripts)} jobs."')
    submit_all.write_text("\n".join(lines) + "\n")
    submit_all.chmod(submit_all.stat().st_mode | stat.S_IEXEC)

    print(f"\n{'='*50}")
    print(f"Generated {len(scripts)} scripts + submit_all.sh")
    print(f"  Submit all:  bash {submit_all}")
    print(f"  Submit one:  sbatch {OUT_DIR}/<name>.sh")
    print(f"  W&B project: {WANDB_PROJECT}")


if __name__ == "__main__":
    main()
