#!/bin/bash -l
#SBATCH --job-name=evaluate_gene_perturbation_1
#SBATCH --output=./%x_%j.out
#SBATCH --time=01:00:00
#SBATCH --partition=normal
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --account=a132

set -x

ulimit -c 0

srun -ul --environment=../bionemo_clariden.toml bash -c "
    pip install cell-gears==0.0.2
    MASTER_ADDR=\$(scontrol show hostnames \$SLURM_JOB_NODELIST | head -n 1) \
    MASTER_PORT=29500 \
    RANK=\${SLURM_PROCID} \
    LOCAL_RANK=\${SLURM_LOCALID} \
    WORLD_SIZE=\${SLURM_NTASKS} \
    python run_perturbation.py \
    --checkpoint ../save/train_medium_condtech_my_init_weights_4gpu_lrx2_955577/epoch_epoch=14.ckpt \
    --data-name adamson \
    --epochs 15 \
    --lr 1e-4 \
    --batch-size 16 \
    --eval-batch-size 16 \
    --save-dir ./my_results/myinit_lrx2_seed1 \
    --seed 1
"