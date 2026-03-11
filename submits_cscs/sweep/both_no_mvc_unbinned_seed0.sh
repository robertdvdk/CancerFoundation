#!/bin/bash -l
#SBATCH --job-name=sweep_both_no_mvc_unbinned_seed0
#SBATCH --output=./%x_%j.out
#SBATCH --time=06:00:00
#SBATCH --partition=normal
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --account=a132

# Both tasks, no MVC, unbinned (seed=0)

set -x
ulimit -c 0

SAVE_DIR="./save/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "$SAVE_DIR"
TRAIN_DIR="/iopsstor/scratch/cscs/rvander/DATA/brain_processed/processed_data/train"

srun -ul --environment=./bionemo_bristen.toml bash -c "
    MASTER_ADDR=\$(scontrol show hostnames \$SLURM_JOB_NODELIST | head -n 1) \
    MASTER_PORT=29500 \
    RANK=\${SLURM_PROCID} \
    LOCAL_RANK=\${SLURM_LOCALID} \
    WORLD_SIZE=\${SLURM_NTASKS} \
    python pretrain.py \
    --gpus 4 \
    --save-dir \"$SAVE_DIR\" \
    --train-path \"$TRAIN_DIR\" \
    --wandb \"brain\" \
    --wandb-name \"${SLURM_JOB_NAME}_${SLURM_JOB_ID}\" \
    --max-seq-len 1200 \
    --batch-size 64 \
    --nlayers 6 \
    --nheads 8 \
    --embsize 128 \
    --d-hid 256 \
    --epochs 51 \
    --lr 0.0001 \
    --warmup-ratio-or-step 1 \
    --trunc-by-sample \
    --loss mse \
    --balance-primary technology \
    --zero-percentages 0.2 0.4 0.6 \
    --strategy ddp \
    --precision bf16-mixed \
    --compile \
    --log-interval 100 \
    --conditions technology \
    --where-condition end \
    --num-workers 24 \
    --training-tasks both \
    --gen-method quick \
    --input-emb-style theirs \
    --their-init-weights \
    --eval-every-n-epochs 5 \
    --eval-datasets /iopsstor/scratch/cscs/rvander/DATA/brain_processed/neftel_ss2.h5ad /iopsstor/scratch/cscs/rvander/DATA/brain_processed/ji_skin.h5ad /iopsstor/scratch/cscs/rvander/DATA/brain_processed/kim_lung.h5ad \
    --input-style log1p \
    --seed 0
"

if [ -d "./lightning_logs/version_${SLURM_JOB_ID}" ]; then
    mv "./lightning_logs/version_${SLURM_JOB_ID}" "$SAVE_DIR/lightning_log"
fi
cp "$TRAIN_DIR/vocab.json" "$SAVE_DIR/vocab.json" 2>/dev/null || true
cp "$0" "$SAVE_DIR/run_script.sh"
mv "./${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" "$SAVE_DIR/slurm.out" 2>/dev/null || true
echo "Job finished. Outputs and logs are in $SAVE_DIR"
