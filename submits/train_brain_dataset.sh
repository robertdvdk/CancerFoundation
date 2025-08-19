#!/bin/bash -l
#SBATCH --job-name=train_brain_dataset
#SBATCH --output=./%x_%j.out
#SBATCH --time=00:10:00
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=15

set -e

SAVE_DIR="./save/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
TRAIN_DIR="/cluster/dataset/boeva/rvander/DATA/processed_data/train"
mkdir -p "$SAVE_DIR"


srun singularity run \
    --pwd /cluster/work/boeva/rvander/CancerFoundation \
    --bind /cluster/work/boeva/rvander/CancerFoundation:/cluster/work/boeva/rvander/CancerFoundation \
    --bind /cluster/dataset/boeva/rvander/DATA:/cluster/dataset/boeva/rvander/DATA \
    --nv /cluster/customapps/biomed/boeva/fbarkmann/bionemo-framework_nightly.sif \
    python pretrain.py \
    --gpus 4 \
    --save-dir "$SAVE_DIR" \
    --max-seq-len 1200 \
    --batch-size 32 \
    --nlayers 6 \
    --nheads 8 \
    --embsize 128 \
    --d-hi 256 \
    --epochs 1 \
    --lr 0.0001 \
    --warmup-ratio-or-step 10000 \
    --val-check-interval 0.5 \
    --trunc-by-sample \
    --loss mse \
    --balance-primary technology \
    --train-path "$TRAIN_DIR" \
    --zero-percentages 0.2 0.4 0.6 \
    --strategy='ddp' \
    --seed 0 \
    --compile \
    --wandb "brain" \
    --wandb-name "${SLURM_JOB_NAME}_${SLURM_JOB_ID}"

if [ -d "./lightning_logs/version_${SLURM_JOB_ID}" ]; then
    mv "./lightning_logs/version_${SLURM_JOB_ID}" "$SAVE_DIR/lightning_log"
fi

cp "./$TRAIN_DIR/vocab.json" "$SAVE_DIR/vocab.json"
cp "$0" "$SAVE_DIR/run_script.sh"
mv ./"${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" "$SAVE_DIR/slurm.out"
echo "Job finished. Outputs and logs are in $SAVE_DIR"