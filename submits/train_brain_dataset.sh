#!/bin/bash -l
#SBATCH --job-name=train_brain_dataset
#SBATCH --output=./slurmlogs/outputs/train_brain_dataset_%j_%t.out
#SBATCH --error=./slurmlogs/errors/train_brain_dataset_%j_%t.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=15

mkdir -p ./slurmlogs/outputs
mkdir -p ./slurmlogs/errors

srun singularity run \
    --pwd /cluster/work/boeva/rvander/CancerFoundation \
    --bind /cluster/work/boeva/rvander/CancerFoundation:/cluster/work/boeva/rvander/CancerFoundation \
    --bind /cluster/dataset/boeva/rvander/DATA:/cluster/dataset/boeva/rvander/DATA \
    --nv /cluster/customapps/biomed/boeva/fbarkmann/bionemo-framework_nightly.sif \
    python pretrain.py \
    --gpus 4 \
    --save-dir ./save/cf_brain-$(date +%b%d-%H-%M-%Y) \
    --max-seq-len 1200 \
    --batch-size 32 \
    --nlayers 6 \
    --nheads 8 \
    --embsize 128 \
    --d-hi 256 \
    --epochs 50 \
    --lr 0.0001 \
    --warmup-ratio-or-step 10000 \
    --val-check-interval 0.5 \
    --trunc-by-sample \
    --loss mse \
    --balance-primary technology \
    --train-path /cluster/dataset/boeva/rvander/DATA/processed_data/train \
    --zero-percentages 0.2 0.4 0.6 \
    --strategy='ddp' \
    --seed 0 \
    --compile