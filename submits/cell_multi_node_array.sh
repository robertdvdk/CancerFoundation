#!/bin/bash -l
#SBATCH --job-name=cf-pretrain

srun --environment=bionemo python pretrain.py \
    --gpus 2 \
    --save-dir ./save/CF-$(date +%b%d-%H-%M-%Y) \
    --max-seq-len 1200 \
    --batch-size 32 \
    --eval-batch-size 32 \
    --nlayers 6 \
    --nheads 8 \
    --embsize 256 \
    --d-hi 512 \
    --epochs 15 \
    --lr 0.0001 \
    --warmup-ratio-or-step 10000 \
    --log-interval 32 \
    --trunc-by-sample \
    --loss "mse" \
    --train-path "./training_data/pretraining_data/pretraining_cells/" \
    --zero-percentages 0.2 0.4 0.6 \
    --conditions "technology" \
    --balance-primary "tissue" \
    --balance-secondary "technology" \
    --wandb "cells"