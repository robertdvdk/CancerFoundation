#!/bin/bash -l
#SBATCH --job-name=cf-pretrain

singularity run --pwd /cluster/work/boeva/fbarkmann/CancerFoundation --bind /cluster/work/boeva/fbarkmann/CancerFoundation:/cluster/work/boeva/fbarkmann/CancerFoundation --nv /cluster/customapps/biomed/boeva/fbarkmann/bionemo-framework_nightly.sif python pretrain.py \
    --gpus 4 \
    --save-dir ./save/CF-$(date +%b%d-%H-%M-%Y) \
    --max-seq-len 1200 \
    --batch-size 32 \
    --eval-batch-size 32 \
    --nlayers 6 \
    --nheads 8 \
    --embsize 256 \
    --d-hi 512 \
    --epochs 2 \
    --lr 0.0001 \
    --warmup-ratio-or-step 10000 \
    --log-interval 32 \
    --grad-accu-steps 2 \
    --trunc-by-sample \
    --loss "mse" \
    --train-path "./training_data/pretraining_data/pretraining_cells/" \
    --zero-percentages 0.2 0.4 0.6 \
    --conditions "technology" \
    --balance-primary "tissue" \
    --balance-secondary "technology" \
    --wandb "cells" \
    --strategy "ddp_find_unused_parameters_true"