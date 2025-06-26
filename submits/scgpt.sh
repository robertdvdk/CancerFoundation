#!/bin/bash -l
#SBATCH --job-name=cf-pretrain

srun --environment=bionemo python pretrain.py \
    --gpus 4 \
    --save-dir ./save/scgpt-$(date +%b%d-%H-%M-%Y) \
    --max-seq-len 1200 \
    --batch-size 32 \
    --nlayers 12 \
    --nheads 8 \
    --embsize 512 \
    --d-hi 512 \
    --epochs 3 \
    --lr 0.0001 \
    --warmup-ratio-or-step 5000 \
    --val-check-interval 0.25 \
    --trunc-by-sample \
    --loss mse \
    --balance-primary tissue \
    --balance-secondary technology \
    --train-path training_data/pretraining_data/pretraining_cells \
    --pretrained weights/scgpt \
    --wandb debug \
    --zero-percentages 0.2 0.4 0.6 \
    --strategy='ddp_find_unused_parameters_true'
