#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --exclusive

srun --environment=bionemo python pretrain.py \
    --gpus 4 \
    --save-dir ./save/cf_oversampling-$(date +%b%d-%H-%M-%Y) \
    --max-seq-len 1200 \
    --batch-size 64 \
    --nlayers 6 \
    --nheads 8 \
    --embsize 256 \
    --d-hi 512 \
    --epochs 15 \
    --lr 0.0001 \
    --warmup-ratio-or-step 10000 \
    --val-check-interval 0.5 \
    --trunc-by-sample \
    --loss mse \
    --balance-primary tissue \
    --balance-secondary technology \
    --train-path training_data/pretraining_data/cancer_gpt \
    --wandb cancer_gpt \
    --wandb-entity cancerfoundation \
    --zero-percentages 0.2 0.4 0.6 \
    --strategy='ddp_find_unused_parameters_true'
