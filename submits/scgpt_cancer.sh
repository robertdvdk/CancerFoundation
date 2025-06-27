#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --exclusive

srun --environment=bionemo python pretrain.py \
    --gpus 4 \
    --save-dir ./save/scgpt_cancer-$(date +%b%d-%H-%M-%Y) \
    --max-seq-len 1200 \
    --batch-size 32 \
    --nlayers 12 \
    --nheads 8 \
    --embsize 512 \
    --d-hi 512 \
    --epochs 4 \
    --lr 0.0001 \
    --warmup-ratio-or-step 5000 \
    --val-check-interval 0.25 \
    --trunc-by-sample \
    --loss mse \
    --balance-primary tissue \
    --balance-secondary technology \
    --train-path training_data/pretraining_data/pretraining_cells \
    --pretrained weights/scgpt_cancer \
    --wandb scgpt \
    --wandb-entity cancerfoundation \
    --zero-percentages 0.2 0.4 0.6 \
    --strategy='ddp_find_unused_parameters_true'
