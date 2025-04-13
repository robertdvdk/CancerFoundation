#!/bin/bash -l
#SBATCH --job-name=edf-example
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -p debug

# Run job step
srun --environment=bionemo accelerate launch ./pretrain.py \
    --save-dir ./save/cancerfoundation-2025/04/13-21:59:21 \
    --max-seq-len 2000 \
    --batch-size 16 \
    --do-dat \
    --eval-batch-size 64 \
    --nlayers 6 \
    --nheads 8 \
    --embsize 128 \
    --d-hi 128 \
    --epochs 10 \
    --lr 0.0001 
    --warmup-ratio-or-step 10000 \
    --log-interval 16 \
    --trunc-by-sample \
    --loss mse \
    --conditions technology \
    --balance-primary tissue \
    --balance-secondary technology\
    --train-path training_data/pretraining_data/dataset_pretraining \
    --wandb debug \
    --zero-percentages 0.2 0.4 0.6