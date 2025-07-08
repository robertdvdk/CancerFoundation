#!/bin/bash -l
#SBATCH --job-name=cf-pretrain

singularity run --pwd /cluster/work/boeva/fbarkmann/CancerFoundation --bind /cluster/work/boeva/fbarkmann/CancerFoundation:/cluster/work/boeva/fbarkmann/CancerFoundation --nv /cluster/customapps/biomed/boeva/fbarkmann/bionemo-framework_nightly.sif python pretrain.py \
    --gpus 4 \
    --save-dir ./save/cf_oversampling-$(date +%b%d-%H-%M-%Y) \
    --max-seq-len 1200 \
    --batch-size 32 \
    --nlayers 6 \
    --nheads 8 \
    --embsize 256 \
    --d-hi 512 \
    --epochs 15 \
    --lr 0.0001 \
    --warmup-ratio-or-step 10000 \
    --grad-accu-steps 2 \
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