#!/bin/bash

#SBATCH --job-name=finetune_brain_cancer
#SBATCH --output=finetune_brain_cancer.txt
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH -p gpu
#SBATCH -A a0-05



# Set visible GPUs to what SLURM assigned (if available)
export CUDA_VISIBLE_DEVICES=$(echo $SLURM_JOB_GPUS | tr ',' '\n' | paste -sd "," -)
echo "Setting CUDA_VISIBLE_DEVICES to: $CUDA_VISIBLE_DEVICES"

# Activate conda
source ~/.bashrc
conda activate cancerfoundation

# Check env
echo
echo "which python"
which python

LOG_INTERVAL=16
MAX_LENGTH=1200
per_proc_batch_size=16
LAYERS=6
EMBSIZE=128
JOB_NAME="cancerfoundation"

VOCAB_PATH="./vocab.json"

echo "starting script"

accelerate launch --config-file="./config.yaml" \
    pretrain.py \
    --save-dir ./save/$JOB_NAME-$(date +%b%d-%H-%M-%Y) \
    --max-seq-len $MAX_LENGTH \
    --batch-size $per_proc_batch_size \
    --eval-batch-size $(($per_proc_batch_size)) \
    --nlayers $LAYERS \
    --nheads 8 \
    --embsize $EMBSIZE \
    --d-hi 128 \
    --grad-accu-steps 4 \
    --epochs 10 \
    --lr 0.0001 \
    --warmup-ratio-or-step 10000 \
    --log-interval $LOG_INTERVAL \
    --trunc-by-sample \
    --loss "mse" \
    --vocab $VOCAB_PATH \
    --train-path "../CancerFoundation_training_bionemo_2/data" \
    --eval-path "../CancerFoundation_training_bionemo_2/data" \
    --zero-percentages 0.2 0.4 0.6

