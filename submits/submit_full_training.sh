#!/bin/bash -l
#SBATCH --job-name=cf-pretrain
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1


# Run job step


LOG_INTERVAL=16
MAX_LENGTH=1200
per_proc_batch_size=32
LAYERS=6
EMBSIZE=256
JOB_NAME="cancerfoundation"



srun --environment=bionemo accelerate launch ./pretrain.py \
    --save-dir ./save/$JOB_NAME-$(date +%b%d-%H-%M-%Y) \
    --max-seq-len $MAX_LENGTH \
    --batch-size $per_proc_batch_size \
    --eval-batch-size $(($per_proc_batch_size)) \
    --nlayers $LAYERS \
    --nheads 8 \
    --embsize $EMBSIZE \
    --d-hi 512 \
    --epochs 15 \
    --lr 0.0001 \
    --warmup-ratio-or-step 10000 \
    --log-interval $LOG_INTERVAL \
    --trunc-by-sample \
    --loss "mse" \
    --train-path "./dataset_pretraining" \
    --zero-percentages 0.2 0.4 0.6 \
    --balance-primary "tissue" \
    --balance-secondary "technology" \
    --wandb "debug"