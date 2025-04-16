#!/bin/bash -l
#SBATCH --job-name=cf-pretrain
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00


# Run job step


LOG_INTERVAL=16
MAX_LENGTH=1200
per_proc_batch_size=32
LAYERS=6
EMBSIZE=256
JOB_NAME="cancerfoundation"

ACCEL_PROCS=$(( $SLURM_NNODES * 4 ))

MAIN_ADDR=$(echo "${SLURM_NODELIST}" | sed 's/[],].*//g; s/\[//g')
MAIN_PORT=128485

srun --environment=bionemo accelerate launch \
    --num_machines=$SLURM_NNODES --num_processes=$ACCEL_PROCS \
    --machine_rank $SLURM_PROCID \
    --main_process_ip $MAIN_ADDR --main_process_port $MAIN_PORT \
    ./pretrain.py \
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
    --conditions "technology" \
    --do-dat \
    --wandb "debug"