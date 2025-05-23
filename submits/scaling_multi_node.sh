#!/bin/bash -l
#SBATCH --job-name=cf-pretrain
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=72
#SBATCH --exclusive

set -x -e

# Run job step
# Training setup
GPUS_PER_NODE=4
# so processes know who to talk to
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=$(shuf -i 40000-65000 -n 1)
NNODES=$SLURM_NNODES


TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
REPORT_PATH="report_$SLURM_JOB_ID"

LOG_INTERVAL=16
MAX_LENGTH=1200
per_proc_batch_size=128
LAYERS=8
EMBSIZE=512
JOB_NAME="debug"
SAVE_DIR="./save/scaling_data_${SLURM_NNODES}"

export GPUS_PER_NODE=4




srun --environment=bionemo bash -c "echo \$SLURM_PROCID; ${JOBREPORT} --ignore-gpu-binding -o $REPORT_PATH -- python \
    ./pretrain.py \
    --gpus $GPUS_PER_NODE \
    --num-nodes $SLURM_NNODES \
    --strategy "ddp_find_unused_parameters_true" \
    --save-dir $SAVE_DIR \
    --max-seq-len $MAX_LENGTH \
    --batch-size $per_proc_batch_size \
    --eval-batch-size $(($per_proc_batch_size)) \
    --nlayers $LAYERS \
    --nheads 8 \
    --embsize $EMBSIZE \
    --d-hi 512 \
    --epochs 5 \
    --num-epochs 1 \
    --lr 0.0001 \
    --warmup-ratio-or-step 10000 \
    --log-interval $LOG_INTERVAL \
    --trunc-by-sample \
    --loss "mse" \
    --train-path "./pretraining_cells/" \
    --zero-percentages 0.2 0.4 0.6 \
    --balance-primary "tissue" \
    --balance-secondary "technology" \
    --grad-accu-steps 8 \
    --wandb "fulldata""

${JOBREPORT} print $REPORT_PATH