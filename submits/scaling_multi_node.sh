#!/bin/bash -l
#SBATCH --job-name=cf-pretrain
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=72
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:4
#SBATCH --exclusive

set -x -e

# Run job step
# Training setup
GPUS_PER_NODE=4
# so processes know who to talk to
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=$(shuf -i 40000-65000 -n 1)
NNODES=$SLURM_NNODES


LOG_INTERVAL=16
MAX_LENGTH=1200
per_proc_batch_size=128
LAYERS=6
EMBSIZE=256
JOB_NAME="debug"
SAVE_DIR="./save/scaling_data_${SLURM_NNODES}"

export GPUS_PER_NODE=4
export PORT=$(shuf -i 40000-65000 -n 1)

head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

srun --environment=bionemo bash -c "${JOBREPORT} -o report -- accelerate launch \
    --num_processes $((SLURM_NNODES * GPUS_PER_NODE)) \
    --num_machines $SLURM_NNODES \
    --machine_rank \$SLURM_PROCID \
    --rdzv_backend c10d \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --mixed_precision bf16 \
    ./pretrain.py \
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
    --train-path "./pretraining_cells" \
    --zero-percentages 0.2 0.4 0.6 \
    --balance-primary "tissue" \
    --balance-secondary "technology" \
    --wandb "fulldata""

${JOBREPORT} print report