#!/bin/bash -l
#SBATCH --job-name=cf-pretrain
#SBATCH --ntasks-per-gpu=1
#SBATCH --time=12:00:00
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=per_task:1


# Run job step
LOG_INTERVAL=16
MAX_LENGTH=1200
per_proc_batch_size=256
LAYERS=6
EMBSIZE=256
JOB_NAME="debug"
SAVE_DIR="./save/scaling_data_${SLURM_NNODES}"
export GPUS_PER_NODE=4
PORT=$(($RANDOM + ($RANDOM % 2) * 32768))
CURRENT_EPOCH=$SLURM_ARRAY_TASK_ID

head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

srun --environment=bionemo ${JOBREPORT} -o report -- accelerate launch \
    --num_processes $((SLURM_NNODES * GPUS_PER_NODE)) \
    --num_machines $SLURM_NNODES \
    --machine_rank $SLURM_PROCID \
    --rdzv_backend c10d \
    --main_process_ip $head_node_ip \
    --main_process_port $PORT \
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
    --conditions "technology" \
    --do-dat \
    --wandb "fulldata"

${JOBREPORT} print report