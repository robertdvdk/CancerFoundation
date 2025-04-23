#!/bin/bash -l
#SBATCH --job-name=cf-pretrain
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:15:00
#SBATCH --array [0-2]%1

# Run job step


LOG_INTERVAL=16
MAX_LENGTH=1200
per_proc_batch_size=32
LAYERS=6
EMBSIZE=256
JOB_NAME="debug"
SAVE_DIR="./save/resume_test"
export GPUS_PER_NODE=4

CURRENT_EPOCH=$SLURM_ARRAY_TASK_ID

export GPUS_PER_NODE=4
head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

if [ $CURRENT_EPOCH -eq 0 ]; then
  echo "Running first epoch (epoch $CURRENT_EPOCH)"

srun --environment=bionemo accelerate launch \
    --num_processes $((SLURM_NNODES * GPUS_PER_NODE)) \
    --num_machines $SLURM_NNODES \
    --machine_rank $SLURM_PROCID \
    --rdzv_backend c10d \
    --main_process_ip $head_node_ip \
    --main_process_port 29501 \
    --mixed_precision bf16 \
    ./pretrain.py \
    --save-dir  $SAVE_DIR \
    --max-seq-len $MAX_LENGTH \
    --batch-size $per_proc_batch_size \
    --eval-batch-size $(($per_proc_batch_size)) \
    --nlayers $LAYERS \
    --nheads 8 \
    --embsize $EMBSIZE \
    --d-hi 512 \
    --epochs 3 \
    --num-epochs 1 \
    --lr 0.0001 \
    --warmup-ratio-or-step 10000 \
    --log-interval $LOG_INTERVAL \
    --trunc-by-sample \
    --loss "mse" \
    --train-path "./debug_data/" \
    --zero-percentages 0.2 0.4 0.6 \
    --balance-primary "tissue" \
    --balance-secondary "technology" \
    --conditions "technology" \
    --do-dat \
    --wandb "debug"

else
PREV_EPOCH=$((CURRENT_EPOCH - 1))
CHECKPOINT_PATH="$SAVE_DIR/epoch_$PREV_EPOCH"

srun --environment=bionemo accelerate launch \
    --num_processes $((SLURM_NNODES * GPUS_PER_NODE)) \
    --num_machines $SLURM_NNODES \
    --machine_rank $SLURM_PROCID \
    --rdzv_backend c10d \
    --main_process_ip $head_node_ip \
    --main_process_port 29501 \
    --mixed_precision bf16 \
    ./pretrain.py \
    --resume-from-checkpoint $CHECKPOINT_PATH \
    --num-epochs 1

fi