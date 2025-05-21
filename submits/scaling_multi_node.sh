#!/bin/bash -l
#SBATCH --job-name=cf-pretrain
#SBATCH --time=12:00:00
#SBATCH --ntasks-per-gpu=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72

# Run job step
LOG_INTERVAL=16
MAX_LENGTH=1200
per_proc_batch_size=128
LAYERS=6
EMBSIZE=256
JOB_NAME="debug"
SAVE_DIR="./save/scaling_data_${SLURM_NNODES}"
export GPUS_PER_NODE=4
export PORT=$(shuf -i 40000-65000 -n 1)
CURRENT_EPOCH=$SLURM_ARRAY_TASK_ID

head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

srun --environment=bionemo --export=ALL,LOCAL_RANK=\$SLURM_LOCALID ${JOBREPORT} -o report -- accelerate launch \
    --machine_rank $SLURM_PROCID \
    --main_process_ip $head_node_ip \
    --main_process_port $PORT \
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
    --wandb "fulldata"

${JOBREPORT} print report