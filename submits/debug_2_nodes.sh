#!/bin/bash -l
#SBATCH --job-name=cf-pretrain
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --exclusive

# Run job step


LOG_INTERVAL=16
MAX_LENGTH=1200
per_proc_batch_size=32
LAYERS=6
EMBSIZE=256
JOB_NAME="debug"

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$(( RANDOM % (50000 - 30000 + 1 ) + 30000 ))
export GPUS_PER_NODE=$SLURM_GPUS_PER_NODE
export NNODES=$SLURM_NNODES
export NUM_PROCESSES=$(expr $NNODES \* $GPUS_PER_NODE)
export MULTIGPU_FLAG="--multi_gpu"

if [ $NNODES == "1" ]
then
        export MULTIGPU_FLAG=""
fi

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "Using $NNODES nodes, $NUM_PROCESSES GPUs total"

srun --environment=bionemo accelerate launch \
    --main_process_ip ${MASTER_ADDR} \
    --main_process_port ${MASTER_PORT} \
    --machine_rank $SLURM_NODEID \
    $MULTIGPU_FLAG \
    --same_network \
    --num_processes $NUM_PROCESSES \
    --num_machines $NNODES \
    --mixed_precision bf16 \
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
    --train-path "./debug_data/" \
    --zero-percentages 0.2 0.4 0.6 \
    --balance-primary "tissue" \
    --balance-secondary "technology" \
    --conditions "technology" \
    --do-dat \
    --wandb "debug"