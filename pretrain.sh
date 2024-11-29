#!/bin/bash
 
LOG_INTERVAL=16
MAX_LENGTH=1200
per_proc_batch_size=8
LAYERS=6
EMBSIZE=128
JOB_NAME="cancerfoundation"

VOCAB_PATH="./vocab.json"

accelerate launch --config-file="./config.yaml"\
    pretrain.py \
    --save-dir ./save/$JOB_NAME-$(date +%b%d-%H-%M-%Y) \
    --max-seq-len $MAX_LENGTH \
    --batch-size $per_proc_batch_size \
    --eval-batch-size $(($per_proc_batch_size)) \
    --nlayers $LAYERS \
    --nheads 8 \
    --embsize $EMBSIZE \
    --d-hi 256\
    --grad-accu-steps 2 \
    --epochs 15 \
    --lr 0.0001 \
    --warmup-ratio-or-step 10000 \
    --log-interval $LOG_INTERVAL \
    --trunc-by-sample \
    --loss "mse" \
    --vocab $VOCAB_PATH \
    --train-path "./train" \
    --eval-path "./eval" \
    --zero-percentages 0.2 0.4 0.6 \
    --conditions "technology" \
    --balance-primary "tissue" \
    --balance-secondary "technology"
