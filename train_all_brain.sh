#!/usr/bin/env bash
set -euo pipefail

DATASETS=(brain_woneftel brain_wneftel brain_wss2 brain_w10x)
SEEDS=(1 2)

for ds in "${DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        stamp=$(date +%b%d-%H-%M-%Y)
        echo "=============================================="
        echo "Training on ${ds} seed=${seed}  (save: ./save/CF-${ds}-seed${seed}-${stamp})"
        echo "=============================================="

        python pretrain.py \
            --gpus 1 \
            --save-dir "./save/CF-${ds}-seed${seed}-${stamp}" \
            --max-seq-len 1200 \
            --nlayers 6 \
            --nheads 8 \
            --embsize 128 \
            --d-hid 256 \
            --epochs 50 \
            --lr 0.0001 \
            --warmup-ratio-or-step 1 \
            --trunc-by-sample \
            --loss "mse" \
            --train-path "./DATA/${ds}/processed_data/train" \
            --balance-primary "technology" \
            --zero-percentages 0.2 0.4 0.6 \
            --strategy='auto' \
            --seed "${seed}" \
            --log-interval 50 \
            --training-tasks "both" \
            --input-emb-style "theirs" \
            --their-init-weights \
            --num-workers 8 \
            --batch-size 64 \
            --precision "bf16-mixed" \
            --gen-method "quick" \
            --wandb "brain" \
            --wandb-name "CancerFoundation_${ds#brain_}_seed${seed}" \
            --compile \
            --input-style "binned" \
            --eval-every-n-epochs 5 \
            --eval-datasets DATA/neftel_ss2.h5ad
    done
done

echo "All eight training runs finished."
