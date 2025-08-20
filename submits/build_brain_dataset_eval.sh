#!/bin/bash -l
#SBATCH --job-name=create_brain_dataset_eval
#SBATCH --output=./%x_%j.out
#SBATCH --time=00:05:00
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=15

singularity run \
    --pwd /cluster/work/boeva/rvander/CancerFoundation \
    --bind /cluster/work/boeva/rvander/CancerFoundation:/cluster/work/boeva/rvander/CancerFoundation \
    --bind /cluster/dataset/boeva/rvander/DATA:/cluster/dataset/boeva/rvander/DATA \
    --nv /cluster/customapps/biomed/boeva/fbarkmann/bionemo-framework_nightly.sif \
    python ./scripts/h5ads_to_sc.py \
    --h5ad-path /cluster/dataset/boeva/rvander/DATA/brain/raw_data/eval \
    --vocab-path /cluster/dataset/boeva/rvander/DATA/brain/processed_data/train/vocab.json \
    --data-path /cluster/dataset/boeva/rvander/DATA/brain/processed_data/eval