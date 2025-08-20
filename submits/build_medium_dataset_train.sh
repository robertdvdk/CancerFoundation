#!/bin/bash -l
#SBATCH --job-name=create_medium_dataset_train
#SBATCH --output=./%x_%j.out
#SBATCH --time=00:30:00
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
    --h5ad-path /cluster/dataset/boeva/rvander/DATA/medium/raw_data/train \
    --data-path /cluster/dataset/boeva/rvander/DATA/medium/processed_data/train