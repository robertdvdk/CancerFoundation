#!/bin/bash -l
#SBATCH --job-name=create_dataset
#SBATCH --output=./slurmlogs/outputs/create_dataset_%j.out
#SBATCH --error=./slurmlogs/errors/create_dataset_%j.err
#SBATCH --time=00:05:00

singularity run \
    --pwd /cluster/work/boeva/rvander/CancerFoundation \
    --bind /cluster/work/boeva/rvander/CancerFoundation:/cluster/work/boeva/rvander/CancerFoundation \
    --bind /cluster/dataset/boeva/rvander/DATA:/cluster/dataset/boeva/rvander/DATA \
    --nv /cluster/customapps/biomed/boeva/fbarkmann/bionemo-framework_nightly.sif \
    python ./scripts/h5ads_to_sc.py \
    --h5ad-path /cluster/dataset/boeva/rvander/DATA/raw_data/train \
    --data-path /cluster/dataset/boeva/rvander/DATA/processed_data/train