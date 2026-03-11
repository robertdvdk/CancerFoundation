#!/bin/bash
# Submit all 16 sweep experiments on bristen
set -e

echo "Submitting baseline_seed0..." && sbatch submits_cscs/sweep/baseline_seed0.sh
echo "Submitting baseline_seed42..." && sbatch submits_cscs/sweep/baseline_seed42.sh
echo "Submitting both_no_mvc_seed0..." && sbatch submits_cscs/sweep/both_no_mvc_seed0.sh
echo "Submitting both_no_mvc_seed42..." && sbatch submits_cscs/sweep/both_no_mvc_seed42.sh
echo "Submitting pcpt_mvc_seed0..." && sbatch submits_cscs/sweep/pcpt_mvc_seed0.sh
echo "Submitting pcpt_mvc_seed42..." && sbatch submits_cscs/sweep/pcpt_mvc_seed42.sh
echo "Submitting pcpt_no_mvc_seed0..." && sbatch submits_cscs/sweep/pcpt_no_mvc_seed0.sh
echo "Submitting pcpt_no_mvc_seed42..." && sbatch submits_cscs/sweep/pcpt_no_mvc_seed42.sh
echo "Submitting baseline_unbinned_seed0..." && sbatch submits_cscs/sweep/baseline_unbinned_seed0.sh
echo "Submitting baseline_unbinned_seed42..." && sbatch submits_cscs/sweep/baseline_unbinned_seed42.sh
echo "Submitting both_no_mvc_unbinned_seed0..." && sbatch submits_cscs/sweep/both_no_mvc_unbinned_seed0.sh
echo "Submitting both_no_mvc_unbinned_seed42..." && sbatch submits_cscs/sweep/both_no_mvc_unbinned_seed42.sh
echo "Submitting pcpt_mvc_unbinned_seed0..." && sbatch submits_cscs/sweep/pcpt_mvc_unbinned_seed0.sh
echo "Submitting pcpt_mvc_unbinned_seed42..." && sbatch submits_cscs/sweep/pcpt_mvc_unbinned_seed42.sh
echo "Submitting pcpt_no_mvc_unbinned_seed0..." && sbatch submits_cscs/sweep/pcpt_no_mvc_unbinned_seed0.sh
echo "Submitting pcpt_no_mvc_unbinned_seed42..." && sbatch submits_cscs/sweep/pcpt_no_mvc_unbinned_seed42.sh

echo "\nSubmitted 16 jobs."
