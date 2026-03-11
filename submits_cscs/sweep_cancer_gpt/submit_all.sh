#!/bin/bash
# Submit all 16 sweep experiments on bristen (cancer_gpt)
set -e

echo "Submitting baseline_seed0..." && sbatch submits_cscs/sweep_cancer_gpt/baseline_seed0.sh
echo "Submitting baseline_seed42..." && sbatch submits_cscs/sweep_cancer_gpt/baseline_seed42.sh
echo "Submitting both_no_mvc_seed0..." && sbatch submits_cscs/sweep_cancer_gpt/both_no_mvc_seed0.sh
echo "Submitting both_no_mvc_seed42..." && sbatch submits_cscs/sweep_cancer_gpt/both_no_mvc_seed42.sh
echo "Submitting pcpt_mvc_seed0..." && sbatch submits_cscs/sweep_cancer_gpt/pcpt_mvc_seed0.sh
echo "Submitting pcpt_mvc_seed42..." && sbatch submits_cscs/sweep_cancer_gpt/pcpt_mvc_seed42.sh
echo "Submitting pcpt_no_mvc_seed0..." && sbatch submits_cscs/sweep_cancer_gpt/pcpt_no_mvc_seed0.sh
echo "Submitting pcpt_no_mvc_seed42..." && sbatch submits_cscs/sweep_cancer_gpt/pcpt_no_mvc_seed42.sh
echo "Submitting baseline_unbinned_seed0..." && sbatch submits_cscs/sweep_cancer_gpt/baseline_unbinned_seed0.sh
echo "Submitting baseline_unbinned_seed42..." && sbatch submits_cscs/sweep_cancer_gpt/baseline_unbinned_seed42.sh
echo "Submitting both_no_mvc_unbinned_seed0..." && sbatch submits_cscs/sweep_cancer_gpt/both_no_mvc_unbinned_seed0.sh
echo "Submitting both_no_mvc_unbinned_seed42..." && sbatch submits_cscs/sweep_cancer_gpt/both_no_mvc_unbinned_seed42.sh
echo "Submitting pcpt_mvc_unbinned_seed0..." && sbatch submits_cscs/sweep_cancer_gpt/pcpt_mvc_unbinned_seed0.sh
echo "Submitting pcpt_mvc_unbinned_seed42..." && sbatch submits_cscs/sweep_cancer_gpt/pcpt_mvc_unbinned_seed42.sh
echo "Submitting pcpt_no_mvc_unbinned_seed0..." && sbatch submits_cscs/sweep_cancer_gpt/pcpt_no_mvc_unbinned_seed0.sh
echo "Submitting pcpt_no_mvc_unbinned_seed42..." && sbatch submits_cscs/sweep_cancer_gpt/pcpt_no_mvc_unbinned_seed42.sh

echo "\nSubmitted 16 jobs."
