#!/bin/bash
# Submit all 8 sweep experiments on bristen
set -e

echo "Submitting baseline..." && sbatch submits_cscs/sweep/baseline.sh
echo "Submitting both_no_mvc..." && sbatch submits_cscs/sweep/both_no_mvc.sh
echo "Submitting pcpt_mvc..." && sbatch submits_cscs/sweep/pcpt_mvc.sh
echo "Submitting pcpt_no_mvc..." && sbatch submits_cscs/sweep/pcpt_no_mvc.sh
echo "Submitting baseline_unbinned..." && sbatch submits_cscs/sweep/baseline_unbinned.sh
echo "Submitting both_no_mvc_unbinned..." && sbatch submits_cscs/sweep/both_no_mvc_unbinned.sh
echo "Submitting pcpt_mvc_unbinned..." && sbatch submits_cscs/sweep/pcpt_mvc_unbinned.sh
echo "Submitting pcpt_no_mvc_unbinned..." && sbatch submits_cscs/sweep/pcpt_no_mvc_unbinned.sh

echo "\nSubmitted 8 jobs."
