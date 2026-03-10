#!/bin/bash
# Submit all 4 sweep experiments on bristen
set -e

echo "Submitting baseline..." && sbatch submits_cscs/sweep/baseline.sh
echo "Submitting both_no_mvc..." && sbatch submits_cscs/sweep/both_no_mvc.sh
echo "Submitting pcpt_mvc..." && sbatch submits_cscs/sweep/pcpt_mvc.sh
echo "Submitting pcpt_no_mvc..." && sbatch submits_cscs/sweep/pcpt_no_mvc.sh

echo "\nSubmitted 4 jobs."
