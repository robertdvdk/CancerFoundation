#!/bin/bash
# Submit all 15 sweep experiments on bristen
set -e

echo "Submitting baseline..." && sbatch submits_cscs/sweep/baseline.sh
echo "Submitting pcpt_only..." && sbatch submits_cscs/sweep/pcpt_only.sh
echo "Submitting no_mvc..." && sbatch submits_cscs/sweep/no_mvc.sh
echo "Submitting emb_mine..." && sbatch submits_cscs/sweep/emb_mine.sh
echo "Submitting gen_quick..." && sbatch submits_cscs/sweep/gen_quick.sh
echo "Submitting gen_theirs..." && sbatch submits_cscs/sweep/gen_theirs.sh
echo "Submitting gen_mine..." && sbatch submits_cscs/sweep/gen_mine.sh
echo "Submitting small..." && sbatch submits_cscs/sweep/small.sh
echo "Submitting large..." && sbatch submits_cscs/sweep/large.sh
echo "Submitting their_init..." && sbatch submits_cscs/sweep/their_init.sh
echo "Submitting normalise_bins..." && sbatch submits_cscs/sweep/normalise_bins.sh
echo "Submitting pre_ln_gelu..." && sbatch submits_cscs/sweep/pre_ln_gelu.sh
echo "Submitting ordinal_ce..." && sbatch submits_cscs/sweep/ordinal_ce.sh
echo "Submitting explicit_zero..." && sbatch submits_cscs/sweep/explicit_zero.sh
echo "Submitting emb_mine_gen_mine..." && sbatch submits_cscs/sweep/emb_mine_gen_mine.sh

echo "\nSubmitted 15 jobs."
