#!/bin/bash -l
#SBATCH --job-name=check_data
#SBATCH --output=./%x_%j.out
#SBATCH --time=00:05:00
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --account=a132

srun -ul --environment=./bionemo_clariden.toml python -c "
import anndata as ad
import numpy as np

# Eval dataset
print('=== Eval dataset ===')
adata = ad.read_h5ad('/capstor/scratch/cscs/rvander/DATA/brain/neftel_ss2.h5ad')
X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
print(f'shape={X.shape}, mean={X.mean():.3f}, std={X.std():.3f}, min={X.min():.3f}, max={X.max():.3f}, zero_frac={np.mean(X==0):.3f}')

# Training dataset (memory-mapped) - read raw memmap
print('=== Training dataset (first 100 rows) ===')
from bionemo.scdl.io.single_cell_memmap_dataset import SingleCellMemMapDataset
mm = SingleCellMemMapDataset('/capstor/scratch/cscs/rvander/DATA/brain/processed_data/train/mem.map')
vals = np.concatenate([mm[i].numpy() for i in range(min(100, mm.number_of_rows()))])
nonzero = vals[vals != 0]
print(f'n_cells={mm.number_of_rows()}, sampled=100, mean={vals.mean():.3f}, std={vals.std():.3f}, min={vals.min():.3f}, max={vals.max():.3f}, zero_frac={np.mean(vals==0):.3f}')
print(f'non-zero: mean={nonzero.mean():.3f}, std={nonzero.std():.3f}, min={nonzero.min():.3f}, max={nonzero.max():.3f}')
"
