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

# Training dataset (memory-mapped)
print('=== Training dataset (first 100 rows) ===')
from cancerfoundation.data.dataset import SingleCellDataset
ds = SingleCellDataset('/capstor/scratch/cscs/rvander/DATA/brain/processed_data/train')
rows = np.stack([ds[i]['expressions'].numpy() for i in range(min(100, len(ds)))])
pad_mask = rows != ds.pad_value
vals = rows[pad_mask]
print(f'n_cells={len(ds)}, sampled=100, mean={vals.mean():.3f}, std={vals.std():.3f}, min={vals.min():.3f}, max={vals.max():.3f}, zero_frac={np.mean(vals==0):.3f}')
"
