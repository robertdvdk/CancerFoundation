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
rows = np.stack([mm[i].numpy() for i in range(min(100, mm.number_of_rows()))])
nonpad = rows[rows != 0]
print(f'n_cells={mm.number_of_rows()}, sampled=100, mean={rows.mean():.3f}, std={rows.std():.3f}, min={rows.min():.3f}, max={rows.max():.3f}, zero_frac={np.mean(rows==0):.3f}')
print(f'non-zero: mean={nonpad.mean():.3f}, std={nonpad.std():.3f}, min={nonpad.min():.3f}, max={nonpad.max():.3f}')
"
