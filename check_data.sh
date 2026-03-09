#!/bin/bash -l
#SBATCH --job-name=check_data
#SBATCH --output=./%x_%j.out
#SBATCH --time=00:05:00
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --account=a132

TOML=${TOML:-./bionemo_bristen.toml}
srun -ul --environment=$TOML python -c "
import anndata as ad
import numpy as np

# Eval dataset
print('=== Eval dataset ===')
adata = ad.read_h5ad('/capstor/scratch/cscs/rvander/DATA/brain/neftel_ss2.h5ad')
X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
print(f'shape={X.shape}, mean={X.mean():.3f}, std={X.std():.3f}, min={X.min():.3f}, max={X.max():.3f}, zero_frac={np.mean(X==0):.3f}')

# Training dataset - raw data.npy (bypass bionemo API)
print('=== Training dataset (raw data.npy) ===')
TRAIN_PATH = '/capstor/scratch/cscs/rvander/DATA/brain_processed/processed_data/train/mem.map'
data = np.fromfile(f'{TRAIN_PATH}/data.npy', dtype=np.float32)
sample = data[:500000]
nonzero = sample[sample != 0]
print(f'total_values={data.shape[0]}, mean={sample.mean():.3f}, std={sample.std():.3f}, min={sample.min():.3f}, max={sample.max():.3f}, zero_frac={np.mean(sample==0):.3f}')
print(f'non-zero: mean={nonzero.mean():.3f}, std={nonzero.std():.3f}, min={nonzero.min():.3f}, max={nonzero.max():.3f}')
print(f'first 20 values: {data[:20]}')

# Also check via bionemo API for comparison
print('=== Training dataset (bionemo API, first 10 rows) ===')
from bionemo.scdl.io.single_cell_memmap_dataset import SingleCellMemMapDataset
mm = SingleCellMemMapDataset(TRAIN_PATH)
vals = np.concatenate([mm[i].numpy().ravel() for i in range(min(10, mm.number_of_rows()))])
print(f'n_cells={mm.number_of_rows()}, mean={vals.mean():.3f}, std={vals.std():.3f}, min={vals.min():.3f}, max={vals.max():.3f}')
print(f'first 20 values: {vals[:20]}')
"
