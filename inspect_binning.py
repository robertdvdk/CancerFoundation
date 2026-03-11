"""Inspect what binning does to expression values."""

import numpy as np
import torch

from cancerfoundation.data.preprocess import binning

# Simulate a typical cell's expression vector (log1p-transformed counts)
# Mix of zeros (dropout) and non-zero values typical of scRNA-seq
np.random.seed(42)
n_genes = 50
raw = np.zeros(n_genes)
nonzero_idx = np.random.choice(n_genes, size=20, replace=False)
raw[nonzero_idx] = np.random.exponential(scale=2.0, size=20)
raw = np.log1p(raw)  # log1p transform, as is standard

expr = torch.tensor(raw, dtype=torch.float32)

n_bins = 51  # default in most configs

binned = binning(row=expr.clone(), n_bins=n_bins)
binned_normed = binning(row=expr.clone(), n_bins=n_bins) / n_bins

print("=== Raw expression (log1p-transformed) ===")
print(f"Shape: {expr.shape}")
print(f"Min: {expr.min():.4f}  Max: {expr.max():.4f}  Mean: {expr.mean():.4f}")
print(f"Zero count: {(expr == 0).sum()}/{len(expr)}")
print(f"Values: {expr.numpy().round(3)}")

print("\n=== After binning (n_bins={}) ===".format(n_bins))
print(f"Min: {binned.min():.0f}  Max: {binned.max():.0f}  Mean: {binned.float().mean():.2f}")
print(f"Unique values: {sorted(binned.unique().tolist())}")
print(f"Zero count: {(binned == 0).sum()}/{len(binned)}  (zeros stay zero)")
print(f"Values: {binned.numpy()}")

print("\n=== After binning + normalise_bins (/ n_bins) ===")
print(f"Min: {binned_normed.min():.4f}  Max: {binned_normed.max():.4f}")
print(f"Values: {binned_normed.numpy().round(4)}")

print("\n=== Side-by-side (non-zero only) ===")
print(f"{'Raw':>8s} {'Binned':>8s} {'Normed':>8s}")
for i in range(len(expr)):
    if expr[i] > 0:
        print(f"{expr[i]:8.3f} {binned[i]:8.0f} {binned_normed[i]:8.4f}")

# Show the quantile bin edges used internally
nonzero = expr[expr > 0].numpy()
bins = np.quantile(nonzero, np.linspace(0, 1, n_bins - 1))
print(f"\n=== Quantile bin edges ({len(bins)} edges for {n_bins} bins) ===")
print(np.round(bins, 4))
