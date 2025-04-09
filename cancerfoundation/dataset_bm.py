from typing import OrderedDict
from bionemo.scdl.io.single_cell_memmap_dataset import SingleCellMemMapDataset
from torch.utils.data import Dataset
import polars as pl
import torch
import time

class scDataset(Dataset):
    def __init__(self, path: str, vocab: str, cache_size: int = 32):
        super().__init__()

        self.path = path

        # Load and join index + metadata, then convert to list of dicts
        index = pl.read_ndjson(f"{path}/index.jsonl")
        metadata = pl.read_json(f"{path}/metadata.json")
        self.obs = index.join(metadata, on="path", how="left").to_dicts()

        # Load vocab (already a plain dict)
        self.gene_vocab = pl.read_json(vocab)[0].to_dict(as_series=False)

        # genes.json is already a dict: {path: [genes]}
        self.genes = pl.read_json(f"{path}/genes.json").to_dicts()[0]

        # encodings.json is a dict of {feature_name: [[encodings]]}
        self.encodings = pl.read_json(f"{path}/encodings.json").to_dicts()[0]

        self._memmap_cache = OrderedDict()
        self._memmap_cache_size = cache_size

        self.gene_indices = {}
        for path, gene_list in self.genes.items():
            mapped = [
                self.gene_vocab[name][0]
                for name in gene_list
                if name in self.gene_vocab
            ]
            reverse_index = [
                i for i, name in enumerate(gene_list)
                if name in self.gene_vocab
            ]
            self.gene_indices[path] = (mapped, reverse_index)

    def __len__(self):
        return len(self.obs)

    def get_memmap(self, path):
        if path in self._memmap_cache:
            self._memmap_cache.move_to_end(path)  # Mark as recently used
            return self._memmap_cache[path]

        memmap = SingleCellMemMapDataset(path)

        # Add new memmap to cache
        self._memmap_cache[path] = memmap
        if len(self._memmap_cache) > self._memmap_cache_size:
            self._memmap_cache.popitem(last=False)  # Remove least recently used

        return memmap

    def get_metadata(self, index: int):
        row = self.obs[index]
        out = {}
        for key in self.encodings:
            if key in row and row[key] is not None:
                out[key] = self.encodings[key][0][row[key]]
        return out

    def __getitem__(self, index: int):
        t0 = time.perf_counter()

        row = self.obs[index]
        path = row["path"]
        cell_idx = row["cell_index"]

        t1 = time.perf_counter()
        expressions = self.get_memmap(path).get_row_padded(cell_idx)[0]

        #expressions = memmap[cell_idx]
        t2 = time.perf_counter()

        gene_names = self.genes[path]

        """indices = torch.as_tensor(expressions[1], dtype=torch.long)       # for indexing
        values = torch.as_tensor(expressions[0], dtype=torch.float64)     # match dtype

        # Preallocate dense tensor
        dense_expressions = torch.zeros(len(gene_names), dtype=torch.float64)
        # Use indexing to assign values
        dense_expressions[indices] = values"""

        dense_expressions = torch.as_tensor(expressions)

        t3 = time.perf_counter()

        out_genes, valid_indices = self.gene_indices[path]
        if out_genes:
            out_expressions = dense_expressions[list(valid_indices)]
        else:
            out_genes = []
            out_expressions = torch.tensor([])

        t4 = time.perf_counter()

        out = {
            "expressions": out_expressions,
            "genes": list(out_genes),
        }

        for key in self.encodings:
            if key in row and row[key] is not None:
                out[key] = self.encodings[key][row[key]]

        t5 = time.perf_counter()

        # Save timings to file
        log_line = (
            f"Index {index} | "
            f"Metadata: {t1 - t0:.4f}s | "
            f"Memmap: {t2 - t1:.4f}s | "
            f"Dense: {t3 - t2:.4f}s | "
            f"GeneFilter: {t4 - t3:.4f}s | "
            f"Encoding: {t5 - t4:.4f}s | "
            f"Total: {t5 - t0:.4f}s\n"
        )

        with open("out.txt", "a") as f:
            f.write(log_line)

        return out

