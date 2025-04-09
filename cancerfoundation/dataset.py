from typing import OrderedDict
from bionemo.scdl.io.single_cell_memmap_dataset import SingleCellMemMapDataset
from torch.utils.data import Dataset
import polars as pl
import torch

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
        row = self.obs[index]
        path = row["path"]
        cell_idx = row["cell_index"]

        expressions = self.get_memmap(path).get_row_padded(cell_idx)[0]
        gene_names = self.genes[path]
        dense_expressions = torch.as_tensor(expressions)

        out_genes, valid_indices = self.gene_indices[path]
        if out_genes:
            out_expressions = dense_expressions[list(valid_indices)]
        else:
            out_genes = []
            out_expressions = torch.tensor([])

        out = {
            "expressions": out_expressions,
            "genes": list(out_genes),
        }

        for key in self.encodings:
            if key in row and row[key] is not None:
                out[key] = self.encodings[key][row[key]]

        return out

