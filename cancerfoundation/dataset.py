import os
from typing import Dict

import torch
import numpy as np
import pandas as pd
from typing import Dict, Optional, Union, List
import json

class scDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        path: Union[str, os.PathLike],
        metadata: Optional[List[str]] = None,  # List of metadata keys to enable
        test_metadata_completeness: bool = False,
    ) -> None:
        self.path = path
        self.obs_df = pd.read_csv(f"{path}/obs.csv")

        # Initialize metadata dynamically
        self.metadata = {}
        if metadata:
            for meta_key in metadata:
                meta_file = os.path.join(self.path, f"{meta_key}.json")
                if os.path.exists(meta_file):
                    with open(meta_file, 'r') as f:
                        self.metadata[meta_key] = json.load(f)
                else:
                    raise FileNotFoundError(f"{meta_key}.json not found in path.")

                # Optionally test metadata completeness
                if test_metadata_completeness:
                    self.__test_metadata_completeness(meta_key)

    def __test_metadata_completeness(self, meta_key: str):
        datasets = self.obs_df['dataset'].unique()
        for dataset in datasets:
            file_path = os.path.join(self.path, dataset, f"{meta_key}.bin")

            if not os.path.exists(file_path):
                raise FileNotFoundError(f"The file '{file_path}' does not exist.")
    
    def get_metadata_cardinality(self, metadata_key:str):
        return len(self.metadata[metadata_key])

    def __len__(self) -> int:
        return len(self.obs_df)

    def __getitem__(self, idx: int) -> Dict[str, np.array]:
        cell_id = self.obs_df.iloc[idx]['cell_id']
        dataset = self.obs_df.iloc[idx]['dataset']
        data_dir = os.path.join(self.path, dataset)

        # Placeholder for gene_feature, change as necessary
        gene_feature = np.load(
            os.path.join(data_dir, "gene.npy")
        ).flatten()

        gene_data_np = np.memmap(
            os.path.join(data_dir, "rna_X.data.bin"), 
            dtype=np.float32, mode='r', 
        )
        gene_indices_np = np.memmap(
            os.path.join(data_dir, "rna_X.indices.bin"), 
            dtype=np.int32, mode='r', 
        )
        gene_indptr_np = np.memmap(
            os.path.join(data_dir, "rna_X.indptr.bin"),
            dtype=np.int32, mode='r',
        )
        start = gene_indptr_np[cell_id]
        end = gene_indptr_np[cell_id + 1]
        nz_data = gene_data_np[start:end]
        nz_indices = gene_indices_np[start:end]
        gene_x = np.zeros(gene_feature.size, dtype=np.float32)
        gene_x[nz_indices] = nz_data

        batch = {
            'genes': gene_feature, 
            'expressions': gene_x,
        }

        # Add metadata dynamically
        for meta_key in self.metadata:
            batch[meta_key] = np.memmap(os.path.join(data_dir, f"{meta_key}.bin"), dtype='int32', mode='r')[cell_id]
        return batch
    
    def get_metadata(self, idx: int) -> Dict[str, np.array]:
        cell_id = self.obs_df.iloc[idx]['cell_id']
        dataset = self.obs_df.iloc[idx]['dataset']
        data_dir = os.path.join(self.path, dataset)
        batch = {}
        for meta_key in self.metadata:
            batch[meta_key] = np.memmap(os.path.join(data_dir, f"{meta_key}.bin"), dtype='int32', mode='r')[cell_id]
        return batch
    

