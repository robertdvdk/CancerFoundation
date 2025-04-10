from pathlib import Path
import sys
sys.path.insert(0, "./")

from cancerfoundation.dataset import DatasetDir
from argparse import ArgumentParser
from bionemo.scdl.io.single_cell_collection import SingleCellCollection

import scanpy as sc
import pandas as pd
import json


GENE_ID = "_cf_gene_id"
CLS_TOKEN = "<cls>"
PAD_TOKEN = "<pad>"


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--h5ad-path", type=Path)
    parser.add_argument("--data-path", type=Path)
    
    return parser.parse_args()


def _generate_vocab_from_h5ads(h5ads: Path, cls_token: str, pad_token: str) -> dict[str, int]:
    genes = set()
    for path in h5ads.iterdir():
        if not path.name.endswith(".h5ad"):
            continue
        var_names = sc.read_h5ad(path, backed="r").var_names
        genes.update(var_names)
    vocab = {gene: i for i, gene in enumerate([cls_token, pad_token] + list(genes))}
    return vocab

def _save_vocab_to_dir(vocab: dict[str, int], data_dir: DatasetDir) -> None:
    with open(data_dir.vocab_path , "w") as f:
        json.dump(vocab, f)
        
def _add_gene_id_to_h5ads(h5ads: Path, vocab: dict[str, int]) -> None:
    for path in h5ads.iterdir():
        if not path.name.endswith(".h5ad"):
            continue
        adata = sc.read_h5ad(path)
        adata.var[GENE_ID] = adata.var_names.map(vocab)
        adata.write_h5ad(path)


def convert_columns_to_categorical_with_mapping(df):
    df_copy = df.copy()
    
    category_mappings = {}
    
    df_categorical = pd.DataFrame(index=df.index)
    
    for column in df.columns:
        df_copy[column] = df_copy[column].astype('category')
        
        category_mappings[column] = dict(zip(df_copy[column].cat.categories, 
                                            range(len(df_copy[column].cat.categories))))
        
        df_categorical[column] = df_copy[column].cat.codes
    
    return df_categorical, category_mappings


def main():
    
    args =get_args()
    h5ad_path = args.h5ad_path
    columns = ['sample', 'disease', 'technology', 'tissue', "suspension", "project_id"]
    
    data_path = DatasetDir(args.data_path)
    data_path.mkdir()
    # Generate and save vocabulary
    vocab = _generate_vocab_from_h5ads(h5ad_path, CLS_TOKEN, PAD_TOKEN)
    _save_vocab_to_dir(vocab, data_path)
    
    # Add gene IDs to h5ad files
    _add_gene_id_to_h5ads(h5ad_path, vocab)
    
    # Create and process memmap files
    memmaps = SingleCellCollection(data_path.data_dir / "tmp")
    memmaps.load_h5ad_multi(h5ad_path, max_workers=12)
    
    # Collect observations
    obs_list = []
    for fname in memmaps.fname_to_mmap.keys():
        adata = sc.read_h5ad(h5ad_path / (fname.name + ".h5ad"), backed="r")
        obs_list.append(adata.obs[columns])
    
    obs = pd.concat(obs_list)
    
    obs, mapping = convert_columns_to_categorical_with_mapping(obs)
    
    obs.to_parquet(data_path.obs_path)
    
    with data_path.mapping_path.open("w") as f:
        json.dump(mapping, f, indent=4)
    
    # Flatten and create memmap dataset
    memmaps.flatten(data_path.memmap_path, destroy_on_copy=True)
    
if __name__ == "__main__":
    main()