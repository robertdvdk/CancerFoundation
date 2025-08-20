from pathlib import Path
import sys

sys.path.insert(0, "./")

from cancerfoundation.data.dataset import DatasetDir
from argparse import ArgumentParser
from bionemo.scdl.io.single_cell_collection import SingleCellCollection

import scanpy as sc
import pandas as pd
import json
import scipy.sparse
import anndata


GENE_ID = "_cf_gene_id"
CLS_TOKEN = "<cls>"
PAD_TOKEN = "<pad>"


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--h5ad-path", type=Path)
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--vocab-path", type=Path, required=False)
    return parser.parse_args()


def _generate_vocab_from_h5ads(
    h5ads: Path, cls_token: str, pad_token: str
) -> dict[str, int]:
    genes = set()
    for path in h5ads.iterdir():
        if not path.name.endswith(".h5ad"):
            continue
        var_names = sc.read_h5ad(path, backed="r").var_names
        genes.update(var_names)
    vocab = {gene: i for i, gene in enumerate([cls_token, pad_token] + list(genes))}
    return vocab


def _save_vocab_to_dir(vocab: dict[str, int], data_dir: DatasetDir) -> None:
    with open(data_dir.vocab_path, "w") as f:
        json.dump(vocab, f)


def _add_gene_id_to_h5ads(h5ads: Path, vocab: dict[str, int], data_path: Path) -> None:
    # Create the output directory for the new, consistent h5ad files
    output_h5ads_dir = data_path / "h5ads"
    output_h5ads_dir.mkdir(exist_ok=True, parents=True)

    # Create a list of all gene names from the vocab, excluding special tokens
    all_genes_in_vocab = [
        gene for gene in vocab.keys() if gene not in [CLS_TOKEN, PAD_TOKEN]
    ]

    for path in h5ads.iterdir():
        if not path.name.endswith(".h5ad"):
            continue

        print(f"Processing and aligning {path.name}...")
        adata = sc.read_h5ad(path)

        final_var = pd.DataFrame(index=all_genes_in_vocab)

        # --- FIX 1: Create as a csr_matrix to satisfy AnnData's requirements ---
        final_X = scipy.sparse.csr_matrix(
            (adata.n_obs, len(all_genes_in_vocab)), dtype=adata.X.dtype
        )
        # This will no longer produce a warning
        final_adata = anndata.AnnData(X=final_X, obs=adata.obs, var=final_var)

        common_genes = adata.var_names.intersection(all_genes_in_vocab)

        # --- FIX 2: Temporarily convert to LIL for the assignment, then convert back ---
        # Convert to LIL format for efficient modification
        final_adata.X = final_adata.X.tolil()

        # Perform the efficient assignment
        final_adata[:, common_genes].X = adata[:, common_genes].X

        # Convert back to CSR for efficient storage and downstream use
        final_adata.X = final_adata.X.tocsr()

        final_adata.var[GENE_ID] = final_adata.var_names.map(vocab).astype(int)
        final_adata.write_h5ad(output_h5ads_dir / path.name)


def convert_columns_to_categorical_with_mapping(df):
    df_copy = df.copy()

    category_mappings = {}

    df_categorical = pd.DataFrame(index=df.index)

    for column in df.columns:
        df_copy[column] = df_copy[column].astype("category")

        category_mappings[column] = dict(
            zip(
                df_copy[column].cat.categories,
                range(len(df_copy[column].cat.categories)),
            )
        )

        df_categorical[column] = df_copy[column].cat.codes

    return df_categorical, category_mappings


def main():
    args = get_args()

    h5ad_path = args.h5ad_path
    columns = ["sample", "cancer_type", "technology", "tissue"]

    data_path = DatasetDir(args.data_path)
    data_path.mkdir()
    # Generate and save vocabulary
    if args.vocab_path is None:
        vocab = _generate_vocab_from_h5ads(h5ad_path, CLS_TOKEN, PAD_TOKEN)
    else:
        vocab = json.load(args.vocab_path.open())
    _save_vocab_to_dir(vocab, data_path)
    # Add gene IDs to h5ad files
    _add_gene_id_to_h5ads(h5ad_path, vocab, args.data_path)

    # Create and process memmap files
    memmaps = SingleCellCollection(data_path.data_dir / "tmp")
    memmaps.load_h5ad_multi(args.data_path / "h5ads", max_workers=12)

    # Collect observations
    obs_list = []
    for i, fname in enumerate(memmaps.fname_to_mmap.keys()):
        print(f"Processing {i + 1}/{len(memmaps.fname_to_mmap)}: {fname.name}")
        adata = sc.read_h5ad(h5ad_path / (fname.name + ".h5ad"), backed="r")

        # Extract tissue name from the filename (e.g., '..._kidney' -> 'kidney')
        tissue_name = fname.name.split("_")[1]

        # Create the 'tissue' column and assign the extracted name to all cells
        adata.obs["tissue"] = tissue_name

        obs_list.append(adata.obs[columns])

    obs = pd.concat(obs_list)

    obs, mapping = convert_columns_to_categorical_with_mapping(obs)

    obs.to_parquet(data_path.obs_path)

    with data_path.mapping_path.open("w") as f:
        json.dump(mapping, f, indent=4)

    # Flatten and create memmap dataset
    memmaps.flatten(data_path.memmap_path, destroy_on_copy=True)

    print("Conversion completed successfully.")


if __name__ == "__main__":
    main()
