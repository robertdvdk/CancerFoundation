import scanpy as sc
import numpy as np
import argparse
import os


def merge_embeddings_to_h5ad(
    original_h5ad_path: str,
    embedding_npy_path: str,
    output_h5ad_path: str,
    embedding_key: str = "scfoundation_embedding",
):
    """
    Loads an AnnData object and a NumPy embedding file, adds the embeddings
    to the AnnData object, and saves the result to a new .h5ad file.
    """
    # --- 1. Input Validation ---
    if not os.path.exists(original_h5ad_path):
        raise FileNotFoundError(
            f"Original AnnData file not found at: {original_h5ad_path}"
        )
    if not os.path.exists(embedding_npy_path):
        raise FileNotFoundError(
            f"Embedding .npy file not found at: {embedding_npy_path}"
        )

    print(f"Loading original data from {original_h5ad_path}")
    adata = sc.read_h5ad(original_h5ad_path)

    print(f"Loading embeddings from {embedding_npy_path}")
    embeddings = np.load(embedding_npy_path)

    # --- 2. Shape Validation (Crucial) ---
    print(f"AnnData object has {adata.n_obs} cells.")
    print(f"Embeddings file has {embeddings.shape[0]} rows.")
    if adata.n_obs != embeddings.shape[0]:
        raise ValueError(
            "The number of cells in the AnnData object does not match the "
            "number of embeddings. Cannot merge."
        )

    # --- 3. Merge and Save ---
    print(f"Adding embeddings to AnnData object with key '.obsm[{embedding_key}]'")
    adata.obsm[embedding_key] = embeddings

    print(f"Saving merged AnnData object to {output_h5ad_path}")
    adata.write_h5ad(output_h5ad_path)
    print("✅ Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge NumPy embeddings into an .h5ad file."
    )
    parser.add_argument(
        "-i",
        "--input_h5ad",
        type=str,
        required=True,
        help="Path to the original .h5ad file.",
    )
    parser.add_argument(
        "-e",
        "--embedding_npy",
        type=str,
        required=True,
        help="Path to the .npy file containing the cell embeddings.",
    )
    parser.add_argument(
        "-o",
        "--output_h5ad",
        type=str,
        required=True,
        help="Path for the new output .h5ad file.",
    )
    parser.add_argument(
        "-k",
        "--key",
        type=str,
        default="scfoundation_embedding",
        help="Key to use for storing embeddings in .obsm (default: 'scfoundation_embedding').",
    )
    args = parser.parse_args()

    merge_embeddings_to_h5ad(
        original_h5ad_path=args.input_h5ad,
        embedding_npy_path=args.embedding_npy,
        output_h5ad_path=args.output_h5ad,
        embedding_key=args.key,
    )
