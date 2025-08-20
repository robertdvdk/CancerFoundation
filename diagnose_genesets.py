import scanpy as sc
import json
from pathlib import Path
from argparse import ArgumentParser
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_args():
    """Parses command-line arguments."""
    parser = ArgumentParser(
        description="Diagnose h5ad files for expressed genes with out-of-bounds IDs."
    )
    parser.add_argument(
        "--h5ad-dir",
        type=Path,
        required=True,
        help="Directory containing the ORIGINAL .h5ad files.",
    )
    parser.add_argument(
        "--vocab-path",
        type=Path,
        required=True,
        help="Path to the generated global vocab.json file.",
    )
    return parser.parse_args()


def main():
    """Main function to run the diagnosis."""
    args = get_args()

    logging.info(f"Loading global vocabulary from {args.vocab_path}...")
    with open(args.vocab_path, "r") as f:
        global_vocab = json.load(f)
    logging.info(f"Vocabulary loaded with {len(global_vocab)} total genes.")

    crashing_files = []
    safe_files = []

    h5ad_files = sorted(
        list(args.h5ad_dir.glob("*.h5ad"))
    )  # Sorted for consistent order
    logging.info(f"Found {len(h5ad_files)} files to check. Starting diagnosis...")
    print("-" * 70)

    for i, file_path in enumerate(h5ad_files):
        logging.info(f"Processing ({i+1}/{len(h5ad_files)}): {file_path.name}")
        try:
            adata = sc.read_h5ad(file_path)  # Load full data, not backed
            local_gene_names = adata.var_names
            local_gene_count = len(local_gene_names)

            # Find all genes in this file that have an out-of-bounds global ID
            problematic_genes = [
                gene
                for gene in local_gene_names
                if global_vocab.get(gene, -1) >= local_gene_count
            ]

            if not problematic_genes:
                logging.info("  [OK] No out-of-bounds gene IDs found in this file.")
                continue

            # Check if any of these problematic genes are actually expressed (non-zero)
            adata_problem_subset = adata[:, problematic_genes]

            # The sum of all expression data for the problematic genes
            total_expression_of_problem_genes = adata_problem_subset.X.sum()

            if total_expression_of_problem_genes > 0:
                logging.error(
                    f"  [CRASH CONFIRMED] This file will cause an error.\n"
                    f"    - It has {local_gene_count} unique genes.\n"
                    f"    - It contains problematic genes with high IDs (e.g., '{problematic_genes[0]}' -> ID {global_vocab.get(problematic_genes[0])}).\n"
                    f"    - Crucially, at least one of these is EXPRESSED."
                )
                crashing_files.append(file_path.name)
            else:
                logging.warning(
                    f"  [NO CRASH] This file works by coincidence.\n"
                    f"    - It has {local_gene_count} unique genes.\n"
                    f"    - It contains problematic genes with high IDs (e.g., '{problematic_genes[0]}' -> ID {global_vocab.get(problematic_genes[0])}).\n"
                    f"    - However, NONE of these genes are expressed (all counts are 0)."
                )
                safe_files.append(file_path.name)

        except Exception as e:
            logging.error(f"Could not process {file_path.name}. Error: {e}")

    print("-" * 70)
    logging.info("Diagnosis finished.")

    if crashing_files:
        print("\nSummary of files that WILL CRASH:")
        for fname in crashing_files:
            print(f"- {fname}")
    if safe_files:
        print("\nSummary of files that WORK BY COINCIDENCE ('ghost genes'):")
        for fname in safe_files:
            print(f"- {fname}")


if __name__ == "__main__":
    main()
