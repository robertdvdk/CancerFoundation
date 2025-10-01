import sys

sys.path.insert(0, "../")
from model.embedding import embed
import scanpy as sc
import os

for model_name in [f"epoch_{i}" for i in range(1, 16)]:
    data_dir = "./data/gene_expression/"
    dataset = data_dir.split("/")[-2].lower()

    model_dir = "../model/assets/" + model_name
    adata_path = data_dir + dataset + ".h5ad"
    adata = sc.read_h5ad(adata_path)

    embed_adata = embed(
        adata_or_file=adata,
        model_dir=model_dir,
        batch_key="sample",
        batch_size=64,
        max_length=1200,
    )
    os.makedirs("./data", exist_ok=True)
    embed_adata.write_h5ad(
        data_dir + "CancerGPT_" + model_name + "_" + dataset + ".h5ad"
    )
