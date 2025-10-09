import numpy as np
import torch
import sys
import os
import scanpy as sc
from scib_metrics.benchmark import BioConservation, BatchCorrection, Benchmarker
import pandas as pd
import json
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from utils import Dataset, DataCollator

sys.path.insert(0, "../")
from cancerfoundation.model.model import CancerFoundation


if __name__ == "__main__":
    dataset_name = "neftel_ss2"
    CancerGPT_model_list = [
        "train_brain_base_theirvalenc_theirgeneflag_941885",
    ]
    # CancerGPT_model_list = [f"epoch_{i}" for i in range(1, 16)]
    baseline_list = []
    for model in CancerGPT_model_list:
        for file in os.listdir(f"../save/{model}/"):
            if file.endswith(".ckpt"):
                chkpt_file = file
        with open(f"../save/{model}/vocab.json", "r") as f:
            vocab = json.load(f)
        trained_model = (
            CancerFoundation.load_from_checkpoint(
                f"../save/{model}/{chkpt_file}", vocab=vocab
            )
            .to("cuda")
            .eval()
        )

        adata_path = f"./data/{dataset_name}/{dataset_name}.h5ad"
        adata = sc.read_h5ad(adata_path)

        adata.var["genes"] = adata.var.index

        adata.var["id_in_vocab"] = [
            vocab[gene] if gene in vocab else -1 for gene in adata.var["genes"]
        ]
        adata = adata[:, adata.var["id_in_vocab"] >= 0]

        if adata.n_vars > trained_model.max_seq_len:
            sc.pp.highly_variable_genes(
                adata,
                n_top_genes=trained_model.max_seq_len,
                flavor="cell_ranger",
                batch_key="sample",
            )
            adata = adata[:, adata.var.highly_variable]
        adata.var["genes"] = adata.var.index

        pad_id = vocab["<pad>"]
        genes = adata.var["genes"].tolist()
        gene_ids = np.array([vocab.get(gene, pad_id) for gene in genes], dtype=int)
        # gene_ids = np.array(vocab(genes), dtype=int)
        count_matrix = adata.X
        count_matrix = (
            count_matrix
            if isinstance(count_matrix, np.ndarray)
            else count_matrix.toarray()
        )

        dataset = Dataset(count_matrix, gene_ids, vocab, pad_value=-2)
        collator = DataCollator(
            do_padding=True,
            pad_token_id=vocab["<pad>"],
            pad_value=-2,
            do_mlm=False,
            do_binning=True,
            max_length=trained_model.max_seq_len + 2,
            sampling=True,
            keep_first_n_tokens=1,
            scale_bins=True,
        )
        data_loader = DataLoader(
            dataset,
            batch_size=64,
            sampler=SequentialSampler(dataset),
            collate_fn=collator,
            drop_last=False,
            num_workers=min(len(os.sched_getaffinity(0)), 64),
            pin_memory=True,
        )
        device = next(trained_model.parameters()).device
        cell_embeddings = np.zeros(
            (len(dataset), trained_model.embsize), dtype=np.float32
        )
        normalize = True
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
            count = 0
            for data_dict in tqdm(data_loader, desc="Embedding cells"):
                input_gene_ids = data_dict["gene"].to(device)
                src_key_padding_mask = input_gene_ids.eq(pad_id)
                embeddings = trained_model.model.embed(
                    input_gene_ids,
                    data_dict["expr"].to(device),
                    src_key_padding_mask=src_key_padding_mask,
                    conditions=torch.full(
                        (input_gene_ids.size(0),), 2, dtype=torch.long, device=device
                    ),
                )[0]
                # get the <cls> position embedding
                embeddings = embeddings[:, 0, :]
                embeddings = embeddings.cpu().numpy()
                cell_embeddings[count : count + len(embeddings)] = embeddings
                count += len(embeddings)

        if normalize:
            cell_embeddings = cell_embeddings / np.linalg.norm(
                cell_embeddings, axis=1, keepdims=True
            )
        print(cell_embeddings)

        adata.obsm["CancerGPT"] = cell_embeddings
        adata.write_h5ad(f"./data/{dataset_name}/CancerGPT_{model}_{dataset_name}.h5ad")

    if dataset_name == "neftel_ss2":
        label = "subtype"
    else:
        label = "celltype"

    all_results = pd.DataFrame()

    for _ in range(1):
        adata = sc.read_h5ad(
            f"./data/{dataset_name}/CancerGPT_{CancerGPT_model_list[0]}_{dataset_name}.h5ad"
        )
        for model in CancerGPT_model_list:
            adata.obsm[model] = sc.read_h5ad(
                f"./data/{dataset_name}/CancerGPT_{model}_{dataset_name}.h5ad"
            ).obsm["CancerGPT"]
        for model in baseline_list:
            if model.startswith("scGPT"):
                adata.obsm[model] = sc.read_h5ad(
                    f"./data/{dataset_name}/{model}_{dataset_name}.h5ad"
                ).obsm["X_scGPT"]
            elif model.startswith("scFoundation"):
                adata.obsm[model] = sc.read_h5ad(
                    f"./data/{dataset_name}/{model}_{dataset_name}.h5ad"
                ).obsm["scFoundation_embedding"]

        del adata.obsm["CancerGPT"]

        if dataset_name == "ji_skin":
            del adata.obsm["X_cnv"]
        all_keys = list(adata.obsm.keys())

        bio_conservation = BioConservation(
            nmi_ari_cluster_labels_kmeans=False, nmi_ari_cluster_labels_leiden=True
        )
        batch_correction = BatchCorrection(pcr_comparison=False)

        bm = Benchmarker(
            adata,
            batch_key="sample",
            label_key=label,
            embedding_obsm_keys=all_keys,
            n_jobs=6,
            bio_conservation_metrics=bio_conservation,
            batch_correction_metrics=batch_correction,
        )
        bm.benchmark()
        curr_result = bm.get_results(min_max_scale=False)
        curr_result = curr_result[curr_result.index != "Metric Type"]
        all_results = pd.concat([all_results, curr_result])
    all_results = all_results.apply(pd.to_numeric, errors="coerce")
    print(all_results)
    all_results.to_csv(f"./data/{dataset_name}/batch_integration_evaluation.csv")
