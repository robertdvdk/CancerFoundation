import torch
from enum import Enum
import sys
import types
import os
import scanpy as sc
from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection
import pandas as pd

sys.path.insert(0, "../")
from model.embedding import embed

if __name__ == "__main__":
    dataset = "neftel_ss2"
    CancerGPT_model_list = ["medium", "scale_bins", "condtech_done", "epoch_15"]
    # CancerGPT_model_list = [f"epoch_{i}" for i in range(1, 16)]
    print(CancerGPT_model_list)
    baseline_list = []
    for model in CancerGPT_model_list:
        if not os.path.exists(f"../model/assets/{model}/model.pth"):
            print("Converting model checkpoints to .pth format...")
            # Root package
            sys.modules.setdefault(
                "cancerfoundation", types.ModuleType("cancerfoundation")
            )

            # Submodule: cancerfoundation.loss with LossType enum
            class LossType(Enum):
                ORDINALCROSSENTROPY = "ordinal_cross_entropy"
                CORN = "corn"
                MSE = "mse"

            loss_mod = types.ModuleType("cancerfoundation.loss")
            sys.modules["cancerfoundation.loss"] = loss_mod
            setattr(loss_mod, "LossType", LossType)

            def extract_and_modify_weights(ckpt_path, output_pth_path):
                """
                Loads a PyTorch Lightning checkpoint, modifies the model's state_dict keys,
                and saves the result to a new .pth file in a single pass.

                Args:
                    ckpt_path (str): Path to the input checkpoint file (.ckpt).
                    output_pth_path (str): Path to save the modified model weights file (.pth).
                """
                # 1. Load the checkpoint and extract the model weights (state_dict)
                print(f"➡️ Loading checkpoint from: {ckpt_path}")
                checkpoint = torch.load(ckpt_path, map_location="cpu")
                original_state_dict = checkpoint["state_dict"]
                print("✅ Checkpoint loaded and state_dict extracted.")

                # 2. Modify the state_dict keys in-memory
                new_state_dict = {}
                print("⚙️  Modifying model state_dict keys...")
                for key, value in original_state_dict.items():
                    # First, remove the "model." prefix added by PyTorch Lightning
                    new_key = key.replace("model.", "", 1)

                    # Apply specific transformations to key names
                    if new_key.startswith("encoder") and not new_key.startswith(
                        "encoder.layers."
                    ):
                        # Rename "encoder." to "gene_encoder." for non-layer keys (e.g., cls_token)
                        new_key = "gene_encoder" + new_key[len("encoder") :]
                    elif "transformer_encoder" in new_key:
                        # Standardize "transformer_encoder" to just "encoder"
                        new_key = new_key.replace("transformer_encoder", "encoder")

                    new_state_dict[new_key] = value
                print("✅ Keys successfully modified.")

                # 3. Save the new, modified state_dict to the final output file
                print(f"💾 Saving modified model to: {output_pth_path}")
                torch.save(new_state_dict, output_pth_path)
                print(f"🎉 Successfully saved modified model to {output_pth_path}")

            files = os.listdir(f"../model/assets/{model}/")
            chkpt_file = None
            for file in files:
                if file.endswith(".ckpt"):
                    chkpt_file = file
            if chkpt_file:
                extract_and_modify_weights(
                    ckpt_path=f"../model/assets/{model}/{chkpt_file}",
                    output_pth_path=f"../model/assets/{model}/model.pth",
                )
            else:
                print(f"No checkpoint file found for model: {model}")

    for model in CancerGPT_model_list:
        # if not os.path.exists(f"./data/{dataset}/CancerGPT_{model}_{dataset}.h5ad"):
        print("Generating embeddings...")
        model_dir = f"../model/assets/{model}"
        adata_path = f"./data/{dataset}/{dataset}.h5ad"
        adata = sc.read_h5ad(adata_path)

        embed_adata = embed(
            adata_or_file=adata,
            model_dir=model_dir,
            batch_key="sample",
            batch_size=64,
            max_length=1200,
            scale_bins="scale_bins" in model,
        )
        os.makedirs(f"./data/{dataset}", exist_ok=True)
        embed_adata.write_h5ad(f"./data/{dataset}/CancerGPT_{model}_{dataset}.h5ad")

    if dataset == "neftel_ss2":
        label = "subtype"
    else:
        label = "celltype"

    all_results = pd.DataFrame()

    for _ in range(5):
        adata = sc.read_h5ad(
            f"./data/{dataset}/CancerGPT_{CancerGPT_model_list[0]}_{dataset}.h5ad"
        )
        for model in CancerGPT_model_list:
            adata.obsm[model] = sc.read_h5ad(
                f"./data/{dataset}/CancerGPT_{model}_{dataset}.h5ad"
            ).obsm["CancerGPT"]
        for model in baseline_list:
            if model.startswith("scGPT"):
                adata.obsm[model] = sc.read_h5ad(
                    f"./data/{dataset}/{model}_{dataset}.h5ad"
                ).obsm["X_scGPT"]
            elif model.startswith("scFoundation"):
                adata.obsm[model] = sc.read_h5ad(
                    f"./data/{dataset}/{model}_{dataset}.h5ad"
                ).obsm["scFoundation_embedding"]

        del adata.obsm["CancerGPT"]

        if dataset == "ji_skin":
            del adata.obsm["X_cnv"]
        all_keys = list(adata.obsm.keys())

        # results may vary slightly given differnet seeds
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
    grouped_results = all_results.groupby(all_results.index).describe().stack()
    print(grouped_results)
    grouped_results.to_csv(f"./data/{dataset}/batch_integration_evaluation.csv")
