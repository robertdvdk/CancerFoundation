#!/usr/bin/env python3
"""
Fine-tuning Pre-trained Model for Perturbation Prediction

This script converts the Tutorial_Perturbation.ipynb notebook into a runnable script
that accepts a checkpoint path as a command-line argument.

Usage:
    python run_perturbation.py --checkpoint <path_to_checkpoint>
    python run_perturbation.py --checkpoint ../save/train_brain_base_7336129/epoch_epoch=49.ckpt
"""

import json
import sys
import time
import copy
import argparse
from pathlib import Path
from typing import Dict, Union
import warnings

import torch
import numpy as np
import matplotlib
from torch import nn

from torch_geometric.loader import DataLoader
from gears import PertData
from gears.inference import deeper_analysis, non_dropout_analysis
from gears.utils import create_cell_graph_dataset_for_prediction

sys.path.insert(0, "../")
from cancerfoundation.loss import masked_mse_loss
from anndata import AnnData
from cancerfoundation.model.model import CancerFoundation


def compute_perturbation_metrics(
    results: Dict,
    ctrl_adata: AnnData,
    non_zero_genes: bool = False,
    return_raw: bool = False,
) -> Dict:
    """
    Given results from a model run and the ground truth, compute metrics

    Args:
        results (:obj:`Dict`): The results from a model run
        ctrl_adata (:obj:`AnnData`): The adata of the control condtion
        non_zero_genes (:obj:`bool`, optional): Whether to only consider non-zero
            genes in the ground truth when computing metrics
        return_raw (:obj:`bool`, optional): Whether to return the raw metrics or
            the mean of the metrics. Default is False.

    Returns:
        :obj:`Dict`: The metrics computed
    """
    from scipy.stats import pearsonr

    # metrics:
    #   Pearson correlation of expression on all genes, on DE genes,
    #   Pearson correlation of expression change on all genes, on DE genes,

    metrics_across_genes = {
        "pearson": [],
        "pearson_de": [],
        "pearson_delta": [],
        "pearson_de_delta": [],
    }

    metrics_across_conditions = {
        "pearson": [],
        "pearson_delta": [],
    }

    conditions = np.unique(results["pert_cat"])
    assert "ctrl" not in conditions, "ctrl should not be in test conditions"
    condition2idx = {c: np.where(results["pert_cat"] == c)[0] for c in conditions}

    mean_ctrl = np.array(ctrl_adata.X.mean(0)).flatten()  # (n_genes,)
    assert ctrl_adata.X.max() <= 1000, "gene expression should be log transformed"

    true_perturbed = results["truth"]  # (n_cells, n_genes)
    assert true_perturbed.max() <= 1000, "gene expression should be log transformed"

    true_mean_perturbed_by_condition = np.array(
        [true_perturbed[condition2idx[c]].mean(0) for c in conditions]
    )  # (n_conditions, n_genes)
    true_mean_delta_by_condition = true_mean_perturbed_by_condition - mean_ctrl
    zero_rows = np.where(np.all(true_mean_perturbed_by_condition == 0, axis=1))[
        0
    ].tolist()
    # zero_cols = np.where(np.all(true_mean_perturbed_by_condition == 0, axis=0))[
    #     0
    # ].tolist()

    pred_perturbed = results["pred"]  # (n_cells, n_genes)
    pred_mean_perturbed_by_condition = np.array(
        [pred_perturbed[condition2idx[c]].mean(0) for c in conditions]
    )  # (n_conditions, n_genes)
    pred_mean_delta_by_condition = pred_mean_perturbed_by_condition - mean_ctrl

    def corr_over_genes(x, y, conditions, res_list, skip_rows=[], non_zero_mask=None):
        """compute pearson correlation over genes for each condition"""
        for i, c in enumerate(conditions):
            if i in skip_rows:
                continue
            x_, y_ = x[i], y[i]
            if non_zero_mask is not None:
                x_ = x_[non_zero_mask[i]]
                y_ = y_[non_zero_mask[i]]
            res_list.append(pearsonr(x_, y_)[0])

    corr_over_genes(
        true_mean_perturbed_by_condition,
        pred_mean_perturbed_by_condition,
        conditions,
        metrics_across_genes["pearson"],
        zero_rows,
        non_zero_mask=true_mean_perturbed_by_condition != 0 if non_zero_genes else None,
    )
    corr_over_genes(
        true_mean_delta_by_condition,
        pred_mean_delta_by_condition,
        conditions,
        metrics_across_genes["pearson_delta"],
        zero_rows,
        non_zero_mask=true_mean_perturbed_by_condition != 0 if non_zero_genes else None,
    )

    def find_DE_genes(adata, condition, geneid2idx, non_zero_genes=False, top_n=20):
        """
        Find the DE genes for a condition
        """
        key_components = next(
            iter(adata.uns["rank_genes_groups_cov_all"].keys())
        ).split("_")
        assert len(key_components) == 3, "rank_genes_groups_cov_all key is not valid"

        condition_key = "_".join([key_components[0], condition, key_components[2]])

        de_genes = adata.uns["rank_genes_groups_cov_all"][condition_key]
        if non_zero_genes:
            de_genes = adata.uns["top_non_dropout_de_20"][condition_key]

        de_genes = de_genes[:top_n]

        de_idx = [geneid2idx[i] for i in de_genes]

        return de_idx, de_genes

    geneid2idx = dict(zip(ctrl_adata.var.index.values, range(len(ctrl_adata.var))))
    de_idx = {
        c: find_DE_genes(ctrl_adata, c, geneid2idx, non_zero_genes)[0]
        for c in conditions
    }
    mean_ctrl_de = np.array(
        [mean_ctrl[de_idx[c]] for c in conditions]
    )  # (n_conditions, n_diff_genes)

    true_mean_perturbed_by_condition_de = np.array(
        [
            true_mean_perturbed_by_condition[i, de_idx[c]]
            for i, c in enumerate(conditions)
        ]
    )  # (n_conditions, n_diff_genes)
    zero_rows_de = np.where(np.all(true_mean_perturbed_by_condition_de == 0, axis=1))[
        0
    ].tolist()
    true_mean_delta_by_condition_de = true_mean_perturbed_by_condition_de - mean_ctrl_de

    pred_mean_perturbed_by_condition_de = np.array(
        [
            pred_mean_perturbed_by_condition[i, de_idx[c]]
            for i, c in enumerate(conditions)
        ]
    )  # (n_conditions, n_diff_genes)
    pred_mean_delta_by_condition_de = pred_mean_perturbed_by_condition_de - mean_ctrl_de

    corr_over_genes(
        true_mean_perturbed_by_condition_de,
        pred_mean_perturbed_by_condition_de,
        conditions,
        metrics_across_genes["pearson_de"],
        zero_rows_de,
    )
    corr_over_genes(
        true_mean_delta_by_condition_de,
        pred_mean_delta_by_condition_de,
        conditions,
        metrics_across_genes["pearson_de_delta"],
        zero_rows_de,
    )

    if not return_raw:
        for k, v in metrics_across_genes.items():
            metrics_across_genes[k] = np.mean(v)
        for k, v in metrics_across_conditions.items():
            metrics_across_conditions[k] = np.mean(v)
    metrics = metrics_across_genes

    return metrics


def map_raw_id_to_vocab_id(
    raw_ids: Union[np.ndarray, torch.Tensor],
    gene_ids: np.ndarray,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Map some raw ids which are indices of the raw gene names to the indices of the

    Args:
        raw_ids: the raw ids to map
        gene_ids: the gene ids to map to
    """
    if isinstance(raw_ids, torch.Tensor):
        device = raw_ids.device
        dtype = raw_ids.dtype
        return_pt = True
        raw_ids = raw_ids.cpu().numpy()
    elif isinstance(raw_ids, np.ndarray):
        return_pt = False
        dtype = raw_ids.dtype
    else:
        raise ValueError("raw_ids must be either torch.Tensor or np.ndarray.")

    if raw_ids.ndim != 1:
        raise ValueError(f"raw_ids must be 1d, got {raw_ids.ndim}d.")

    if gene_ids.ndim != 1:
        raise ValueError(f"gene_ids must be 1d, got {gene_ids.ndim}d.")

    mapped_ids: np.ndarray = gene_ids[raw_ids]
    assert mapped_ids.shape == raw_ids.shape
    if return_pt:
        return torch.from_numpy(mapped_ids).type(dtype).to(device)
    return mapped_ids.astype(dtype)


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    scaler,
    criterion,
    device,
    n_genes,
    gene_ids,
    pad_token,
    vocab,
    include_zero_gene,
    max_seq_len,
    amp,
    log_interval,
    epoch,
    scheduler,
):
    """
    Train the model for one epoch.
    """
    model.train()
    total_loss, total_mse = 0.0, 0.0
    start_time = time.time()

    num_batches = len(train_loader)
    for batch, batch_data in enumerate(train_loader):
        batch_size = len(batch_data.y)
        batch_data.to(device)
        x: torch.Tensor = batch_data.x  # (batch_size * n_genes, 2)
        ori_gene_values = x[:, 0].view(batch_size, n_genes)
        pert_flags = x[:, 1].long().view(batch_size, n_genes)
        target_gene_values = batch_data.y  # (batch_size, n_genes)

        if include_zero_gene in ["all", "batch-wise"]:
            if include_zero_gene == "all":
                input_gene_ids = torch.arange(n_genes, device=device, dtype=torch.long)
            else:
                input_gene_ids = (
                    ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]
                )
            # sample input_gene_id
            if len(input_gene_ids) > max_seq_len:
                input_gene_ids = torch.randperm(len(input_gene_ids), device=device)[
                    :max_seq_len
                ]
            input_values = ori_gene_values[:, input_gene_ids]
            input_pert_flags = pert_flags[:, input_gene_ids]
            target_values = target_gene_values[:, input_gene_ids]

            mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, gene_ids)
            mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)

            src_key_padding_mask = torch.zeros_like(
                input_values, dtype=torch.bool, device=device
            )

        tens = {
            "gene": mapped_input_gene_ids,
            "masked_expr": input_values,
            "expr": target_values,
            "pert_flags": input_pert_flags,
            "src_key_padding_mask": src_key_padding_mask,
        }

        with torch.cuda.amp.autocast(enabled=amp):
            output_dict = model(tens)
            output_values = output_dict["pred"]

            masked_positions = torch.ones_like(
                input_values, dtype=torch.bool
            )  # Use all
            loss = loss_mse = criterion(output_values, target_values, masked_positions)

        model.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always")
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
                error_if_nonfinite=False if scaler.is_enabled() else True,
            )
            if len(w) > 0:
                print(
                    f"Found infinite gradient. This may be caused by the gradient "
                    f"scaler. The current scale is {scaler.get_scale()}. This warning "
                    "can be ignored if no longer occurs after autoscaling of the scaler."
                )
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_mse += loss_mse.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            cur_mse = total_mse / log_interval
            print(
                f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                f"lr {lr:05.4f} | ms/batch {ms_per_batch:5.2f} | "
                f"loss {cur_loss:5.2f} | mse {cur_mse:5.2f} |"
            )
            total_loss = 0
            total_mse = 0
            start_time = time.time()


def eval_perturb(loader, model, device, n_genes, gene_ids, include_zero_gene):
    """
    Run model in inference mode using a given data loader
    """
    model.eval()
    model.to(device)
    pert_cat = []
    pred = []
    truth = []
    pred_de = []
    truth_de = []
    results = {}

    for itr, batch in enumerate(loader):
        pert_cat.extend(batch.pert)
        batch_size = len(batch.y)
        batch.to(device)
        x: torch.Tensor = batch.x  # (batch_size * n_genes, 2)
        ori_gene_values = x[:, 0].view(batch_size, n_genes)
        pert_flags = x[:, 1].long().view(batch_size, n_genes)

        if include_zero_gene in ["all", "batch-wise"]:
            if include_zero_gene == "all":
                input_gene_ids = torch.arange(n_genes, device=device, dtype=torch.long)
            else:
                input_gene_ids = (
                    ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]
                )
            input_values = ori_gene_values[:, input_gene_ids]
            input_pert_flags = pert_flags[:, input_gene_ids]

            mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, gene_ids)
            mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)

            src_key_padding_mask = torch.zeros_like(
                input_values, dtype=torch.bool, device=device
            )
        with torch.no_grad():
            tens = {
                "gene": mapped_input_gene_ids,
                "masked_expr": input_values,
                "expr": input_values,
                "pert_flags": input_pert_flags,
                "src_key_padding_mask": src_key_padding_mask,
            }
            p = model.model(tens)["pred"]
            t = batch.y
            pred.extend(p.cpu())
            truth.extend(t.cpu())

            # Differentially expressed genes
            for itr, de_idx in enumerate(batch.de_idx):
                pred_de.append(p[itr, de_idx])
                truth_de.append(t[itr, de_idx])

    # all genes
    results["pert_cat"] = np.array(pert_cat)
    pred = torch.stack(pred)
    truth = torch.stack(truth)
    results["pred"] = pred.detach().cpu().numpy().astype(float)
    results["truth"] = truth.detach().cpu().numpy().astype(float)

    pred_de = torch.stack(pred_de)
    truth_de = torch.stack(truth_de)
    results["pred_de"] = pred_de.detach().cpu().numpy().astype(float)
    results["truth_de"] = truth_de.detach().cpu().numpy().astype(float)

    return results


def plot_perturbation(
    model: nn.Module,
    query: str,
    pert_data,
    gene2idx,
    gene_ids,
    device,
    include_zero_gene,
    eval_batch_size,
    n_genes,
    save_file: str = None,
    pool_size: int = None,
) -> matplotlib.figure.Figure:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="ticks", rc={"axes.facecolor": (0, 0, 0, 0)}, font_scale=1.5)

    adata = pert_data.adata
    cond2name = dict(adata.obs[["condition", "condition_name"]].values)
    gene_raw2id = dict(zip(adata.var.index.values, adata.var.gene_name.values))

    de_idx = [
        gene2idx[gene_raw2id[i]]
        for i in adata.uns["top_non_dropout_de_20"][cond2name[query]]
    ]
    genes = [
        gene_raw2id[i] for i in adata.uns["top_non_dropout_de_20"][cond2name[query]]
    ]
    truth = adata[adata.obs.condition == query].X.toarray()[:, de_idx]

    # Prediction logic (simplified - you may need to implement predict function)
    ctrl_adata = adata[adata.obs["condition"] == "ctrl"]
    if pool_size is None:
        pool_size = len(ctrl_adata.obs)

    gene_list = pert_data.gene_names.values.tolist()
    if query.split("+")[1] == "ctrl":
        pert = [query.split("+")[0]]
    else:
        pert = query.split("+")

    cell_graphs = create_cell_graph_dataset_for_prediction(
        pert, ctrl_adata, gene_list, device, num_samples=pool_size
    )
    loader = DataLoader(cell_graphs, batch_size=eval_batch_size, shuffle=False)
    preds = []
    model.eval()
    with torch.no_grad():
        for batch_data in loader:
            batch_size = len(batch_data.y)
            batch_data.to(device)
            x: torch.Tensor = batch_data.x
            ori_gene_values = x[:, 0].view(batch_size, n_genes)
            pert_flags = x[:, 1].long().view(batch_size, n_genes)

            if include_zero_gene == "all":
                input_gene_ids = torch.arange(n_genes, device=device, dtype=torch.long)
            else:
                input_gene_ids = (
                    ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]
                )
            input_values = ori_gene_values[:, input_gene_ids]
            input_pert_flags = pert_flags[:, input_gene_ids]

            mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, gene_ids)
            mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)

            src_key_padding_mask = torch.zeros_like(
                input_values, dtype=torch.bool, device=device
            )

            tens = {
                "gene": mapped_input_gene_ids,
                "masked_expr": input_values,
                "expr": input_values,
                "pert_flags": input_pert_flags,
                "src_key_padding_mask": src_key_padding_mask,
            }
            pred_gene_values = model.model(tens)["pred"]
            preds.append(pred_gene_values)
    preds = torch.cat(preds, dim=0)
    pred = np.mean(preds.detach().cpu().numpy(), axis=0)[de_idx]

    ctrl_means = adata[adata.obs["condition"] == "ctrl"].to_df().mean()[de_idx].values

    pred = pred - ctrl_means
    truth = truth - ctrl_means

    fig, ax = plt.subplots(figsize=[16.5, 4.5])
    plt.title(query)
    plt.boxplot(truth, showfliers=False, medianprops=dict(linewidth=0))

    for i in range(pred.shape[0]):
        _ = plt.scatter(i + 1, pred[i], color="red")

    plt.axhline(0, linestyle="dashed", color="green")

    ax.xaxis.set_ticklabels(genes, rotation=90)

    plt.ylabel("Change in Gene Expression over Control", labelpad=10)
    plt.tick_params(axis="x", which="major", pad=5)
    plt.tick_params(axis="y", which="major", pad=5)
    sns.despine()

    if save_file:
        fig.savefig(save_file, bbox_inches="tight", transparent=False)

    return fig


def reinitialize_weights(m: nn.Module):
    """
    This function re-initializes the weights of a model's layers.
    It's designed to be used with the model.apply() method.
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


def main():
    parser = argparse.ArgumentParser(description="Run perturbation prediction")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the checkpoint file to load",
    )
    parser.add_argument(
        "--data-name",
        type=str,
        default="adamson",
        choices=["adamson", "norman"],
        help="Dataset name (default: adamson)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="simulation",
        help="Data split type (default: simulation)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Training batch size (default: 16)"
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=4,
        help="Evaluation batch size (default: 4)",
    )
    parser.add_argument(
        "--epochs", type=int, default=15, help="Number of training epochs (default: 15)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.2, help="Dropout rate (default: 0.2)"
    )
    parser.add_argument(
        "--early-stop",
        type=int,
        default=10,
        help="Early stopping patience (default: 10)",
    )
    parser.add_argument(
        "--log-interval", type=int, default=100, help="Logging interval (default: 100)"
    )
    parser.add_argument(
        "--no-amp", action="store_true", help="Disable automatic mixed precision"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save results (default: auto-generated)",
    )
    parser.add_argument(
        "--reinit-weights",
        action="store_true",
        help="Reinitialize model weights before training",
    )

    args = parser.parse_args()

    # Settings for data processing
    pad_token = "<pad>"
    special_tokens = [pad_token, "<cls>"]
    # pad_value = 1
    # pert_pad_id = 0
    include_zero_gene = "all"
    max_seq_len = 1200

    # Training settings
    # MLM = True
    # CLS = False
    # CCE = False
    # MVC = False
    # ECS = False
    amp = not args.no_amp

    # Dataset-specific plotting settings
    if args.data_name == "norman":
        perts_to_plot = ["SAMD1+ZBTB1"]
    elif args.data_name == "adamson":
        perts_to_plot = ["KCTD16+ctrl"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create save directory
    if args.save_dir is None:
        save_dir = Path(
            f"./save/dev_perturb_{args.data_name}-{time.strftime('%b%d-%H-%M')}/"
        )
    else:
        save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {save_dir}")

    # Log running info
    print(f"Running on {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Loading checkpoint from: {args.checkpoint}")

    # Load data
    print(f"Loading {args.data_name} dataset...")
    pert_data = PertData("./data", default_pert_graph=False)
    pert_data.load(data_name=args.data_name)
    pert_data.prepare_split(split=args.split, seed=1)
    pert_data.get_dataloader(
        batch_size=args.batch_size, test_batch_size=args.eval_batch_size
    )

    # Load vocabulary
    checkpoint_dir = Path(args.checkpoint).parent
    vocab_file = checkpoint_dir / "vocab.json"

    if vocab_file.exists():
        with open(vocab_file, "r") as f:
            vocab = json.load(f)
        for s in special_tokens:
            if s not in vocab:
                vocab[s] = len(vocab)
    else:
        print(f"Warning: vocab.json not found at {vocab_file}")
        genes = pert_data.adata.var["gene_name"].tolist()
        vocab = {gene: idx for idx, gene in enumerate(genes + special_tokens)}

    # Match genes in vocabulary
    pert_data.adata.var["id_in_vocab"] = [
        1 if gene in vocab else -1 for gene in pert_data.adata.var["gene_name"]
    ]
    gene_ids_in_vocab = np.array(pert_data.adata.var["id_in_vocab"])
    print(
        f"Match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
        f"in vocabulary of size {len(vocab)}."
    )
    genes = pert_data.adata.var["gene_name"].tolist()
    gene_ids = np.array(
        [vocab[gene] if gene in vocab else vocab["<pad>"] for gene in genes], dtype=int
    )
    n_genes = len(genes)
    gene2idx = pert_data.node_map

    # Load model
    print(f"Loading model from checkpoint: {args.checkpoint}")
    model = CancerFoundation.load_from_checkpoint(
        args.checkpoint,
        vocab=vocab,
        perturbation=True,
        strict=False,
        their_init_weights=False,
        compile_model=False,
        dropout=args.dropout,
    )
    # ntokens = len(vocab)

    # Reinitialize weights if requested
    if args.reinit_weights:
        print("Reinitializing model weights...")
        model.apply(reinitialize_weights)
    else:
        print("Using pretrained weights from checkpoint")
    model.to(device)

    # Setup training
    criterion = masked_mse_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    # Training loop
    print("Starting training...")
    # best_val_loss = float("inf")
    best_val_corr = 0
    best_model = None
    patience = 0

    train_loader = pert_data.dataloader["train_loader"]
    q = next(iter(train_loader))
    print(q.x[:, 0].min(), q.x[:, 0].max(), q.x[:, 0].mean(), q.x[:, 0])
    if q.x.shape[1] > 1:
        print(q.x[:, 1].min(), q.x[:, 1].max(), q.x[:, 1].mean(), q.x[:, 1])
    else:
        print(q.pert.min(), q.pert.max(), q.pert.mean(), q.pert)
    print(q.y.min(), q.y.max(), q.y.mean(), q.y)

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train_loader = pert_data.dataloader["train_loader"]
        valid_loader = pert_data.dataloader["val_loader"]

        train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            criterion,
            device,
            n_genes,
            gene_ids,
            pad_token,
            vocab,
            include_zero_gene,
            max_seq_len,
            amp,
            args.log_interval,
            epoch,
            scheduler,
        )

        val_res = eval_perturb(
            valid_loader, model, device, n_genes, gene_ids, include_zero_gene
        )
        val_metrics = compute_perturbation_metrics(
            val_res, pert_data.adata[pert_data.adata.obs["condition"] == "ctrl"]
        )

        print(f"Val metrics at epoch {epoch}: ")
        print(val_metrics)

        elapsed = time.time() - epoch_start_time
        print(f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | ")

        val_score = val_metrics["pearson"]
        if val_score > best_val_corr:
            best_val_corr = val_score
            best_model = copy.deepcopy(model)
            print(f"Best model with score {val_score:5.4f}")
            patience = 0
        else:
            patience += 1
            if patience >= args.early_stop:
                print(f"Early stop at epoch {epoch}")
                break

        scheduler.step()

    # Save best model
    print("Saving best model...")
    torch.save(best_model.state_dict(), save_dir / "best_model.pt")

    # Generate plots
    print("Generating perturbation plots...")
    for p in perts_to_plot:
        plot_perturbation(
            best_model,
            p,
            pert_data,
            gene2idx,
            gene_ids,
            device,
            include_zero_gene,
            args.eval_batch_size,
            n_genes,
            pool_size=300,
            save_file=f"{save_dir}/{p}.png",
        )

    # Evaluate on test set
    print("Evaluating on test set...")
    test_loader = pert_data.dataloader["test_loader"]
    test_res = eval_perturb(
        test_loader, best_model, device, n_genes, gene_ids, include_zero_gene
    )
    test_metrics = compute_perturbation_metrics(
        test_res, pert_data.adata[pert_data.adata.obs["condition"] == "ctrl"]
    )
    print("Test metrics:")
    print(test_metrics)

    # Save test metrics
    with open(f"{save_dir}/test_metrics.json", "w") as f:
        json.dump(test_metrics, f)

    # Deeper analysis
    print("Running deeper analysis...")
    deeper_res = deeper_analysis(pert_data.adata, test_res)
    non_dropout_res = non_dropout_analysis(pert_data.adata, test_res)

    metrics = ["pearson_delta", "pearson_delta_de"]
    metrics_non_dropout = [
        "pearson_delta_top20_de_non_dropout",
        "pearson_top20_de_non_dropout",
    ]
    subgroup_analysis = {}
    for name in pert_data.subgroup["test_subgroup"].keys():
        subgroup_analysis[name] = {}
        for m in metrics:
            subgroup_analysis[name][m] = []
        for m in metrics_non_dropout:
            subgroup_analysis[name][m] = []

    for name, pert_list in pert_data.subgroup["test_subgroup"].items():
        for pert in pert_list:
            for m in metrics:
                subgroup_analysis[name][m].append(deeper_res[pert][m])
            for m in metrics_non_dropout:
                subgroup_analysis[name][m].append(non_dropout_res[pert][m])

    print("\nSubgroup analysis:")
    for name, result in subgroup_analysis.items():
        for m in result.keys():
            mean_value = np.mean(subgroup_analysis[name][m])
            print(f"test_{name}_{m}: {mean_value}")

    print(f"\nAll results saved to {save_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
