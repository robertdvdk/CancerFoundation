import json
import sys
import time
import copy
import argparse
from pathlib import Path
from typing import List, Dict, Union, Optional
import warnings

import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn
from torch_geometric.loader import DataLoader
from gears import PertData
from gears.inference import deeper_analysis, non_dropout_analysis
from gears.utils import create_cell_graph_dataset_for_prediction
from anndata import AnnData

sys.path.insert(0, "../")

from cancerfoundation.model.model import CancerFoundation
from cancerfoundation.loss import masked_mse_loss


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def map_raw_id_to_vocab_id(
    raw_ids: Union[np.ndarray, torch.Tensor],
    gene_ids: np.ndarray,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Map some raw ids which are indices of the raw gene names to the indices of the
    vocabulary.

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


matplotlib.rcParams["savefig.transparent"] = False
warnings.filterwarnings("ignore")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Gene Perturbation Prediction")

    # Data settings
    parser.add_argument(
        "--data-name",
        type=str,
        default="adamson",
        choices=["adamson", "norman"],
        help="Dataset name",
    )
    parser.add_argument(
        "--data-dir", type=str, default="./data", help="Directory containing the data"
    )
    parser.add_argument(
        "--split", type=str, default="simulation", help="Data split strategy"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Model settings
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file (alternative to load_model)",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.0, help="Dropout probability"
    )

    # Training settings
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Training batch size"
    )
    parser.add_argument(
        "--eval-batch-size", type=int, default=32, help="Evaluation batch size"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--epochs", type=int, default=15, help="Number of training epochs"
    )
    parser.add_argument(
        "--schedule-interval", type=int, default=1, help="Scheduler step interval"
    )
    parser.add_argument(
        "--early-stop", type=int, default=10, help="Early stopping patience"
    )
    parser.add_argument(
        "--amp", action="store_true", default=True, help="Use automatic mixed precision"
    )

    # Data processing settings
    parser.add_argument(
        "--include-zero-gene",
        type=str,
        default="all",
        choices=["all", "batch-wise"],
        help="How to handle zero-valued genes",
    )
    parser.add_argument(
        "--max-seq-len", type=int, default=1536, help="Maximum sequence length"
    )

    # Logging and output
    parser.add_argument(
        "--log-interval", type=int, default=100, help="Logging interval"
    )
    parser.add_argument(
        "--save-dir", type=str, default=None, help="Directory to save results"
    )
    parser.add_argument(
        "--plot-pool-size",
        type=int,
        default=300,
        help="Pool size for plotting perturbations",
    )

    args = parser.parse_args()
    return args


def setup_logging(save_dir: Path):
    """Setup logging configuration."""
    import logging

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Create file handler
    fh = logging.FileHandler(save_dir / "run.log")
    fh.setLevel(logging.INFO)

    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # Add handlers
    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"Running on {time.strftime('%Y-%m-%d %H:%M:%S')}")
    return logger


def load_data(args, logger) -> PertData:
    """Load and prepare perturbation data."""
    logger.info(f"Loading data from {args.data_dir}")
    pert_data = PertData(args.data_dir)
    pert_data.load(data_name=args.data_name)
    pert_data.prepare_split(
        split=args.split, seed=42
    )  # We always keep the seed 42, because we want the test set to be the same for every run
    pert_data.get_dataloader(
        batch_size=args.batch_size, test_batch_size=args.eval_batch_size
    )
    logger.info("Data loaded successfully")
    return pert_data


def setup_vocabulary(args, pert_data, logger):
    """Setup vocabulary from pretrained model or create new one."""
    pad_token = "<pad>"
    special_tokens = [pad_token, "<cls>"]

    if args.load_model is not None:
        model_dir = Path(args.load_model)
        vocab_file = model_dir / "vocab.json"

        # Load vocabulary from JSON file
        if vocab_file.exists():
            with open(vocab_file, "r") as f:
                vocab = json.load(f)
            for s in special_tokens:
                if s not in vocab:
                    vocab[s] = len(vocab)
        else:
            logger.warning(
                f"vocab.json not found at {vocab_file}, creating new vocabulary"
            )
            genes = pert_data.adata.var["gene_name"].tolist()
            vocab = {gene: idx for idx, gene in enumerate(genes + special_tokens)}

        pert_data.adata.var["id_in_vocab"] = [
            1 if gene in vocab else -1 for gene in pert_data.adata.var["gene_name"]
        ]
        gene_ids_in_vocab = np.array(pert_data.adata.var["id_in_vocab"])
        logger.info(
            f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
            f"in vocabulary of size {len(vocab)}."
        )
        genes = pert_data.adata.var["gene_name"].tolist()

        # Return checkpoint path for model loading
        checkpoint_file = model_dir / "best_model.pt"
        if not checkpoint_file.exists():
            # Look for .ckpt files (PyTorch Lightning checkpoints)
            ckpt_files = list(model_dir.glob("*.ckpt"))
            if ckpt_files:
                checkpoint_file = ckpt_files[0]
                logger.info(f"Using checkpoint file: {checkpoint_file}")
            else:
                checkpoint_file = None
                logger.warning(f"No checkpoint found in {model_dir}")

        return vocab, genes, checkpoint_file
    else:
        genes = pert_data.adata.var["gene_name"].tolist()
        vocab = {gene: idx for idx, gene in enumerate(genes + special_tokens)}
        return vocab, genes, None


def create_model(args, vocab, genes, checkpoint_file=None):
    """Create and initialize the CancerFoundation model."""
    gene_ids = np.array(
        [vocab[gene] if gene in vocab else vocab["<pad>"] for gene in genes], dtype=int
    )
    n_genes = len(genes)

    # If checkpoint_file exists, load from checkpoint
    if checkpoint_file is not None and checkpoint_file.exists():
        logger_msg = f"Loading model from checkpoint: {checkpoint_file}"
        print(logger_msg)

        # Load checkpoint
        chkpt = torch.load(checkpoint_file, weights_only=False)

        # Check if keys have _orig_mod prefix (from compiled models)
        if "state_dict" in chkpt:
            first_key = list(chkpt["state_dict"].keys())[0]
            if "_orig_mod" in first_key:
                print("Detected compiled model checkpoint. Remapping keys...")
                new_state_dict = {}
                for key in chkpt["state_dict"].keys():
                    new_key = key.replace("._orig_mod", "")
                    new_state_dict[new_key] = chkpt["state_dict"][key]
                chkpt["state_dict"] = new_state_dict

                # Save the modified checkpoint temporarily
                import tempfile

                with tempfile.NamedTemporaryFile(
                    mode="wb", suffix=".ckpt", delete=False
                ) as tmp:
                    temp_checkpoint_path = tmp.name
                    torch.save(chkpt, tmp)

                try:
                    model = CancerFoundation.load_from_checkpoint(
                        temp_checkpoint_path,
                        vocab=vocab,
                        perturbation=True,
                        strict=False,
                        their_init_weights=False,
                        compile_model=False,
                        dropout=args.dropout,
                    )

                    # Verify which keys matched/didn't match
                    model_keys = set(model.state_dict().keys())
                    checkpoint_keys = set(new_state_dict.keys())

                    missing_in_checkpoint = model_keys - checkpoint_keys
                    unexpected_in_checkpoint = checkpoint_keys - model_keys

                    if missing_in_checkpoint:
                        print(
                            f"\nWarning: {len(missing_in_checkpoint)} keys in model but not in checkpoint:"
                        )
                        for key in sorted(
                            list(missing_in_checkpoint)[:10]
                        ):  # Show first 10
                            print(f"  - {key}")
                        if len(missing_in_checkpoint) > 10:
                            print(f"  ... and {len(missing_in_checkpoint) - 10} more")

                    if unexpected_in_checkpoint:
                        print(
                            f"\nWarning: {len(unexpected_in_checkpoint)} keys in checkpoint but not in model:"
                        )
                        for key in sorted(
                            list(unexpected_in_checkpoint)[:10]
                        ):  # Show first 10
                            print(f"  - {key}")
                        if len(unexpected_in_checkpoint) > 10:
                            print(
                                f"  ... and {len(unexpected_in_checkpoint) - 10} more"
                            )

                    if not missing_in_checkpoint and not unexpected_in_checkpoint:
                        print("\n✓ All keys matched perfectly after remapping!")
                    else:
                        matched_keys = len(model_keys & checkpoint_keys)
                        total_keys = len(model_keys)
                        print(
                            f"\n✓ {matched_keys}/{total_keys} keys matched successfully"
                        )

                    print("Model loaded successfully with remapped keys!")
                finally:
                    # Clean up temporary file
                    import os

                    os.unlink(temp_checkpoint_path)
            else:
                # No remapping needed
                model = CancerFoundation.load_from_checkpoint(
                    str(checkpoint_file),
                    vocab=vocab,
                    perturbation=True,
                    strict=True,
                    their_init_weights=False,
                    compile_model=False,
                    dropout=args.dropout,
                )
                print("Model loaded successfully!")
        else:
            raise ValueError("Checkpoint does not contain 'state_dict' key")

    else:
        model = None

    # If no model loaded from checkpoint, we would need to create one
    # But for perturbation task, we need a pretrained model
    if model is None:
        raise ValueError(
            "No checkpoint provided or failed to load. "
            "For perturbation prediction, a pretrained model is required. "
            "Please provide --load_model argument."
        )

    return model, gene_ids, n_genes


def train(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer,
    scheduler,
    scaler,
    criterion,
    args,
    gene_ids,
    n_genes,
    vocab,
    device,
    logger,
    epoch,
) -> None:
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

        if args.include_zero_gene in ["all", "batch-wise"]:
            if args.include_zero_gene == "all":
                input_gene_ids = torch.arange(n_genes, device=device, dtype=torch.long)
            else:
                input_gene_ids = (
                    ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]
                )
            # sample input_gene_id
            if len(input_gene_ids) > args.max_seq_len:
                input_gene_ids = torch.randperm(len(input_gene_ids), device=device)[
                    : args.max_seq_len
                ]
            input_values = ori_gene_values[:, input_gene_ids]
            input_pert_flags = pert_flags[:, input_gene_ids]
            target_values = target_gene_values[:, input_gene_ids]

            mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, gene_ids)
            mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)

            src_key_padding_mask = torch.zeros_like(
                input_values, dtype=torch.bool, device=device
            )

        # Prepare tensors for CancerFoundation model
        tens = {
            "gene": mapped_input_gene_ids,
            "masked_expr": input_values,
            "expr": target_values,
            "pert_flags": input_pert_flags,
            "src_key_padding_mask": src_key_padding_mask,
        }

        with torch.cuda.amp.autocast(enabled=args.amp):
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
                logger.warning(
                    f"Found infinite gradient. This may be caused by the gradient "
                    f"scaler. The current scale is {scaler.get_scale()}. This warning "
                    "can be ignored if no longer occurs after autoscaling of the scaler."
                )
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_mse += loss_mse.item()
        if batch % args.log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / args.log_interval
            cur_loss = total_loss / args.log_interval
            cur_mse = total_mse / args.log_interval
            logger.info(
                f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                f"lr {lr:05.4f} | ms/batch {ms_per_batch:5.2f} | "
                f"loss {cur_loss:5.2f} | mse {cur_mse:5.2f} |"
            )
            total_loss = 0
            total_mse = 0
            start_time = time.time()


def eval_perturb(
    loader: DataLoader, model: nn.Module, device: torch.device, args, gene_ids
) -> Dict:
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

    n_genes = len(gene_ids)

    for itr, batch in enumerate(loader):
        batch.to(device)
        pert_cat.extend(batch.pert)

        batch_size = len(batch.y)
        x: torch.Tensor = batch.x  # (batch_size * n_genes, 2)
        ori_gene_values = x[:, 0].view(batch_size, n_genes)
        pert_flags = x[:, 1].long().view(batch_size, n_genes)

        if args.include_zero_gene in ["all", "batch-wise"]:
            if args.include_zero_gene == "all":
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
            # CancerFoundation's forward method returns a dict
            output_dict = model(tens)
            p = output_dict["pred"]
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


def predict(
    model: nn.Module,
    pert_list: List[List[str]],
    pert_data,
    args,
    gene_ids,
    pool_size: Optional[int] = None,
) -> Dict:
    """
    Predict the gene expression values for the given perturbations.

    Args:
        model (:class:`torch.nn.Module`): The model to use for prediction.
        pert_list (:obj:`List[List[str]]`): The list of perturbations to predict.
        pert_data: The perturbation data object.
        args: Arguments object.
        gene_ids: Gene ID mappings.
        pool_size (:obj:`int`, optional): For each perturbation, use this number
            of cells in the control and predict their perturbation results. Report
            the stats of these predictions. If `None`, use all control cells.
    """
    adata = pert_data.adata
    ctrl_adata = adata[adata.obs["condition"] == "ctrl"]
    if pool_size is None:
        pool_size = len(ctrl_adata.obs)
    gene_list = pert_data.gene_names.values.tolist()
    n_genes = len(gene_ids)

    for pert in pert_list:
        for i in pert:
            if i not in gene_list:
                raise ValueError(
                    "The gene is not in the perturbation graph. Please select from GEARS.gene_list!"
                )

    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        results_pred = {}
        for pert in pert_list:
            cell_graphs = create_cell_graph_dataset_for_prediction(
                pert, ctrl_adata, gene_list, device, num_samples=pool_size
            )
            loader = DataLoader(
                cell_graphs, batch_size=args.eval_batch_size, shuffle=False
            )
            preds = []
            for batch_data in loader:
                batch_data.to(device)
                x: torch.Tensor = batch_data.x
                # Infer batch_size from the shape of x (batch_size * n_genes, 2)
                batch_size = x.shape[0] // n_genes
                ori_gene_values = x[:, 0].view(batch_size, n_genes)
                pert_flags = x[:, 1].long().view(batch_size, n_genes)

                if args.include_zero_gene == "all":
                    input_gene_ids = torch.arange(
                        n_genes, device=device, dtype=torch.long
                    )
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

                pred_gene_values = model(tens)["pred"]
                preds.append(pred_gene_values)

            preds = torch.cat(preds, dim=0)
            results_pred["_".join(pert)] = np.mean(preds.detach().cpu().numpy(), axis=0)

    return results_pred


def plot_perturbation(
    model: nn.Module,
    query: str,
    pert_data,
    args,
    gene_ids,
    save_file: Optional[str] = None,
    pool_size: Optional[int] = None,
):
    """Plot perturbation results."""
    sns.set_theme(style="ticks", rc={"axes.facecolor": (0, 0, 0, 0)}, font_scale=1.5)

    adata = pert_data.adata
    gene2idx = pert_data.node_map
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
    if query.split("+")[1] == "ctrl":
        pred = predict(
            model,
            [[query.split("+")[0]]],
            pert_data,
            args,
            gene_ids,
            pool_size=pool_size,
        )
        pred = pred[query.split("+")[0]][de_idx]
    else:
        pred = predict(
            model, [query.split("+")], pert_data, args, gene_ids, pool_size=pool_size
        )
        pred = pred["_".join(query.split("+"))][de_idx]
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


def run_training_loop(
    model,
    pert_data,
    optimizer,
    scheduler,
    scaler,
    criterion,
    args,
    gene_ids,
    n_genes,
    vocab,
    device,
    logger,
    save_dir,
):
    """Run the main training loop."""
    best_val_corr = 0
    best_model = None
    patience = 0

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train_loader = pert_data.dataloader["train_loader"]
        valid_loader = pert_data.dataloader["val_loader"]

        train(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            criterion,
            args,
            gene_ids,
            n_genes,
            vocab,
            device,
            logger,
            epoch,
        )

        val_res = eval_perturb(valid_loader, model, device, args, gene_ids)
        val_metrics = compute_perturbation_metrics(
            val_res, pert_data.adata[pert_data.adata.obs["condition"] == "ctrl"]
        )
        logger.info(f"val_metrics at epoch {epoch}: ")
        logger.info(val_metrics)

        elapsed = time.time() - epoch_start_time
        logger.info(f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | ")

        val_score = val_metrics["pearson"]
        if val_score > best_val_corr:
            best_val_corr = val_score
            best_model = copy.deepcopy(model)
            logger.info(f"Best model with score {val_score:5.4f}")
            patience = 0
        else:
            patience += 1
            if patience >= args.early_stop:
                logger.info(f"Early stop at epoch {epoch}")
                break

        scheduler.step()

    # If no improvement was found, use the last model
    if best_model is None:
        best_model = model
        logger.info("No improvement found, using final model")

    torch.save(best_model.state_dict(), save_dir / "best_model.pt")
    return best_model


def run_evaluation(best_model, pert_data, args, gene_ids, device, logger, save_dir):
    """Run evaluation on test set."""
    # Get perturbations to plot
    if args.data_name == "norman":
        perts_to_plot = ["SAMD1+ZBTB1"]
    elif args.data_name == "adamson":
        perts_to_plot = ["KCTD16+ctrl"]
    else:
        perts_to_plot = []

    # Plot perturbations
    for p in perts_to_plot:
        plot_perturbation(
            best_model,
            p,
            pert_data,
            args,
            gene_ids,
            pool_size=args.plot_pool_size,
            save_file=f"{save_dir}/{p}.png",
        )

    # Evaluate on test set
    test_loader = pert_data.dataloader["test_loader"]
    test_res = eval_perturb(test_loader, best_model, device, args, gene_ids)
    test_metrics = compute_perturbation_metrics(
        test_res, pert_data.adata[pert_data.adata.obs["condition"] == "ctrl"]
    )

    # save the dicts in json
    with open(f"{save_dir}/test_metrics.json", "w") as f:
        json.dump(test_metrics, f)

    # Perform deeper analysis
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

    for name, result in subgroup_analysis.items():
        for m in result.keys():
            mean_value = np.mean(subgroup_analysis[name][m])
            logger.info("test_" + name + "_" + m + ": " + str(mean_value))


def main():
    """Main function to orchestrate the entire pipeline."""
    # Parse arguments
    args = parse_args()

    # Handle checkpoint argument - convert to load_model format for compatibility
    if args.checkpoint is not None:
        args.load_model = str(Path(args.checkpoint).parent)

    # Set random seed
    set_seed(args.seed)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup save directory
    if args.save_dir is None:
        save_dir = Path(
            f"./save/dev_perturb_{args.data_name}-{time.strftime('%b%d-%H-%M')}/"
        )
    else:
        save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"saving to {save_dir}")

    # Setup logging
    logger = setup_logging(save_dir)

    # Load data
    pert_data = load_data(args, logger)

    # Setup vocabulary
    vocab, genes, checkpoint_file = setup_vocabulary(args, pert_data, logger)

    # Create model
    model, gene_ids, n_genes = create_model(args, vocab, genes, checkpoint_file)
    model.to(device)

    # Setup training components
    criterion = masked_mse_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, args.schedule_interval, gamma=0.9
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # Run training
    best_model = run_training_loop(
        model,
        pert_data,
        optimizer,
        scheduler,
        scaler,
        criterion,
        args,
        gene_ids,
        n_genes,
        vocab,
        device,
        logger,
        save_dir,
    )

    # Run evaluation
    run_evaluation(best_model, pert_data, args, gene_ids, device, logger, save_dir)


if __name__ == "__main__":
    main()
