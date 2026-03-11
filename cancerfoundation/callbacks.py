from typing import Optional

import numpy as np
import torch
from pytorch_lightning import Callback


class MeanBaselineCallback(Callback):
    """Tracks a running mean of gene expression during training and computes
    a mean-baseline MSE on the validation set for comparison with the model."""

    def __init__(self, n_genes: int, pad_value: float = -2, mask_value: float = -1):
        super().__init__()
        self.n_genes = n_genes
        self.pad_value = pad_value
        self.mask_value = mask_value
        self.gene_sum = torch.zeros(n_genes, dtype=torch.float64)
        self.gene_count = torch.zeros(n_genes, dtype=torch.float64)
        self.gene_mean = torch.zeros(n_genes, dtype=torch.float32)
        self._val_mse_sum = 0.0
        self._val_mse_count = 0

    def _is_valid_expr(self, expr_vals):
        """Identify positions with real expression values (not pad/mask/CLS)."""
        # pad_value=-2, mask_value=-1; real expression values are always >= 0
        return expr_vals >= 0

    def _is_masked(self, masked_expr):
        """Identify positions that were masked by the collator."""
        return masked_expr == self.mask_value

    def _accumulate_from_batch(self, batch):
        """Accumulate gene expression sums/counts from a training batch."""
        if "gene" in batch and "expr" in batch:
            gene_ids = batch["gene"]
            expr_vals = batch["expr"]
            mask = batch.get("gene_key_padding_mask", None)
        elif "pcpt_gene" in batch:
            gene_ids = batch["pcpt_gene"]
            expr_vals = batch["pcpt_expr"]
            mask = batch.get("pcpt_key_padding_mask", None)
        else:
            return

        gene_ids = gene_ids.cpu().long()
        expr_vals = expr_vals.cpu().float()

        valid = self._is_valid_expr(expr_vals)
        if mask is not None:
            valid = valid & ~mask.cpu()

        for b in range(gene_ids.shape[0]):
            v = valid[b]
            gids = gene_ids[b][v]
            vals = expr_vals[b][v]
            in_range = (gids >= 0) & (gids < self.n_genes)
            gids = gids[in_range]
            vals = vals[in_range]
            self.gene_sum.scatter_add_(0, gids, vals.double())
            self.gene_count.scatter_add_(0, gids, torch.ones_like(vals, dtype=torch.float64))

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._accumulate_from_batch(batch)

    def on_validation_epoch_start(self, trainer, pl_module):
        nonzero = self.gene_count > 0
        self.gene_mean = torch.zeros(self.n_genes, dtype=torch.float32)
        self.gene_mean[nonzero] = (self.gene_sum[nonzero] / self.gene_count[nonzero]).float()
        self._val_mse_sum = 0.0
        self._val_mse_count = 0

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if "gene" in batch and "expr" in batch:
            gene_ids = batch["gene"]
            target = batch["expr"]
            mask = batch.get("gene_key_padding_mask", None)
            masked_expr = batch.get("masked_expr", None)
        elif "pcpt_gene" in batch:
            gene_ids = batch["pcpt_gene"]
            target = batch["pcpt_expr"]
            mask = batch.get("pcpt_key_padding_mask", None)
            masked_expr = batch.get("masked_expr", None)
        else:
            return

        gene_ids = gene_ids.cpu().long()
        target = target.cpu().float()

        valid = self._is_valid_expr(target)
        if mask is not None:
            valid = valid & ~mask.cpu()
        if masked_expr is not None:
            valid = valid & self._is_masked(masked_expr.cpu())

        gene_ids_clamped = gene_ids.clamp(0, self.n_genes - 1)
        mean_preds = self.gene_mean[gene_ids_clamped]

        mse = ((mean_preds - target) ** 2) * valid.float()
        self._val_mse_sum += mse.sum().item()
        self._val_mse_count += valid.sum().item()

    def on_validation_epoch_end(self, trainer, pl_module):
        if self._val_mse_count > 0:
            mean_baseline_mse = self._val_mse_sum / self._val_mse_count
        else:
            mean_baseline_mse = 0.0
        pl_module.log(
            "val/mean_baseline_mse",
            mean_baseline_mse,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )


class CellTypeProbeCallback(Callback):
    """Periodically evaluates cell-type classification accuracy using a linear
    probe (logistic regression) on frozen CLS embeddings.

    Uses scanpy's pbmc3k_processed dataset (2,638 cells, 8 cell types) as an
    external benchmark.  Embeddings are produced via the model's `embed()`
    method; a stratified 80/20 train/test split is held fixed across epochs.

    Logged to wandb: accuracy, macro-F1, and per-class F1.
    """

    def __init__(
        self,
        eval_every_n_epochs: int,
        test_size: float = 0.2,
        seed: Optional[int] = 42,
    ):
        super().__init__()
        self.eval_every_n_epochs = eval_every_n_epochs
        self.test_size = test_size
        self.seed = seed
        self._adata = None
        self._train_idx = None
        self._test_idx = None

    def _load_data(self):
        """Load pbmc3k and prepare a fixed train/test split (once)."""
        if self._adata is not None:
            return
        import scanpy as sc
        from sklearn.model_selection import train_test_split

        processed = sc.datasets.pbmc3k_processed()
        # Use the raw layer (all genes, log-normalized) so embed() can do
        # its own HVG selection and binning.
        self._adata = processed.raw.to_adata()
        self._labels = processed.obs["louvain"].values

        idx = np.arange(len(self._adata))
        self._train_idx, self._test_idx = train_test_split(
            idx,
            test_size=self.test_size,
            stratify=self._labels,
            random_state=self.seed,
        )

    def _evaluate(self, trainer, pl_module, epoch):
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, f1_score

        print(f"[CellTypeProbe] Running evaluation at epoch {epoch}...")

        try:
            self._load_data()

            # Embed all cells (2,638 cells — takes ~1s)
            emb_df = pl_module.embed(self._adata)
            X = emb_df.values

            X_train, X_test = X[self._train_idx], X[self._test_idx]
            y_train, y_test = self._labels[self._train_idx], self._labels[self._test_idx]

            clf = LogisticRegression(max_iter=2000, random_state=self.seed)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            f1_macro = f1_score(y_test, y_pred, average="macro")

            if trainer.logger is not None:
                trainer.logger.experiment.log(
                    {
                        "celltype_probe/accuracy": acc,
                        "celltype_probe/f1_macro": f1_macro,
                        "epoch": epoch,
                    }
                )

                # Log per-class F1 to wandb
                try:
                    classes = sorted(set(y_test))
                    per_class_f1 = f1_score(y_test, y_pred, labels=classes, average=None)
                    for cls, f1 in zip(classes, per_class_f1):
                        trainer.logger.experiment.log(
                            {
                                f"celltype_probe/f1_{cls}": f1,
                                "epoch": epoch,
                            }
                        )
                except Exception:
                    pass

            print(f"[CellTypeProbe] epoch {epoch}: acc={acc:.3f}, macro-F1={f1_macro:.3f}")

        except Exception as e:
            print(f"[CellTypeProbe] Evaluation failed: {e}")
            import traceback

            traceback.print_exc()

    def on_train_start(self, trainer, pl_module):
        """Pre-training baseline evaluation (runs inside fit, DDP-safe)."""
        if trainer.global_rank != 0:
            return
        self._evaluate(trainer, pl_module, epoch=-1)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        if trainer.global_rank != 0:
            return
        epoch = trainer.current_epoch
        if epoch % self.eval_every_n_epochs != 0:
            return
        self._evaluate(trainer, pl_module, epoch=epoch)


EVAL_DATASET_KEYS = {
    "neftel_ss2": {"batch_key": "sample", "label_key": "subtype"},
    "ji_skin": {"batch_key": "sample", "label_key": "celltype"},
    "kim_lung": {"batch_key": "sample", "label_key": "celltype"},
}


class ScibMetricsCallback(Callback):
    """Periodically evaluates batch correction and bio conservation metrics
    using scib-metrics on:
      1. The validation set embeddings
      2. External datasets with HVG-based gene selection
      3. The same external datasets with random gene selection

    UMAPs and metric tables are logged to wandb.
    """

    def __init__(
        self,
        eval_every_n_epochs: int,
        dataset_paths: list[str],
        max_seq_len: int,
        val_batch_key: str = "technology",
        val_label_key: str = "cancer_type",
        seed: Optional[int] = 42,
    ):
        super().__init__()
        self.eval_every_n_epochs = eval_every_n_epochs
        self.max_seq_len = max_seq_len
        self.val_batch_key = val_batch_key
        self.val_label_key = val_label_key
        self.seed = seed

        from pathlib import Path

        self.datasets = []
        for p in dataset_paths:
            stem = Path(p).stem
            keys = EVAL_DATASET_KEYS.get(stem, {"batch_key": "sample", "label_key": "celltype"})
            self.datasets.append(
                {
                    "name": stem,
                    "path": p,
                    "batch_key": keys["batch_key"],
                    "label_key": keys["label_key"],
                    "_adata": None,
                    "_random_gene_indices": None,
                    "_random_common_genes": None,
                }
            )

    def _load_dataset(self, ds):
        """Load a dataset once and cache it."""
        if ds["_adata"] is not None:
            return
        import anndata as ad

        ds["_adata"] = ad.read_h5ad(ds["path"])
        if hasattr(ds["_adata"].X, "toarray"):
            ds["_adata"].X = ds["_adata"].X.toarray()

    def _get_random_gene_indices(self, model, ds):
        """Get a fixed random subset of gene indices (intersection with vocab)."""
        if ds["_random_gene_indices"] is not None:
            return ds["_random_gene_indices"]
        self._load_dataset(ds)
        vocab = model.vocab
        common_genes = [g for g in ds["_adata"].var_names if g in vocab]
        rng = np.random.RandomState(self.seed)
        n_select = min(self.max_seq_len, len(common_genes))
        ds["_random_gene_indices"] = sorted(
            rng.choice(len(common_genes), size=n_select, replace=False).tolist()
        )
        ds["_random_common_genes"] = [common_genes[i] for i in ds["_random_gene_indices"]]
        return ds["_random_gene_indices"]

    def _embed_with_genes(self, model, adata, gene_list):
        """Embed an AnnData using a specific list of genes (subset of vocab)."""
        from cancerfoundation.data.preprocess import binning

        device = next(model.model.parameters()).device
        model.model.eval()

        data = adata[:, gene_list].copy()
        if hasattr(data.X, "toarray"):
            data.X = data.X.toarray()

        normalise = model.model.decoder.normalise_bins
        for idx in range(data.n_obs):
            data.X[idx] = binning(data.X[idx], model.n_bins)
            if normalise:
                data.X[idx] = data.X[idx] / model.n_bins

        gene_ids = torch.LongTensor([model.vocab[g] for g in gene_list])
        count_matrix = data.X if isinstance(data.X, np.ndarray) else data.X.toarray()

        embeddings = []
        batch_size = 64
        for i in range(0, len(data), batch_size):
            batch_expr = torch.FloatTensor(count_matrix[i : i + batch_size]).to(device)
            batch_genes = gene_ids.unsqueeze(0).expand(batch_expr.shape[0], -1).to(device)

            # Prepend CLS token
            batch_genes = torch.cat(
                [
                    torch.full(
                        (batch_expr.shape[0], 1),
                        model.cls_token_id,
                        dtype=torch.long,
                        device=device,
                    ),
                    batch_genes,
                ],
                dim=1,
            )
            batch_expr = torch.cat(
                [
                    torch.full(
                        (batch_expr.shape[0], 1),
                        model.pad_value,
                        dtype=batch_expr.dtype,
                        device=device,
                    ),
                    batch_expr,
                ],
                dim=1,
            )
            padding_mask = torch.zeros(batch_genes.shape, dtype=torch.bool, device=device)

            with torch.no_grad():
                if model.model.use_generative_training:
                    output = model.model.embed(
                        src=batch_genes,
                        values=batch_expr,
                        src_key_padding_mask=padding_mask,
                    )
                    transformer_output = output[0]
                else:
                    transformer_output = model.model.encode(
                        src=batch_genes,
                        values=batch_expr,
                        src_key_padding_mask=padding_mask,
                    )

            cell_emb = transformer_output[:, 0, :]
            embeddings.append(cell_emb.cpu().numpy())

        return np.concatenate(embeddings, axis=0)

    def _get_val_embeddings_and_metadata(self, trainer, pl_module):
        """Collect cell embeddings and metadata from the validation set."""
        from cancerfoundation.data.preprocess import binning as bin_fn

        pl_module.model.eval()
        device = next(pl_module.model.parameters()).device

        datamodule = trainer.datamodule
        val_subset = datamodule.val_dataset
        full_dataset = val_subset.dataset
        val_indices = [int(i) for i in val_subset.indices]

        # Get metadata for val cells using the reverse mapping
        obs_df = full_dataset.obs.iloc[val_indices].copy()
        mapping = full_dataset.mapping

        # Reverse-map encoded integers back to string labels
        for col in [self.val_batch_key, self.val_label_key]:
            if col in mapping:
                reverse_map = {int(v): k for k, v in mapping[col].items()}
                obs_df[col] = obs_df[col].map(reverse_map)

        embeddings = []
        batch_size = 64
        normalise = pl_module.model.decoder.normalise_bins

        for start in range(0, len(val_indices), batch_size):
            end = min(start + batch_size, len(val_indices))
            batch_items = [full_dataset[val_indices[j]] for j in range(start, end)]

            genes_list = [item["genes"] for item in batch_items]
            expr_list = [item["expressions"] for item in batch_items]

            # Pad to same length
            max_len = max(g.shape[0] for g in genes_list)
            padded_genes = torch.full(
                (len(genes_list), max_len), pl_module.pad_token_id, dtype=torch.long
            )
            padded_expr = torch.full(
                (len(expr_list), max_len), pl_module.pad_value, dtype=torch.float32
            )
            padding_mask = torch.ones((len(genes_list), max_len), dtype=torch.bool)

            for j, (g, e) in enumerate(zip(genes_list, expr_list)):
                seq_len = min(g.shape[0], max_len)
                padded_genes[j, :seq_len] = g[:seq_len]
                padded_expr[j, :seq_len] = e[:seq_len]
                padding_mask[j, :seq_len] = False

            # Bin expression values (skip CLS and padding positions)
            for j in range(padded_expr.shape[0]):
                valid = ~padding_mask[j] & (padded_expr[j] >= 0)
                if valid.any():
                    vals = padded_expr[j][valid].numpy()
                    binned = bin_fn(vals, pl_module.n_bins)
                    if normalise:
                        binned = binned / pl_module.n_bins
                    padded_expr[j][valid] = torch.from_numpy(np.array(binned, dtype=np.float32))

            # Truncate to max_seq_len (keep CLS at position 0)
            if max_len > self.max_seq_len:
                padded_genes = padded_genes[:, : self.max_seq_len]
                padded_expr = padded_expr[:, : self.max_seq_len]
                padding_mask = padding_mask[:, : self.max_seq_len]

            padded_genes = padded_genes.to(device)
            padded_expr = padded_expr.to(device)
            padding_mask = padding_mask.to(device)

            with torch.no_grad():
                if pl_module.model.use_generative_training:
                    output = pl_module.model.embed(
                        src=padded_genes,
                        values=padded_expr,
                        src_key_padding_mask=padding_mask,
                    )
                    transformer_output = output[0]
                else:
                    transformer_output = pl_module.model.encode(
                        src=padded_genes,
                        values=padded_expr,
                        src_key_padding_mask=padding_mask,
                    )

            cell_emb = transformer_output[:, 0, :]
            embeddings.append(cell_emb.cpu().numpy())

        emb_matrix = np.concatenate(embeddings, axis=0)
        return emb_matrix, obs_df

    def _run_scib_and_log(
        self, adata, batch_key, label_key, embedding_key, prefix, trainer, epoch=None
    ):
        """Run scib-metrics Benchmarker and log results + UMAP to wandb."""
        import matplotlib
        import scanpy as sc

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        try:
            from scib_metrics.benchmark import Benchmarker
        except ImportError:
            print("scib-metrics not installed, skipping ScibMetrics evaluation.")
            return

        if epoch is None:
            epoch = trainer.current_epoch
        logger = trainer.logger
        if logger is None:
            return

        # Compute neighbors and UMAP from the embedding
        sc.pp.neighbors(adata, use_rep=embedding_key)
        sc.tl.umap(adata)

        # Log combined 1x2 UMAP (batch | label)
        has_batch = batch_key in adata.obs.columns
        has_label = label_key in adata.obs.columns
        if has_batch or has_label:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            if has_batch:
                sc.pl.umap(
                    adata,
                    color=batch_key,
                    ax=axes[0],
                    show=False,
                    title=f"{prefix} - batch (epoch {epoch})",
                )
            else:
                axes[0].set_visible(False)
            if has_label:
                sc.pl.umap(
                    adata,
                    color=label_key,
                    ax=axes[1],
                    show=False,
                    title=f"{prefix} - label (epoch {epoch})",
                )
            else:
                axes[1].set_visible(False)
            plt.tight_layout()
            try:
                import wandb

                logger.experiment.log(
                    {
                        f"{prefix}/umap": wandb.Image(fig),
                        "epoch": epoch,
                    }
                )
            except Exception as e:
                print(f"Failed to log UMAP for {prefix}: {e}")
            plt.close(fig)

        # Run scib-metrics benchmark
        try:
            bm = Benchmarker(
                adata,
                batch_key=batch_key,
                label_key=label_key,
                embedding_obsm_keys=[embedding_key],
                n_jobs=1,
            )
            bm.prepare()
            bm.benchmark()
            results = bm.get_results(min_max_scale=False)

            # Print all metrics to stdout
            import wandb

            summary_keys = {"Total", "Bio conservation", "Batch correction"}
            print(f"[ScibMetrics] {prefix} results (epoch {epoch}):")
            for col in results.columns:
                if col == "Embedding":
                    continue
                val = results[col].iloc[0]
                print(f"  {col}: {val:.4f}" if np.isfinite(val) else f"  {col}: {val}")

            # Log only summary metrics to wandb
            for col in summary_keys:
                if col in results.columns:
                    val = results[col].iloc[0]
                    if np.isfinite(val):
                        logger.experiment.log(
                            {
                                f"{prefix}/{col}": val,
                                "epoch": epoch,
                            }
                        )

        except Exception as e:
            print(f"scib-metrics benchmark failed for {prefix}: {e}")
            import traceback

            traceback.print_exc()

    def _evaluate(self, trainer, pl_module, epoch):
        import anndata as ad
        import scanpy as sc

        print(f"[ScibMetrics] Running evaluation at epoch {epoch}...")

        # --- 1. Validation set embeddings ---
        try:
            emb_matrix, obs_df = self._get_val_embeddings_and_metadata(trainer, pl_module)
            val_adata = ad.AnnData(X=emb_matrix, obs=obs_df.reset_index(drop=True))
            val_adata.obsm["X_emb"] = emb_matrix

            self._run_scib_and_log(
                val_adata,
                batch_key=self.val_batch_key,
                label_key=self.val_label_key,
                embedding_key="X_emb",
                prefix="scib/val",
                trainer=trainer,
                epoch=epoch,
            )
        except Exception as e:
            print(f"[ScibMetrics] Validation set evaluation failed: {e}")
            import traceback

            traceback.print_exc()

        # --- 2 & 3. External datasets with HVG and random gene selection ---
        for ds in self.datasets:
            name = ds["name"]

            # HVG evaluation
            try:
                self._load_dataset(ds)
                adata = ds["_adata"].copy()
                vocab = pl_module.vocab
                common_genes = [g for g in adata.var_names if g in vocab]
                adata_common = adata[:, common_genes].copy()

                sc.pp.highly_variable_genes(
                    adata_common, n_top_genes=min(self.max_seq_len, len(common_genes))
                )
                hvg_genes = adata_common.var_names[adata_common.var["highly_variable"]].tolist()

                emb_hvg = self._embed_with_genes(pl_module, adata, hvg_genes)
                hvg_adata = ad.AnnData(X=emb_hvg, obs=adata.obs.copy().reset_index(drop=True))
                hvg_adata.obsm["X_emb"] = emb_hvg

                self._run_scib_and_log(
                    hvg_adata,
                    batch_key=ds["batch_key"],
                    label_key=ds["label_key"],
                    embedding_key="X_emb",
                    prefix=f"scib/{name}_hvg",
                    trainer=trainer,
                    epoch=epoch,
                )
            except Exception as e:
                print(f"[ScibMetrics] {name} HVG evaluation failed: {e}")
                import traceback

                traceback.print_exc()

            # Random gene evaluation
            try:
                self._load_dataset(ds)
                adata = ds["_adata"].copy()
                self._get_random_gene_indices(pl_module, ds)
                random_genes = ds["_random_common_genes"]

                emb_random = self._embed_with_genes(pl_module, adata, random_genes)
                rnd_adata = ad.AnnData(X=emb_random, obs=adata.obs.copy().reset_index(drop=True))
                rnd_adata.obsm["X_emb"] = emb_random

                self._run_scib_and_log(
                    rnd_adata,
                    batch_key=ds["batch_key"],
                    label_key=ds["label_key"],
                    embedding_key="X_emb",
                    prefix=f"scib/{name}_random",
                    trainer=trainer,
                    epoch=epoch,
                )
            except Exception as e:
                print(f"[ScibMetrics] {name} random evaluation failed: {e}")
                import traceback

                traceback.print_exc()

        print(f"[ScibMetrics] Evaluation at epoch {epoch} complete.")

    def on_train_start(self, trainer, pl_module):
        """Pre-training baseline evaluation (runs inside fit, DDP-safe)."""
        if trainer.global_rank != 0:
            return
        if trainer.logger is None:
            return
        self._evaluate(trainer, pl_module, epoch=-1)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        # Only run on rank 0
        if trainer.global_rank != 0:
            return
        # Only run every n epochs (and also at epoch 0 for pre-training baseline)
        epoch = trainer.current_epoch
        if epoch % self.eval_every_n_epochs != 0:
            return
        if trainer.logger is None:
            return
        self._evaluate(trainer, pl_module, epoch=epoch)
