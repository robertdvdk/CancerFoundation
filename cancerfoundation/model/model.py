from pathlib import Path
import pytorch_lightning as pl
import os
from typing import Any, List, Optional, Union
import transformers
import torch
from cancerfoundation.utils import load_pretrained
from cancerfoundation.model.module import TransformerModule
from cancerfoundation.loss import get_loss
from safetensors import safe_open
from cancerfoundation.loss import LossType
from pytorch_lightning.utilities.types import OptimizerLRSchedulerConfig


class CancerFoundation(pl.LightningModule):
    """The main PyTorch Lightning module for the Cancer Foundation model.

    This class encapsulates the entire model training, validation, and optimization pipeline.
    It wraps the `TransformerModule` and handles hyperparameter configuration, loss calculation,
    optimizer and scheduler setup, and the training/validation loops required by PyTorch Lightning.
    """

    def __init__(
        self,
        n_bins: int,
        input_emb_style: str,
        max_seq_len: int,
        input_style: str,
        mask_ratio: float,
        TRUNC_BY_SAMPLE: bool,
        training_tasks: str,
        embsize: int,
        nheads: int,
        d_hid: int,
        nlayers: int,
        dropout: float,
        lr: float,
        epochs: int,
        vocab,
        warmup_ratio_or_step: float,
        scheduler_interval: int,
        scheduler_factor: float,
        compile_model: bool,
        data_path: Union[str, os.PathLike],
        loss_type: LossType = LossType.MSE,
        conditions: Optional[List[str]] = None,
        conditions_nums: Optional[Any] = None,
        mvc_decoder_style: str = "inner product",
        scale_zero_expression: Optional[float] = None,
        do_dat: bool = False,
        explicit_zero_prob: Optional[bool] = False,
        balance_primary: Optional[str] = None,
        balance_secondary: Optional[str] = None,
        zero_percentages: Optional[List[float]] = None,
    ):
        """Initializes the CancerFoundation LightningModule.

        Args:
            n_bins (int): The number of bins for discretizing expression values.
            input_emb_style (str): The style of input embedding ('category' or 'continuous').
            max_seq_len (int): The maximum sequence length.
            input_style (str): Style of input data processing.
            mask_ratio (float): The ratio of tokens to mask for the MLM task.
            TRUNC_BY_SAMPLE (bool): Whether to truncate sequences by sample.
            training_tasks (str): The training tasks to perform ('mlm', 'gen', 'both').
            embsize (int): The embedding size (d_model).
            nheads (int): The number of attention heads.
            d_hid (int): The dimension of the feed-forward hidden layer.
            nlayers (int): The number of transformer layers.
            dropout (float): The dropout rate.
            lr (float): The learning rate.
            epochs (int): The total number of training epochs.
            vocab: The vocabulary mapping gene names to token IDs.
            warmup_ratio_or_step (float): The ratio or number of steps for learning rate warmup.
            scheduler_interval (int): The interval for the StepLR scheduler.
            scheduler_factor (float): The factor for the StepLR scheduler.
            compile_model (bool): If True, compile the model using `torch.compile`.
            data_path (Union[str, os.PathLike]): The path to the data.
            loss_type (LossType, optional): The type of loss function to use. Defaults to LossType.MSE.
            conditions (Optional[List[str]], optional): A list of conditional variables. Defaults to None.
            conditions_nums (Optional[Any], optional): A dictionary mapping condition names to their number of categories. Defaults to None.
            mvc_decoder_style (str, optional): The architecture style for the MVC decoder. Defaults to "inner product".
            scale_zero_expression (Optional[float], optional): A factor to scale the loss for zero-expression values. Defaults to None.
            do_dat (bool, optional): If True, enable Domain Adversarial Training. Defaults to False.
            explicit_zero_prob (Optional[bool], optional): If True, explicitly model zero probability. Defaults to False.
            balance_primary (Optional[str], optional): The primary variable for balanced sampling. Defaults to None.
            balance_secondary (Optional[str], optional): The secondary variable for balanced sampling. Defaults to None.
            zero_percentages (Optional[List[float]], optional): Percentages for balancing zero expression. Defaults to None.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["vocab"])
        self.vocab = vocab
        # Store all parameters
        self.n_bins = n_bins
        self.input_emb_style = input_emb_style
        self.max_seq_len = max_seq_len
        self.input_style = input_style
        self.mask_ratio = (
            [0.25, 0.50, 0.75] if training_tasks in ["gen", "both"] else mask_ratio
        )
        self.TRUNC_BY_SAMPLE = TRUNC_BY_SAMPLE
        self.training_tasks = training_tasks
        self.embsize = embsize
        self.nheads = nheads
        self.d_hid = d_hid
        self.nlayers = nlayers
        self.dropout = dropout
        self.lr = lr
        self.warmup_ratio_or_step = warmup_ratio_or_step
        self.scheduler_interval = scheduler_interval
        self.scheduler_factor = scheduler_factor
        self.loss_type = loss_type
        self.data_path = data_path
        self.epochs = epochs
        self.compile_model = compile_model

        # Training configuration
        self.pad_token = "<pad>"
        self.cls_token = "<cls>"
        self.MVC = True
        self.USE_GENERATIVE_TRAINING = (
            True if self.training_tasks in ["gen", "both"] else False
        )
        self.use_cell_embedding = False
        self.domain_nums = None
        self.explicit_zero_prob = explicit_zero_prob
        self.do_dat = do_dat
        self.conditions = conditions
        self.conditions_nums = conditions_nums

        # Balance sampling parameters
        if balance_primary is None and balance_secondary is not None:
            raise ValueError(
                "balance_secondary is not allowed to be set (not None) if balance_primary is None."
            )
        self.balance_primary = balance_primary
        self.balance_secondary = balance_secondary
        self.zero_percentages = zero_percentages
        self.scale_zero_expression = scale_zero_expression

        # Setup token values based on embedding style
        if self.input_emb_style == "category":
            self.mask_value = self.n_bins + 1
            self.pad_value = self.n_bins  # for padding gene expr values
            self.n_input_bins = self.n_bins + 2
        else:
            self.mask_value = -1
            self.pad_value = -2
            self.n_input_bins = self.n_bins

        self.pad_token_id = self.vocab["<pad>"]
        self.cls_token_id = self.vocab["<cls>"]

        # Initialize dataset and model
        self._setup_model(mvc_decoder_style)

    def _setup_model(self, mvc_decoder_style: str):
        """Initializes the model and its loss function."""
        self.criterion = get_loss(
            loss_type=self.loss_type,
            num_classes=self.n_input_bins if self.n_input_bins else None,
            scale_zero_expression=self.scale_zero_expression,
        )

        self.model = TransformerModule(
            ntoken=len(self.vocab.keys()),
            d_model=self.embsize,
            out_dim=self.criterion.get_in_dim(),
            mvc_decoder_style=mvc_decoder_style,
            nhead=self.nheads,
            d_hid=self.d_hid,
            nlayers=self.nlayers,
            dropout=self.dropout,
            pad_token_id=self.pad_token_id,
            criterion=self.criterion,
            pad_value=self.pad_value,
            do_mvc=self.MVC,
            conditions=self.conditions_nums,
            input_emb_style=self.input_emb_style,
            n_input_bins=self.n_input_bins,
            use_generative_training=self.USE_GENERATIVE_TRAINING,
            do_dat=self.do_dat,
            explicit_zero_prob=self.explicit_zero_prob,
        )
        if self.compile_model:
            self.model = torch.compile(self.model)

    def forward(self, data_dict, use_cell_embedding=None):
        """Performs a forward pass through the underlying `TransformerModule`.

        Args:
            data_dict (dict): A dictionary of input tensors.
            use_cell_embedding (Optional[bool], optional): A flag to control a specific training behavior.
                If None, uses the module's default. Defaults to None.

        Returns:
            dict: The output dictionary from the model, typically containing losses.
        """
        if use_cell_embedding is None:
            use_cell_embedding = self.use_cell_embedding
        return self.model(data_dict, use_cell_embedding=use_cell_embedding)

    def training_step(self, batch, batch_idx):
        """Performs a single training step.

        Args:
            batch (dict): The batch of data from the DataLoader.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The total loss for the batch.
        """
        # Update use_cell_embedding based on global step
        self.use_cell_embedding = (
            self.USE_GENERATIVE_TRAINING and self.global_step > 1000
        )

        loss_dict = self.forward(batch, use_cell_embedding=self.use_cell_embedding)

        # Log training metrics
        for key, value in loss_dict.items():
            self.log(f"train/{key}", value, on_step=True, on_epoch=False, prog_bar=True)

        return loss_dict["total_loss"]

    def validation_step(self, batch, batch_idx):
        """Performs a single validation step.

        Args:
            batch (dict): The batch of data from the DataLoader.
            batch_idx (int): The index of the batch.

        Returns:
            dict: The dictionary of losses for the validation batch.
        """

        if batch_idx == 0:
            print(
                f"Rank {self.trainer.global_rank}: Starting validation with {len(self.trainer.val_dataloaders)} batches"
            )

        loss_dict = self.forward(batch, use_cell_embedding=True)

        # Log validation metrics
        for key, value in loss_dict.items():
            self.log(
                f"val/{key}",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

        return loss_dict

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        """Configures the optimizer and learning rate scheduler.

        Uses an Adam optimizer. If warmup is specified, it uses a cosine learning rate schedule with warmup.
            Otherwise, it uses a step-based decay scheduler.

        Returns:
            dict: The optimizer and scheduler configuration for PyTorch Lightning.
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if self.warmup_ratio_or_step > 0:
            # Calculate total training steps
            total_num_batches = self.trainer.estimated_stepping_batches
            warmup_steps = (
                int(total_num_batches * self.warmup_ratio_or_step)
                if self.warmup_ratio_or_step < 1
                else int(self.warmup_ratio_or_step)
            )

            scheduler = transformers.get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_num_batches,
                last_epoch=-1,
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, self.scheduler_interval, gamma=self.scheduler_factor
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                },
            }

    def load_pretrained_weights(
        self,
        pretrained_model_path: Path,
        gene_mapping: Optional[dict],
        verbose: bool = True,
    ):
        """Loads pretrained weights from a checkpoint file into the current model.

        This method supports both `.safetensors` and PyTorch `.pth`/`.pt` formats.
        It can handle vocab mismatches by re-mapping gene embeddings if a
        `gene_mapping` dictionary is provided.

        Args:
            pretrained_model_path (Path): Path to the pretrained model file.
            gene_mapping (Optional[dict]): A dictionary to map gene names from the
                pretrained vocab to the current vocab.
            verbose (bool, optional): If True, prints information about matched
                and unmatched weights. Defaults to True.
        """
        if pretrained_model_path.name.endswith(".safetensors"):
            tensors = {}
            with safe_open(pretrained_model_path, framework="pt", device="cpu") as f:
                for k in f.keys():
                    tensors[k] = f.get_tensor(k)
        elif pretrained_model_path.name.endswith(
            ".pth"
        ) or pretrained_model_path.name.endswith(".pt"):
            tensors = torch.load(pretrained_model_path, map_location="cpu")
        else:
            raise ValueError("Unsupported file format. Use .safetensors, .pth, or .pt")

        return load_pretrained(self.model, tensors, gene_mapping, verbose=verbose)
