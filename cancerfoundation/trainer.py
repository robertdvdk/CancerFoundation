import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from collections import defaultdict
from pathlib import Path
import time
import os
from typing import Dict, List, Optional, Union
from .data_sampler import get_balanced_sampler
import transformers
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, random_split
from torch import nn
import torch
import json
from .data_collator import AnnDataCollator
from .dataset import SingleCellDataset
from .utils import load_pretrained
from .model import TransformerModel
from cancerfoundation.loss import get_loss
import numpy as np
from safetensors import safe_open
from .loss import LossType
from torch.nn.attention import SDPBackend, sdpa_kernel



class LightningModule(pl.LightningModule):
    def __init__(
        self,
        args: Dict,
        n_bins: int,
        input_emb_style: str,
        max_seq_len: int,
        input_style: str,
        mask_ratio: float,
        TRUNC_BY_SAMPLE: bool,
        training_tasks: str,
        batch_size: int,
        eval_batch_size: int,
        embsize: int,
        nheads: int,
        d_hid: int,
        nlayers: int,
        dropout: float,
        lr: float,
        warmup_ratio_or_step: float,
        scheduler_interval: int,
        scheduler_factor: float,
        data_path: Union[str, os.PathLike],
        loss_type: LossType = LossType.MSE,
        conditions: List[str] = None,
        mvc_decoder_style: str = "inner product",
        scale_zero_expression: Optional[float] = None,
        do_dat: bool = False,
        explicit_zero_prob: Optional[bool] = False,
        balance_primary: Optional[str] = None,
        balance_secondary: Optional[str] = None,
        zero_percentages: Optional[List[float]] = None,
        epochs: int = 100,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Store all parameters
        self.args = args
        self.n_bins = n_bins
        self.input_emb_style = input_emb_style
        self.max_seq_len = max_seq_len
        self.input_style = input_style
        self.mask_ratio = (
            [0.25, 0.50, 0.75] if training_tasks in ["gen", "both"] else mask_ratio
        )
        self.TRUNC_BY_SAMPLE = TRUNC_BY_SAMPLE
        self.training_tasks = training_tasks
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
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
        self.conditions_nums = None
        
        # Balance sampling parameters
        if balance_primary is None and balance_secondary is not None:
            raise ValueError("balance_secondary is not allowed to be set (not None) if balance_primary is None.")
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

        # Initialize dataset and model
        self._setup_data()
        self._setup_model(mvc_decoder_style)

    def _setup_data(self):
        """Initialize dataset and create train/validation splits"""
        self.dataset = SingleCellDataset(
            data_dir=self.data_path,
            pad_value=self.pad_value,
            obs_columns=self.conditions
        )
        
        if self.conditions:
            self.conditions_nums = {}
            for cond in self.conditions:
                self.conditions_nums[cond] = len(self.dataset.mapping[cond].keys())
        
        # Create train/validation split
        self.train_dataset, self.val_dataset = random_split(self.dataset, [0.95, 0.05])
        self.vocab = self.dataset.vocab
        
        self.pad_token_id = self.vocab["<pad>"]
        self.cls_token_id = self.vocab["<cls>"]

    def _setup_model(self, mvc_decoder_style: str):
        """Initialize model and loss function"""
        self.criterion = get_loss(
            loss_type=self.loss_type, 
            num_classes=self.n_input_bins if self.n_input_bins else None, 
            scale_zero_expression=self.scale_zero_expression
        )

        self.model = TransformerModel(
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

    def forward(self, data_dict, use_cell_embedding=None):
        """Forward pass"""
        if use_cell_embedding is None:
            use_cell_embedding = self.use_cell_embedding
        return self.model(data_dict, use_cell_embedding=use_cell_embedding)

    def training_step(self, batch, batch_idx):
        """Training step"""
        # Update use_cell_embedding based on global step
        self.use_cell_embedding = self.USE_GENERATIVE_TRAINING and self.global_step > 1000
        
        loss_dict = self.forward(batch, use_cell_embedding=self.use_cell_embedding)
        
        # Log training metrics
        for key, value in loss_dict.items():
            self.log(f"train/{key}", value, on_step=True, on_epoch=False, prog_bar=True)
        
        return loss_dict["total_loss"]

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        loss_dict = self.forward(batch, use_cell_embedding=self.use_cell_embedding)
        
        # Log validation metrics
        for key, value in loss_dict.items():
            self.log(f"val/{key}", value, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return loss_dict

    def configure_optimizers(self):
        """Configure optimizer and scheduler"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if self.warmup_ratio_or_step > 0:
            # Calculate total training steps
            total_num_batches = len(self.train_dataloader()) * self.epochs
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

    def _get_dataloader(self, dataset, train: bool):
        """Create dataloader with appropriate sampler and collator"""
        # Setup sampler
        if self.balance_primary and train:
            sampler = get_balanced_sampler(
                dataset, 
                primary_condition=self.balance_primary, 
                secondary_condition=self.balance_secondary, 
                oversample=False
            )
        else:
            sampler = RandomSampler(dataset) if train else SequentialSampler(dataset)

        # Setup collator
        collator = AnnDataCollator(
            do_padding=self.max_seq_len is not None,
            pad_token_id=self.pad_token_id,
            pad_value=self.pad_value,
            do_mlm=True,
            do_binning=self.input_style == "binned",
            mlm_probability=self.mask_ratio,
            mask_value=self.mask_value,
            max_length=self.max_seq_len,
            sampling=self.TRUNC_BY_SAMPLE,
            data_style=self.training_tasks,
            n_bins=self.n_bins if self.input_style == "binned" else None,
            conditions=self.conditions,
            zero_percentages=self.zero_percentages,
        )

        batch_size = self.batch_size if train else self.eval_batch_size

        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=collator,
            drop_last=train,
            num_workers=min(len(os.sched_getaffinity(0)), batch_size),
            pin_memory=True,
        )

    def train_dataloader(self):
        """Create training dataloader"""
        return self._get_dataloader(self.train_dataset, train=True)

    def val_dataloader(self):
        """Create validation dataloader"""
        return self._get_dataloader(self.val_dataset, train=False)

    def load_pretrained_weights(self, pretrained_model_path: str, verbose: bool = True):
        """Load pretrained weights"""
        if pretrained_model_path.endswith(".safetensors"):
            tensors = {}
            with safe_open(pretrained_model_path, framework="pt", device="cpu") as f:
                for k in f.keys():
                    tensors[k] = f.get_tensor(k)
        elif pretrained_model_path.endswith(".pth") or pretrained_model_path.endswith(".pt"):
            tensors = torch.load(pretrained_model_path, map_location="cpu")
        else:
            raise ValueError("Unsupported file format. Use .safetensors, .pth, or .pt")
        
        return load_pretrained(self.model, tensors, verbose=verbose)

