import pytorch_lightning as pl
from collections import defaultdict
from pathlib import Path
import os
from typing import Dict, List, Optional, Union, Dict
from typing import Iterator, Optional
from operator import itemgetter
from .data_sampler import get_balanced_sampler
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, random_split

from .data_collator import AnnDataCollator
from .dataset import SingleCellDataset
from torch.utils.data import (
    Dataset,
    Sampler,
    DistributedSampler,
    WeightedRandomSampler
)
import numpy as np



class DatasetFromSampler(Dataset):

    def __init__(self, sampler: Sampler):
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        return len(self.sampler)


class DistributedSamplerWrapper(DistributedSampler):
    """ Convert any Pytorch Sampler to a DistributedSampler """

    def __init__(
        self,
        sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
    ):
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self) -> Iterator[int]:
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))


class SingleCellDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path,
        zero_percentages: list,
        batch_size: int,
        conditions: Dict,
        balance_primary,
        balance_secondary,
        max_seq_len: int,
        input_style: str,
        mask_ratio: float,
        TRUNC_BY_SAMPLE: bool,
        training_tasks: str,
        n_bins: int,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.data_path = data_path
        self.batch_size = batch_size
        self.conditions = conditions
        self.balance_primary = balance_primary
        self.balance_secondary = balance_secondary
        self.max_seq_len = max_seq_len
        self.input_style = input_style
        self.mask_ratio = mask_ratio
        self.TRUNC_BY_SAMPLE = TRUNC_BY_SAMPLE
        self.training_tasks = training_tasks
        self.n_bins = n_bins
        self.zero_percentages = zero_percentages
        
        
        # Setup token values based on embedding style
        if self.input_style == "category":
            self.mask_value = self.n_bins + 1
            self.pad_value = self.n_bins  # for padding gene expr values
            self.n_input_bins = self.n_bins + 2
        else:
            self.mask_value = -1
            self.pad_value = -2
            self.n_input_bins = self.n_bins
        

    def setup(self, stage: str):
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
        self.train_dataset, self.val_dataset = random_split(self.dataset, [1-0.05, 0.05])
        self.vocab = self.dataset.vocab
        
        self.pad_token_id = self.vocab["<pad>"]
        self.cls_token_id = self.vocab["<cls>"]

    def _get_dataloader(self, dataset, train: bool):
        """Create dataloader with appropriate sampler and collator"""
        # Setup sampler
        if self.balance_primary and train:
            sampler = get_balanced_sampler(
                dataset, 
                primary_condition=self.balance_primary, 
                secondary_condition=self.balance_secondary, 
                oversample=True
            )
        else:
            sampler = RandomSampler(dataset) if train else SequentialSampler(dataset)

        
        if self.trainer.world_size > 1:
            sampler = DistributedSamplerWrapper(
                dataset,
                sampler,
                num_replicas=self.trainer.world_size,
                rank=self.trainer.global_rank,
                shuffle=True,
                drop_last=True)
        
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

        batch_size = self.batch_size if train else self.batch_size
        if train:
            num_workers = min(12, min(len(os.sched_getaffinity(0)), self.batch_size))
        else: 
            num_workers = 12
        print(f"Using {num_workers} workers.")
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=collator,
            drop_last=train,
            num_workers=num_workers,
            pin_memory=True,
        )

    def train_dataloader(self):
        """Create training dataloader"""
        return self._get_dataloader(self.train_dataset, train=True)

    def val_dataloader(self):
        """Create validation dataloader"""
        return self._get_dataloader(self.val_dataset, train=False)
