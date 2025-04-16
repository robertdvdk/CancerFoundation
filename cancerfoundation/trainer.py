from collections import defaultdict
import time
import os
from typing import Dict, List, Optional, Union
from .data_sampler import get_balanced_sampler
import transformers
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch import nn
import torch
import json

from .data_collator import AnnDataCollator
from .dataset import SingleCellDataset
from .utils import load_pretrained
from .model import TransformerModel
from cancerfoundation.loss import criterion_neg_log_bernoulli, get_loss, masked_relative_error
import numpy as np
from safetensors import safe_open
from .loss import LossType

from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import random_split
from accelerate.utils.tqdm import tqdm

def with_sdp_kernel(func):
    def wrapped_func(*args, **kwargs):
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True,
            enable_math=True,
            enable_mem_efficient=True
        ):
            return func(*args, **kwargs)
    return wrapped_func


class Trainer:
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
        save_dir: str,
        data_path: Union[str, os.PathLike],
        loss_type: LossType = LossType.MSE,
        resume_from_checkpoint: str = None,
        wandb: str = None,
        conditions: List[str] = None,
        mvc_decoder_style: str = "inner product",
        scale_zero_expression: Optional[float] = None,
        accelerator = None,
        do_dat: bool = False,
        explicit_zero_prob: Optional[bool] = False,
        balance_primary: Optional[str] = None,
        balance_secondary: Optional[str] = None,
        zero_percentages: Optional[List[float]] = None,
    ):
        self.args = args
        self.n_bins = n_bins
        self.input_emb_style = input_emb_style
        self.max_seq_len = max_seq_len
        self.input_style = input_style
        self.mask_ratio = (
            [0.25, 0.50, 0.75] if training_tasks in [
                "gen", "both"] else mask_ratio
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
        self.save_dir = save_dir
        self.accelerator = accelerator
        self.loss_type = loss_type
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

        self.resume_from_checkpoint = resume_from_checkpoint
        self.wandb = wandb
        self.timer = None

        if self.input_emb_style == "category":
            self.mask_value = self.n_bins + 1
            self.pad_value = self.n_bins  # for padding gene expr values
            self.n_input_bins = self.n_bins + 2
        else:
            self.mask_value = -1
            self.pad_value = -2
            self.n_input_bins = self.n_bins

        self.best_val_loss = {"loss": float("inf"), "epoch": -1}
        self.starting_epoch = 0

        self.model = None
        self.has_setup_trainer = False
        self.conditions = conditions
        self.conditions_nums = None
        
        if balance_primary is None and balance_secondary is not None:
            raise ValueError("balance_secondary is not allowed to be set (not None) if balance_primary is None.")
        self.balance_primary = balance_primary
        self.balance_secondary = balance_secondary
        self.zero_percentages = zero_percentages

        self.scale_zero_expression = scale_zero_expression
        
        self.dataset = self.__create_datasets(data_path=data_path)
        
        if self.conditions:
            self.conditions_nums = {}
            for cond in self.conditions:
                self.conditions_nums[cond] = len(self.dataset.mapping[cond].keys())
        
        self.train_dataset, self.eval_dataset = random_split(self.dataset, [0.9, 0.1])
        self.vocab = self.dataset.vocab
        
        self.pad_token_id = self.vocab["<pad>"]
        self.cls_token_id = self.vocab["<cls>"]
        
        self.train_sampler = self._get_random_sampler(self.train_dataset, True)
        self.eval_sampler = self._get_random_sampler(self.eval_dataset, False)
        
        self.train_loader = self._get_dataloader(self.train_dataset, self.train_sampler, train=True)
        self.eval_loader = self._get_dataloader(self.eval_dataset, self.eval_sampler, train=False)
        
        self.criterion = get_loss(
            loss_type=loss_type, num_classes=self.n_input_bins if self.n_input_bins else None, scale_zero_expression=scale_zero_expression)

        self.__set_model(mvc_decoder_style=mvc_decoder_style, gene_expr_out_dim=self.criterion.get_in_dim(), ntoken=len(self.vocab.keys()), criterion=self.criterion)

    def __initiate_wandb(self, run_id: str = None):
        assert (self.resume_from_checkpoint != None) == (run_id != None)
        self.accelerator.init_trackers(
            project_name=self.wandb,
            config=self.args,
            init_kwargs={
                "wandb": (
                    {
                        "entity": "cancerfoundation",
                        "name": self.save_dir,
                        "resume": "allow",
                    }
                    if self.resume_from_checkpoint == None
                    else {"name": self.save_dir, "resume": "must", "id": run_id}
                )
            },
        )
    
    def __create_datasets(
        self,
        data_path: Union[str, os.PathLike],
        
    ):
        with self.accelerator.main_process_first():
            dataset = SingleCellDataset(
                data_dir=data_path,
                pad_value=self.pad_value,
                obs_columns=self.conditions
            )

        return dataset
    
    def _get_random_sampler(self, dataset, train):
        
         with self.accelerator.main_process_first():
            if self.balance_primary and train:
                sampler = get_balanced_sampler(dataset, primary_condition=self.balance_primary, secondary_condition=self.balance_secondary, oversample=False)
            else:
                sampler = RandomSampler(dataset) if train else SequentialSampler(dataset)
            return sampler
        
        
    def _get_dataloader(self, dataset, sampler, train: bool):
        with self.accelerator.main_process_first():
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
            num_workers=min(
                 len(os.sched_getaffinity(0)), batch_size//2), ## REPLACE LATER
            pin_memory=True,
        )

    def __set_model(self, mvc_decoder_style: str, gene_expr_out_dim: int, ntoken: int, criterion):

        self.model = TransformerModel(
            ntoken=ntoken,
            d_model=self.embsize,
            out_dim=gene_expr_out_dim,
            mvc_decoder_style=mvc_decoder_style,
            nhead=self.nheads,
            d_hid=self.d_hid,
            nlayers=self.nlayers,
            dropout=self.dropout,
            pad_token_id=self.pad_token_id,
            criterion=criterion,
            pad_value=self.pad_value,
            do_mvc=self.MVC,
            conditions=self.conditions_nums,
            input_emb_style=self.input_emb_style,
            n_input_bins=self.n_input_bins,
            use_generative_training=self.USE_GENERATIVE_TRAINING,
            do_dat=self.do_dat,
            explicit_zero_prob=self.explicit_zero_prob,
        )

    def accelerate(self):
        (
            self.model,
            self.optimizer,
            self.scheduler,
            self.train_loader,
            self.eval_loader,
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.scheduler,
            self.train_loader,
            self.eval_loader,
        )
        
            
    
    def __setup_training_variables(self, epochs: int) -> None:
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if self.warmup_ratio_or_step > 0:
            total_num_batches = len(self.train_loader) * epochs
            warmup_steps = (
                int(total_num_batches * self.warmup_ratio_or_step)
                if self.warmup_ratio_or_step < 1
                else int(self.warmup_ratio_or_step)
            )

            self.scheduler = transformers.get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_num_batches,
                last_epoch=-1,
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, self.scheduler_interval, gamma=self.scheduler_factor
            )

    def checkpoint(self, epoch: int):
        self.accelerator.print("Checkpointing...")
        path = f"{self.save_dir}/epoch_{epoch}"
        if self.accelerator.is_main_process:
            os.makedirs(path)
            os.makedirs(f"{path}/accelerate")
            with open(f"{path}/info.json", "w") as json_file:
                info = {
                    "epoch": epoch,
                    "best_val_loss": self.best_val_loss,
                    "run_id": (
                        self.accelerator.get_tracker("wandb").run.id
                        if self.wandb != None
                        else None
                    ),
                }
                json.dump(info, json_file, indent=4)

        self.accelerator.wait_for_everyone()
        self.accelerator.save_state(f"{path}/accelerate")
        self.accelerator.print("Checkpointing done!")

    def __log(self, metrics: Dict):

        for key, val in metrics.items():
            if hasattr(val, "item"):
                metrics[key] = val.item()
            else:
                metrics[key] = val
        if self.wandb != None:
            self.accelerator.log(metrics)
        else:
            if self.accelerator.is_main_process:
                path = f"{self.save_dir}/log.json"
                with open(path, "r") as file:
                    data = json.load(file)
                data.append(metrics)
                with open(path, "w") as file:
                    json.dump(data, file, indent=4)
        self.timer = None

    def load_model(self, pretrained_model_path: str, verbose: bool = True) -> Union[torch.nn.Module, None]:
        if pretrained_model_path.endswith(".safetensors"):
            tensors = {}
            with safe_open(pretrained_model_path, framework="pt", device="cpu") as f:
                for k in f.keys():
                    tensors[k] = f.get_tensor(k).to(self.accelerator.device) if self.accelerator else f.get_tensor(k)
        elif pretrained_model_path.endswith(".pth") or pretrained_model_path.endswith(".pt"):
            tensors = torch.load(pretrained_model_path)
        else:
            self.accelerator.print("Unsupported file format.")
            return None
        return load_pretrained(self.model, tensors, verbose=verbose)
    
    def setup_training(self, epochs: int, pretrained_model_path: Optional[str] = None) -> None:
        self.__setup_training_variables(epochs)

        if pretrained_model_path:
            self.model = self.load_model(pretrained_model_path)
            
        self.accelerate()
        if self.resume_from_checkpoint != None:
            self.accelerator.print(
                f"Resume from checkpoint: {self.resume_from_checkpoint}"
            )
            self.save_dir = self.resume_from_checkpoint.rsplit('/', 1)[0]
            self.accelerator.load_state(
                f"{self.resume_from_checkpoint}/accelerate")
            with open(f"{self.resume_from_checkpoint}/info.json", "r") as file:
                # Load the JSON data
                data = json.load(file)
            self.starting_epoch = data["epoch"] + 1
            self.best_val_loss = data["best_val_loss"]
        else:
            if self.accelerator.is_main_process:
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
                # Create the directory
                path = f"{self.save_dir}/log.json"
                with open(path, "w") as file:
                    json.dump([], file, indent=4)

        if self.wandb != None:
            run_id = None
            if self.resume_from_checkpoint != None:
                run_id = data["run_id"]
            self.__initiate_wandb(run_id=run_id)

        self.has_setup_trainer = True

    @with_sdp_kernel
    def train(self, epoch: int, log_interval: int) -> None:
        """
        Evaluate the model on the validation dataset.

        Args:
            epoch (int): The current epoch number.

        Returns:
            None
        """

        for step, data_dict in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
            self.model.train()

            global_iter = step + epoch * len(self.train_loader)
            self.use_cell_embedding = self.USE_GENERATIVE_TRAINING and global_iter > 1000
            
            loss_dict = self.model(data_dict,use_cell_embedding=self.use_cell_embedding)
            self.accelerator.backward(loss_dict["total_loss"])

            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(
                    self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            self.accelerator.log({"train/" + k:v for k,v in loss_dict.items()}, step=step)
            self.accelerator.log({"train/lr": self.scheduler.get_last_lr()[0]}, step=step)
            
                

    @with_sdp_kernel
    @torch.no_grad
    def evaluate(self, epoch: int) -> Dict[str, float]:
        self.model.eval()

        valid_loader = self.eval_loader
        all_loss_dict = defaultdict(list)
        for batch, data_dict in tqdm(enumerate(valid_loader), total=len(valid_loader)):
            loss_dict  = self.model(data_dict, use_cell_embedding=self.use_cell_embedding)
            for key, values in loss_dict.items():
                all_loss_dict[key].append(values)
        
        all_loss_dict = {k : self.accelerator.gather_for_metrics(torch.stack(v)).mean().item() for k, v in all_loss_dict.items()}
        self.accelerator.log({"eval/" + k:v for k,v in all_loss_dict.items()}, step=epoch)
    
            
