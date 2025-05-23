import json
import sys 
from accelerate import Accelerator
import os 

from accelerate import DistributedDataParallelKwargs
sys.path.insert(0, "../")
from utils import get_args
from cancerfoundation.loss import LossType
from cancerfoundation.model.model import TransformerModel
from cancerfoundation.trainer import LightningModule
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

def train_model(
    lightning_module: LightningModule,
    max_epochs: int,
    save_dir: str,
    num_nodes: int = 1,
    gpus: int = 4,
    wandb_project: str = None,
    resume_from_checkpoint: str = None,
    precision: str = "bf16",
    strategy: str = "auto",    
    gradient_clip_val: float = 1.0,
    accumulate_grad_batches: int = 1,
    check_val_every_n_epoch: int = 1,
    pretrained_model_path: str = None,
):
    """
    Train the model using PyTorch Lightning Trainer
    
    Args:
        lightning_module: The LightningModule to train
        max_epochs: Maximum number of epochs
        save_dir: Directory to save checkpoints
        wandb_project: Wandb project name for logging
        resume_from_checkpoint: Path to checkpoint to resume from
        accelerator: Accelerator type ('cpu', 'gpu', 'tpu', 'auto')
        devices: Number of devices to use ('auto', int, or list)
        strategy: Training strategy ('auto', 'ddp', 'deepspeed', etc.)
        precision: Precision ('16-mixed', '32', 'bf16-mixed')
        accumulate_grad_batches: Number of batches to accumulate gradients
        gradient_clip_val: Gradient clipping value
        check_val_every_n_epoch: Validation frequency
        pretrained_model_path: Path to pretrained model weights
    """
    
    # Load pretrained weights if provided
    if pretrained_model_path:
        lightning_module.load_pretrained_weights(pretrained_model_path)
    
    # Setup callbacks
    callbacks = []
    
    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        filename='epoch_{epoch:02d}-val_loss_{val/total_loss:.2f}',
        monitor='val/total_loss',
        mode='min',
        save_top_k=3,
        save_last=True,
        auto_insert_metric_name=False,
    )
    callbacks.append(checkpoint_callback)
    
    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    # Setup logger
    logger = None
    if wandb_project:
        logger = WandbLogger(
            project=wandb_project,
            entity="cancerfoundation",
            config=lightning_module.args,
        )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu",
        devices=gpus,
        num_nodes=num_nodes,
        strategy=strategy,
        precision=precision,
        accumulate_grad_batches=accumulate_grad_batches,
        gradient_clip_val=gradient_clip_val,
        check_val_every_n_epoch=check_val_every_n_epoch,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=50,
        enable_progress_bar=True,
        enable_model_summary=True,
    )
    
    # Start training
    trainer.fit(
        lightning_module,
        ckpt_path=resume_from_checkpoint
    )
    
    return trainer



def main():
    args = get_args()
    trainer = LightningModule(
        args=args,
        n_bins=args.n_bins,
        input_emb_style=args.input_emb_style,
        max_seq_len=args.max_seq_len,
        input_style=args.input_style,
        mask_ratio=args.mask_ratio,
        TRUNC_BY_SAMPLE=args.trunc_by_sample,
        training_tasks=args.training_tasks,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        embsize=args.embsize,
        nheads=args.nheads,
        d_hid=args.d_hid,
        nlayers=args.nlayers,
        dropout=args.dropout,
        lr=args.lr,
        warmup_ratio_or_step=args.warmup_ratio_or_step,
        scheduler_interval=args.scheduler_interval,
        scheduler_factor=args.scheduler_factor,
        loss_type=args.loss,
        do_dat=args.do_dat,
        conditions = args.conditions,
        mvc_decoder_style=args.mvc_decoder_style,
        scale_zero_expression=args.scale_zero_expression,
        data_path=args.train_path,
        zero_percentages=args.zero_percentages,
        balance_primary=args.balance_primary,
        balance_secondary=args.balance_secondary,
    )
    
    train_model(
    lightning_module=trainer,
    max_epochs=1,
    num_nodes=args.num_nodes,
    gpus=args.gpus,
    save_dir="./checkpoints",
    wandb_project="cancer_foundation",
    strategy=args.strategy,
    precision="bf16-mixed",
)



if __name__ == "__main__":
    main()
