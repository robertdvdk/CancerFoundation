import json
import sys 
from accelerate import Accelerator
import os
from typing import Optional

sys.path.insert(0, "../")
from utils import get_args, MyProgressBar
from cancerfoundation.model.model import CancerFoundation
from cancerfoundation.data.data_module import SingleCellDataModule
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

def train_model(
    model: CancerFoundation,
    datamodule: pl.LightningDataModule,
    max_epochs: int,
    save_dir: str,
    num_nodes: int = 1,
    gpus: int = 4,
    wandb_project: Optional[str] = None,
    wandb_entitiy: Optional[str] = None,
    resume_from_checkpoint: Optional[str] = None,
    precision: str = "bf16",
    strategy: str = "auto",    
    gradient_clip_val: float = 1.0,
    accumulate_grad_batches: int = 1,
    check_val_every_n_epoch: int = 1,
    val_check_interval: float = 1.,
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
    
    
    # Setup callbacks
    callbacks = []
    callbacks.append(MyProgressBar(refresh_rate=5))
    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        filename='epoch_{epoch:02d}',
        every_n_epochs=1
    )
    callbacks.append(checkpoint_callback)
    
    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    # Setup logger
    logger = None
    if wandb_project:
        logger = WandbLogger(
            entity=wandb_entitiy,
            project=wandb_project,
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
        val_check_interval=val_check_interval,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=5,
        enable_progress_bar=True,
        enable_model_summary=True,
        limit_train_batches=10, limit_val_batches=5
    )
    
    # Start training
    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=resume_from_checkpoint
    )
    
    return trainer



def main():
    args = get_args()
    
    
    datamodule = SingleCellDataModule(
        data_path=args.train_path,
        zero_percentages=args.zero_percentages,
        batch_size=args.batch_size,
        conditions=args.conditions,
        balance_primary=args.balance_primary,
        balance_secondary=args.balance_secondary,
        max_seq_len=args.max_seq_len,
        input_style=args.input_style,
        mask_ratio=args.mask_ratio,
        TRUNC_BY_SAMPLE=args.trunc_by_sample,
        training_tasks=args.training_tasks,
        n_bins=args.n_bins
    )
    datamodule.setup(stage="fit")
    if args.resume_from_checkpoint:
        model = CancerFoundation.load_from_checkpoint(args.resume_from_checkpoint, vocab=datamodule.vocab)
    else:
        model = CancerFoundation(
            n_bins=args.n_bins,
            vocab=datamodule.vocab,
            input_emb_style=args.input_emb_style,
            max_seq_len=args.max_seq_len,
            input_style=args.input_style,
            mask_ratio=args.mask_ratio,
            TRUNC_BY_SAMPLE=args.trunc_by_sample,
            training_tasks=args.training_tasks,
            embsize=args.embsize,
            nheads=args.nheads,
            d_hid=args.d_hid,
            nlayers=args.nlayers,
            dropout=args.dropout,
            lr=args.lr,
            epochs=args.epochs,
            warmup_ratio_or_step=args.warmup_ratio_or_step,
            scheduler_interval=args.scheduler_interval,
            scheduler_factor=args.scheduler_factor,
            loss_type=args.loss,
            do_dat=args.do_dat,
            conditions = args.conditions,
            conditions_nums = datamodule.conditions_nums if args.conditions else None,
            mvc_decoder_style=args.mvc_decoder_style,
            scale_zero_expression=args.scale_zero_expression,
            data_path=args.train_path,
            zero_percentages=args.zero_percentages,
            balance_primary=args.balance_primary,
            balance_secondary=args.balance_secondary,
        )
    
    if args.pretrained:
        print(f"Loading pretrained weights from {args.pretrained}.")
        vocab_pretrained = json.load(open( args.pretrained / "vocab.json", "r"))
        gene_mapping = {}
        for key, value in datamodule.vocab.items():
            if key in vocab_pretrained:
                gene_mapping[value] = vocab_pretrained[key]
        model.load_pretrained_weights(args.pretrained / "best_model.pt", gene_mapping=gene_mapping)
   
    train_model(
        model=model,
        datamodule=datamodule,
        max_epochs=args.epochs,
        num_nodes=args.num_nodes,
        gpus=args.gpus,
        save_dir=args.save_dir,
        resume_from_checkpoint=args.resume_from_checkpoint,
        val_check_interval=args.val_check_interval,
        wandb_project=args.wandb,
        wandb_entitiy=args.wandb_entity,
        accumulate_grad_batches=args.grad_accu_steps,
        strategy=args.strategy,
        precision="bf16-mixed",
)



if __name__ == "__main__":
    main()
