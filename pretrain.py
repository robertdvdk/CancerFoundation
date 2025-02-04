import json
import sys 
from accelerate import Accelerator
import os 

from accelerate import DistributedDataParallelKwargs



if __name__ == "__main__":
    sys.path.insert(0, "../")
    from utils import get_args
    from cancerfoundation.trainer import Trainer


    args = get_args()
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(gradient_accumulation_steps=args.grad_accu_steps, log_with="wandb", kwargs_handlers=[ddp_kwargs])

    trainer = Trainer(
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
        save_dir=args.save_dir,
        accelerator=accelerator,
        vocab=args.vocab,
        loss_type=args.loss,
        resume_from_checkpoint=args.resume_from_checkpoint,
        wandb = args.wandb,
        conditions = args.conditions,
        mvc_decoder_style=args.mvc_decoder_style,
        scale_zero_expression=args.scale_zero_expression,
        train_data_path=args.train_path,
        eval_data_path=args.eval_path,
        zero_percentages=args.zero_percentages,
        balance_primary=args.balance_primary,
        balance_secondary=args.balance_secondary,
    )

    epochs=args.epochs
    trainer.setup_training(epochs=epochs, pretrained_model_path=args.pretrained_model_path)

    if accelerator.is_main_process:
        with open(f"{trainer.save_dir}/args.json", "w") as file:
            args_ = args
            args_.loss = args.loss.value
            json.dump(vars(args_), file, indent=4)

    for epoch in range(trainer.starting_epoch, epochs):
        accelerator.print(f"Epoch: {epoch}")
        accelerator.print("Training...")
        trainer.train(epoch, log_interval=args.log_interval)
        
        accelerator.print("Evaluating...")
        trainer.evaluate(epoch=epoch)

        trainer.checkpoint(epoch)
    
    accelerator.end_training()
