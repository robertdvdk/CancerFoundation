import argparse
import sys

sys.path.insert(0, "../")
from cancerfoundation.loss import LossType

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--save-dir",
        type=str,
        help="The directory to save the trained model and the results.",
    )

    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        help="The directory to load the checkpoint.",
    )
    parser.add_argument(
        "--pretrained-model-path",
        type=str,
        default=None,
        help="Path to the pretrained model. Use for finetuning."
    )

    # settings for data
    parser.add_argument(
        "--n-hvg",
        type=int,
        default=None,
        help="The number of highly variable genes. If set to 0, will use all genes. "
        "Default is None, which will determine the n_hvg automatically.",
    )

    parser.add_argument(
        "--grad-accu-steps",
        type=int,
        default=1,
        help="The number of gradient accumulation steps. Default is 1.",
    )

    parser.add_argument(
        "--loss",
        type=LossType,
        default=LossType.MSE,
        help="The loss function used for the gene expression prediction. Default is ordinal_cross_entropy (Ordinal Cross Entropy)."
    )
    parser.add_argument(
        "--input-style",
        type=str,
        choices=["normed_raw", "log1p", "binned"],
        default="binned",
        help="The style of the input data. Default is binned.",
    )
    parser.add_argument(
        "--input-emb-style",
        type=str,
        choices=["category", "continuous", "scaling"],
        default="continuous",
        help="The style of the input embedding. Default is continuous.",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=51,
        help="The number of bins to use for the binned input style. Default is 51.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=1200,
        help="The maximum length of the sequence. Default is 1200. The actual used "
        "max length would be the minimum of this value and the length of the longest "
        "sequence in the data.",
    )
    parser.add_argument(
        "--training-tasks",  # choices of "pcpt", "gen", "both"
        type=str,
        default="both",
        choices=["pcpt", "gen", "both"],
        help="The tasks to use for training. pcpt: perception training with maked token "
        "learning. gen: generation. Default is both.",
    )
    parser.add_argument(
        "--mask-ratio",
        type=float,
        default=0.40,
        help="The ratio of masked values in the training data. Default is 0.40. This"
        "value will be ignored if --training-tasks is set to gen or both.",
    )
    parser.add_argument(
        "--trunc-by-sample",
        action="store_true",
        help="Whether to truncate the input by sampling rather than cutting off if "
        "sequence length > max_seq_length. Default is False.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="The batch size for training. Default is 32.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=32,
        help="The batch size for evaluation. Default is 32.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="The number of epochs for training.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="The learning rate for training. Default is 1e-3.",
    )
    parser.add_argument(
        "--scheduler-interval",
        type=int,
        default=100,
        help="The interval iterations for updating the learning rate. Default is 100. "
        "This will only be used when warmup-ratio is 0.",
    )
    parser.add_argument(
        "--scheduler-factor",
        type=float,
        default=0.99,
        help="The factor for updating the learning rate. Default is 0.99. "
        "This will only be used when warmup-ratio is 0.",
    )
    parser.add_argument(
        "--warmup-ratio-or-step",
        type=float,
        default=0.1,
        help="The ratio of warmup steps out of the total training steps. Default is 0.1. "
        "If warmup-ratio is above 0, will use a cosine scheduler with warmup. If "
        "the value is above 1, will use it as the number of warmup steps.",
    )
    parser.add_argument(
        "--no-cls",
        action="store_true",
        help="Whether to deactivate the classification loss. Default is False.",
    )
    # settings for model
    parser.add_argument(
        "--nlayers",
        type=int,
        default=4,
        help="The number of layers for the transformer. Default is 4.",
    )
    parser.add_argument(
        "--nheads",
        type=int,
        default=4,
        help="The number of heads for the transformer. Default is 4.",
    )
    parser.add_argument(
        "--embsize",
        type=int,
        default=64,
        help="The embedding size for the transformer. Default is 64.",
    )
    parser.add_argument(
        "--d-hid",
        type=int,
        default=64,
        help="dimension of the feedforward network model in the transformer. "
        "Default is 64.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="The dropout rate. Default is 0.2.",
    )
    # settings for logging
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="The interval for logging. Default is 100.",
    )
    parser.add_argument(
        "--wandb",
        type=str,
        default=None,
        help="The project name for wandb. If set to None, no logging will occur.",
    )
    parser.add_argument(
        "--conditions",
        nargs='+',
        default=None,
        help="The conditions (obs keys) the model should be invariant to.",
    )
    parser.add_argument(
        "--scale-zero-expression",
        type=float,
        default=None,
        help="How much weight should be placed on predicting if gene expression is above 0. If None, equal weighting is used."
    )
    parser.add_argument(
        "--mvc-decoder-style",
        type=str,
        default="inner product",
        choices=["concat query", "inner product"]
    )
    parser.add_argument(
        "--train-path",
        type=str,
        default=None,
        help="Path to the training data."
    )
    parser.add_argument(
        "--eval-path",
        type=str,
        default=None,
        help="Path to the evaluation data."
    )
    parser.add_argument(
        "--do-dat",
        action="store_true",
        help="Whether or not to do domain adversarial training on the conditions."
    )
    parser.add_argument(
        "--vocab",
        type=str,
        help="Path to the list of genes.",
        default=None,
    )
    parser.add_argument(
        "--balance-primary",
        type=str,
        help="According to which metadata (primary) one should oversample to make the data more balanced.",
        default=None,
    )
    parser.add_argument(
        "--balance-secondary",
        type=str,
        help="According to which metadata (secondary) one should oversample to make the data more balanced.",
        default=None,
    )
    parser.add_argument(
        "--zero-percentages",
        nargs='+',
        type=float,
        default=None,
        help="The percentage of zero-expressed genes sampled.",
    )

    return parser.parse_args()