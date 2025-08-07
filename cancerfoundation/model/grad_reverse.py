import torch
from torch.autograd import Function
from typing import Tuple, cast


class GradientReverse(Function):
    scale = 1.0

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return x

    @staticmethod
    def backward(ctx, *grad_outputs: torch.Tensor) -> Tuple[None, torch.Tensor]:
        grad_output = grad_outputs[0]
        return None, GradientReverse.scale * grad_output.neg()


def grad_reverse(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    GradientReverse.scale = scale
    return cast(torch.Tensor, GradientReverse.apply(x))
