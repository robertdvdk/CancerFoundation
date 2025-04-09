from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F

class LossType(Enum):
    ORDINALCROSSENTROPY = "ordinal_cross_entropy"
    CORN = "corn"
    MSE = "mse"

def compute_weights(num_classes: int, scale_zero_expr: Optional[float]):
    if scale_zero_expr and num_classes != None:
        down_scale_pos_expr = (num_classes - num_classes * scale_zero_expr) / (num_classes - 1)
        return torch.tensor([num_classes * scale_zero_expr] + [down_scale_pos_expr] * (num_classes - 1))
    return None

class AbstractGeneExpressionLoss(nn.Module, ABC):
    """
    Abstract class for computing gene expression loss based on different loss types.

    Args:
        num_classes (int): The number of classes.
        loss_type (LossType): Type of loss function to be used.
        scale_zero_expression (Optional[float], optional): Scaling factor for zero expression values. Defaults to None (uniform weights).
                0.X means that 0.X weight is placed on predicting zero expression values opposed to 1/num_classes
    Attributes:
        _in_dim (int): The dimension the loss function expects.

    """
    def __init__(self, num_classes: Optional[int] = None, scale_zero_expression: Optional[float] = None):
        super().__init__()
        if scale_zero_expression:
            assert 0 <= scale_zero_expression <= 1, "scale_zero_expression must be between 0 and 1"
        self.weight = compute_weights(num_classes=num_classes, scale_zero_expr=scale_zero_expression)
        self.num_classes = num_classes

    @abstractmethod
    def get_in_dim(self) -> int:
        pass

    @abstractmethod
    def forward(self, logits: torch.Tensor, target: torch.Tensor, mask: torch.LongTensor) -> torch.Tensor:
        pass


def masked_mse_loss(
    input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute the masked MSE loss between input and target.
    """
    mask = mask.float()
    loss = F.mse_loss(input * mask, target * mask, reduction="sum")
    return loss / mask.sum()


def criterion_neg_log_bernoulli(
    input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute the negative log-likelihood of Bernoulli distribution
    """
    mask = mask.float()
    bernoulli = torch.distributions.Bernoulli(probs=input)
    masked_log_probs = bernoulli.log_prob((target > 0).float()) * mask
    return -masked_log_probs.sum() / mask.sum()


def masked_relative_error(
    input: torch.Tensor, target: torch.Tensor, mask: torch.LongTensor
) -> torch.Tensor:
    """
    Compute the masked relative error between input and target.
    """
    assert mask.any()
    loss = torch.abs(input[mask] - target[mask]) / (target[mask] + 1e-6)
    return loss.mean()

class MSE(AbstractGeneExpressionLoss):
    def __init__(self, num_classes: Optional[int] = None, scale_zero_expression: Optional[float] = None):
        super().__init__(num_classes, scale_zero_expression)
        assert not scale_zero_expression, "Scaling zero expression is not defined for MSE loss. Change the loss function or set scale_zero_expression to None"

    def forward(self, logits: torch.Tensor, target: torch.Tensor, mask: torch.LongTensor) -> torch.Tensor:
        mask = mask.float()
        loss = F.mse_loss(logits * mask, target * mask, reduction="sum")
        return loss / mask.sum()
    
    def get_in_dim(self) -> int:
        return 1
    
class OrdinalCrossEntropy(AbstractGeneExpressionLoss):
    def __init__(self, num_classes: Optional[int] = None, scale_zero_expression: Optional[float] = None):
        super().__init__(num_classes, scale_zero_expression)
        assert isinstance(num_classes, int) and num_classes > 0, "num_classes must be an integer and positive."

    def forward(self, logits: torch.Tensor, target: torch.Tensor, mask: torch.LongTensor) -> torch.Tensor:
        """
        Args:
            logits (torch.Tensor): A tensor of shape (batch_size, seq_length, num_classes)
                representing raw logits from the model.
            target (torch.Tensor): A tensor of shape (batch_size, seq_length) containing
                the true labels (0 to num_classes-1).
            mask (torch.Tensor): A tensor of shape (batch_size, seq_length) containing
                the mask for ignoring certain elements in the loss computation.

        Returns:
            torch.Tensor: A scalar tensor representing the calculated ordinal cross-entropy loss.
        """
        
        class_range = torch.arange(self.num_classes, device=logits.device).reshape(1, 1, self.num_classes)

        # Expand target to match the shape for broadcasting
        expanded_target = target.unsqueeze(-1)  # Add an extra dimension for classes

        # Perform the comparison in a vectorized manner
        cum_targets = (expanded_target >= class_range).float()

        # Apply sigmoid to logits to get probabilities
        probs = torch.sigmoid(logits)

        # Binary cross-entropy loss across all ordinal binary tasks
        mask = mask.float().unsqueeze(-1).expand_as(probs)
        loss = F.binary_cross_entropy(probs, cum_targets, reduction='none')

        loss = F.binary_cross_entropy(probs, cum_targets, weight=self.weight.to(logits.device) if self.weight != None else None, reduction='none')

        loss = (loss * mask).sum() / mask.sum()

        return loss
    
    def get_in_dim(self) -> int:
        return self.num_classes

class CORN(AbstractGeneExpressionLoss):
    """
    Deep Neural Networks for Rank-Consistent Ordinal Regression Based On Conditional Probabilities
    https://arxiv.org/pdf/2111.08851
    """
    def __init__(self, num_classes: Optional[int] = None, scale_zero_expression: Optional[float] = None):
        super().__init__(num_classes, scale_zero_expression)
        assert isinstance(num_classes, int) and num_classes > 0, "num_classes must be an integer and positive."
    
    def forward(self, logits: torch.Tensor, target: torch.Tensor, mask: torch.LongTensor) -> torch.Tensor:
        """
        Args:
            logits (torch.Tensor): A tensor of shape (batch_size, seq_length, num_classes-1)
                representing raw logits from the model.
            target (torch.Tensor): A tensor of shape (batch_size, seq_length) containing
                the true labels (0 to num_classes-1).
            mask (torch.Tensor): A tensor of shape (batch_size, seq_length) containing
                the mask for ignoring certain elements in the loss computation.

        Returns:
            torch.Tensor: A scalar tensor representing the calculated corn loss.
        """
        _, _, num_classes_minus_one = logits.shape
        mask= mask.reshape(-1)
        logits = logits.reshape(-1, num_classes_minus_one)[mask]
        target = target.reshape(-1)[mask]

        sets = []
        for i in range(self.num_classes-1):
            label_mask = target > i-1
            label_tensor = (target[label_mask] > i).to(torch.int64)
            sets.append((label_mask, label_tensor))

        num_examples = 0
        losses = 0.

        if self.weight is None:
            importance_weights = torch.ones(len(sets), device=logits.device)
        else:
            importance_weights = self.weight.to(logits.device)

        for task_index, s in enumerate(sets):
            train_examples = s[0]
            train_labels = s[1]

            if len(train_labels) < 1:
                continue

            num_examples += len(train_labels)
            pred = logits[train_examples, task_index]

            loss = -torch.sum(F.logsigmoid(pred)*train_labels
                            + (F.logsigmoid(pred) - pred)*(1-train_labels))

            losses += importance_weights[task_index] * loss

        return losses / num_examples if num_examples > 0 else torch.tensor(0.0, device=logits.device)
    
    def get_in_dim(self) -> int:
        return self.num_classes-1
    

loss_dict = {
    LossType.ORDINALCROSSENTROPY: OrdinalCrossEntropy,
    LossType.CORN: CORN,
    LossType.MSE: MSE
}

def get_loss(loss_type: LossType, num_classes: Optional[int] = None, scale_zero_expression: Optional[float] = None):
    if loss_type in loss_dict:
        return loss_dict[loss_type](num_classes, scale_zero_expression)
    else:
        raise ValueError(f"No loss found for type {loss_type}")