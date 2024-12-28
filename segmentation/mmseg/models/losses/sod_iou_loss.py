# by tusipo, 24.4.17
# special for SOD tasks
import torch
import torch.nn.functional as F
import torch.nn as nn

from mmseg.registry import MODELS
from .utils import weight_reduce_loss

def iou_loss(pred, target, weight=None, reduction='mean', avg_factor=None):
    """iou_loss. Calculate the Intersection over Union (IoU) loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C, H, W) before sigmoid.
        target (torch.Tensor): The ground truth with shape (N, H, W).
        weight (torch.Tensor, optional): Sample-wise loss weight. Default: None.
        reduction (str, optional): The method used to reduce the loss.
            Options are 'none', 'mean', and 'sum'. Default: 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Default: None.

    Returns:
        torch.Tensor: The calculated IoU loss.
    """
    # Apply sigmoid to get binary probability distribution
    pred = torch.sigmoid(pred)

    # Flatten the tensors to treat all pixels equally
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)


    # Calculate intersection and union
    intersection = (pred_flat * target_flat).sum(-1)
    total = (pred_flat + target_flat).sum(-1)
    union = total - intersection

    # Calculate IoU and then the IoU loss
    iou = intersection / (union + 1e-6)
    loss = 1 - iou

    # Apply weights and reduction
    if weight is not None:
        loss = loss * weight.view(loss.size(0), -1)
    loss = weight_reduce_loss(loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss

@MODELS.register_module()
class SODIoULoss(nn.Module):
    """IoULoss

    Args:
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean", and "sum".
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        loss_name (str, optional): Name of the loss item. Defaults to 'loss_iou'.
    """

    def __init__(self, reduction='mean', loss_weight=1.0, loss_name='loss_iou'):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self._loss_name = loss_name

    def forward(self, pred, target, weight=None, avg_factor=None, **kwargs):
        """Forward function."""
        loss = self.loss_weight * iou_loss(
            pred,
            target,
            weight=weight,
            reduction=self.reduction,
            avg_factor=avg_factor
        )
        return loss

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
