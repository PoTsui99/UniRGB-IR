import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.core import weight_reduce_loss
from mmseg.ops import softmax
from torch.autograd import Variable

class SoftmaxCrossEntropyOHEMLoss(nn.Module):
    def __init__(self, ignore_index=-1, thresh=0.7, min_kept=256, use_weight=True, reduction='mean', loss_name='loss_ohem',**kwargs):
        super(SoftmaxCrossEntropyOHEMLoss, self).__init__()
        self.ignore_index = ignore_index
        self.thresh = thresh
        self.min_kept = min_kept
        self.reduction = reduction
        self.loss_name_ = loss_name
        self.class_weight = None
        if use_weight:
            # 示例权重，根据实际情况调整
            self.class_weight = torch.FloatTensor([1.4543, 43.8739, 34.2410, 47.3663, 27.4869])

    def forward(self, pred, target):
        """Forward function."""
        assert pred.size(0) == target.size(0)
        assert pred.dim() == 4  # [N, C, H, W]
        assert target.dim() == 3  # [N, H, W]
        n, c, h, w = pred.size()

        # 转换为 softmax 概率
        pred_softmax = softmax(pred, dim=1)
        
        # 使用 OHEM 策略选择困难样本
        pred_scores, pred_classes = pred_softmax.max(dim=1)
        mask = target != self.ignore_index
        target = target[mask]
        pred_scores = pred_scores[mask]
        pred_classes = pred_classes[mask]

        # 根据阈值和 min_kept 过滤样本
        if self.min_kept > 0:
            _, idx = pred_scores.sort()
            threshold_idx = max(self.min_kept, idx[self.min_kept - 1])
            threshold = pred_scores[threshold_idx]
            keep_mask = pred_scores <= max(threshold, self.thresh)
        else:
            keep_mask = pred_scores <= self.thresh

        # 过滤后的目标和预测
        target = target[keep_mask]
        pred = pred.permute(0, 2, 3, 1).reshape(-1, c)[keep_mask, :]

        # 计算损失
        if self.class_weight is not None:
            self.class_weight = self.class_weight.to(pred.device)
        loss = F.cross_entropy(pred, target, weight=self.class_weight, ignore_index=self.ignore_index, reduction=self.reduction)
        
        return loss
