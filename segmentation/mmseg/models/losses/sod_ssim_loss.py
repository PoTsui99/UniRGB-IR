import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS
from .utils import weight_reduce_loss
from math import exp

def create_window(window_size, channel, sigma=1.5):
    """ Helper function to create a Gaussian window for SSIM computation. """
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    gauss = gauss / gauss.sum()
    _1D_window = gauss.unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def _ssim(img1, img2, window, window_size, channel):
    """ The core SSIM computation using convolution operations. """
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map

@MODELS.register_module()
class SODSSIMLoss(nn.Module):
    """ Structural Similarity Index Measure as a loss function. """
    def __init__(self, window_size=11, size_average=True, loss_weight=1.0, reduction='mean'):
        super(SODSSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.channel = 1
        self.register_buffer('window', create_window(self.window_size, self.channel))

    def forward(self, pred, label, **kwargs):  # sucks any additional args, e.g. weight(class-weight)
        pred = pred.unsqueeze(1) if pred.dim() == 3 else pred
        label = label.unsqueeze(1) if label.dim() == 3 else label

        """ Forward method of SSIM loss. """
        if pred.is_cuda:
            self.window = self.window.cuda(pred.get_device())
        self.window = self.window.type_as(pred)

        pred = torch.sigmoid(pred)  # Apply sigmoid to predictions
        ssim_map = _ssim(pred, label, self.window, self.window_size, pred.size(1))
        loss = (1 - ssim_map).mean() if self.size_average else (1 - ssim_map).sum()

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return self.loss_weight * loss

    @property
    def loss_name(self):
        """ Returns the name of the loss function. """
        return 'loss_ssim'
