"""
Feature Pyramid Network (FPN) for Swin Transformer

This implementation is adapted from detectron2's FPN, with additions:
1. LN2d normalization for multi-scale features before fusion
2. Compatibility with Swin Transformer backbone for RGB-T detection
"""

import math
import torch
import torch.nn.functional as F
from torch import nn

from mmcv.cnn import Conv2d, build_norm_layer
from mmengine.model import BaseModule
from mmdet.registry import MODELS

from mmdet.utils import OptConfigType, ConfigType, MultiConfig
from typing import List, Dict, Tuple


@MODELS.register_module()
class LN2d(nn.Module):
    """A LayerNorm variant, popularized by Transformers, that performs
    pointwise mean and variance normalization over the channel dimension for
    inputs that have shape (batch_size, channels, height, width)."""

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


@MODELS.register_module()
class SwinFPN(BaseModule):
    """
    Feature Pyramid Network specifically designed for Swin Transformer.
    Includes Linear Normalization (LN2d) for normalizing features across scales.
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        num_outs: int = 5,
        start_level: int = 0,
        end_level: int = -1,
        norm_cfg: ConfigType = dict(type='BN', requires_grad=True),
        add_ln_norm: bool = True,
        use_residual: bool = True,
        top_block=None,
        fuse_type: str = "sum",
        square_pad: int = 0,
        init_cfg: MultiConfig = None,
    ):
        """
        Args:
            in_channels (List[int]): List of input channels for each input feature map.
            out_channels (int): Number of channels in the output feature maps.
            num_outs (int): Number of output scales.
            start_level (int): Start level of input features to use.
            end_level (int): End level of input features to use. -1 means the last level.
            norm_cfg (dict): Dictionary to construct normalization layers.
            add_ln_norm (bool): Whether to add LayerNorm (LN2d) for feature normalization.
            use_residual (bool): Whether to use residual connections in lateral connections.
            top_block: Optional module used for additional operations on the top level features.
            fuse_type (str): Type of feature fusion, 'sum' or 'avg'.
            square_pad (int): If > 0, pad inputs to a square size.
            init_cfg (dict): Initialization config dict.
        """
        super(SwinFPN, self).__init__(init_cfg=init_cfg)
        
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.add_ln_norm = add_ln_norm
        self.use_residual = use_residual
        self.fuse_type = fuse_type
        assert self.fuse_type in ['sum', 'avg']
        
        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level is not -1, the num_outs should be end_level - start_level
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        
        self.start_level = start_level
        self.end_level = end_level
        
        # Build lateral connections
        self.lateral_convs = nn.ModuleList()
        # Build layer norms for each scale
        self.layer_norms = nn.ModuleList() if self.add_ln_norm else None
        # Build output convs
        self.fpn_convs = nn.ModuleList()
        
        # Build the lateral connections and LN layers
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = Conv2d(
                in_channels[i], out_channels, kernel_size=1, bias=(norm_cfg is None)
            )
            
            fpn_conv = Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=(norm_cfg is None)
            )
            
            # Initialize the weights for convolutions
            nn.init.kaiming_normal_(l_conv.weight, mode='fan_out')
            nn.init.kaiming_normal_(fpn_conv.weight, mode='fan_out')
            
            if norm_cfg is None:
                if l_conv.bias is not None:
                    nn.init.constant_(l_conv.bias, 0)
                if fpn_conv.bias is not None:
                    nn.init.constant_(fpn_conv.bias, 0)
            
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
            
            # Add LN2d for feature normalization
            if self.add_ln_norm:
                self.layer_norms.append(LN2d(in_channels[i]))
        
        # Build top block (for additional levels like P6, P7)
        if isinstance(top_block, dict):
           self.top_block = MODELS.build(top_block)
        else:
           self.top_block = top_block
           
        self._square_pad = square_pad
        
        # Calculate strides for output features
        input_strides = [2 ** (i + 2) for i in range(len(in_channels))]
        strides = input_strides[self.start_level:self.backbone_end_level]
        self._out_feature_strides = {}
        
        # Generate feature names like p2, p3, etc.
        for idx, stride in enumerate(strides):
            self._out_feature_strides[f"p{idx + 2}"] = stride
        
        # Handle top block output features
        if self.top_block is not None:
            last_stage_idx = len(strides) + 1
            for i in range(self.top_block.num_levels):
                self._out_feature_strides[f"p{last_stage_idx + i}"] = strides[-1] * 2 ** (i + 1)
        
        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        self._size_divisibility = strides[-1]
    
    @property
    def size_divisibility(self):
        return self._size_divisibility
    
    @property
    def padding_constraints(self):
        return {"square_size": self._square_pad}
    
    def forward(self, inputs):
        """
        Args:
            inputs (dict[str->Tensor]): mapping feature name to feature tensor
                
        Returns:
            dict[str->Tensor]: mapping from feature name to FPN feature
        """
        # Build laterals
        laterals = []
        
        # Apply LN2d to normalize input features if required
        if self.add_ln_norm:
            norm_inputs = []
            for idx, (feature_name, x) in enumerate(inputs.items()):
                if idx >= self.start_level and idx < self.backbone_end_level:
                    norm_inputs.append(self.layer_norms[idx - self.start_level](x))
                else:
                    norm_inputs.append(x)
            
            # Generate laterals from normalized inputs
            for idx, (feature_name, x) in enumerate(zip(inputs.keys(), norm_inputs)):
                if idx >= self.start_level and idx < self.backbone_end_level:
                    laterals.append(self.lateral_convs[idx - self.start_level](x))
        else:
            # Generate laterals directly from inputs
            for idx, (feature_name, x) in enumerate(inputs.items()):
                if idx >= self.start_level and idx < self.backbone_end_level:
                    laterals.append(self.lateral_convs[idx - self.start_level](x))
        
        # Build top-down path
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode="nearest"
            )
            # Use average fusion if specified
            if self.fuse_type == "avg":
                laterals[i - 1] /= 2.0
        
        # Build outputs
        results = []
        for i, lateral in enumerate(laterals):
            results.append(self.fpn_convs[i](lateral))
        
        # Apply top block if provided (for additional levels like P6, P7)
        if self.top_block is not None:
            if self.top_block.in_feature in inputs:
                top_block_in_feature = inputs[self.top_block.in_feature]
            else:
                feature_names = list(inputs.keys())
                top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
            
            results.extend(self.top_block(top_block_in_feature))
        
        # Add extra levels if num_outs > len(results)
        assert len(self._out_features) <= len(results)
        return {f: res for f, res in zip(self._out_features, results[:len(self._out_features)])}


@MODELS.register_module()
class LastLevelMaxPool(nn.Module):
    """
    This module is used in the original FPN to generate a downsampled
    P6 feature from P5.
    """

    def __init__(self):
        super().__init__()
        self.num_levels = 1
        self.in_feature = "p5"

    def forward(self, x):
        return [F.max_pool2d(x, kernel_size=1, stride=2, padding=0)]


@MODELS.register_module()
class LastLevelP6P7(nn.Module):
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7 from
    C5 feature.
    """

    def __init__(self, in_channels, out_channels, in_feature="p5"):
        super().__init__()
        self.num_levels = 2
        self.in_feature = in_feature
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        # Initialize weights
        nn.init.kaiming_normal_(self.p6.weight, mode='fan_out')
        nn.init.kaiming_normal_(self.p7.weight, mode='fan_out')
        nn.init.constant_(self.p6.bias, 0)
        nn.init.constant_(self.p7.bias, 0)

    def forward(self, c5):
        p6 = self.p6(c5)
        p7 = self.p7(F.relu(p6))
        return [p6, p7] 