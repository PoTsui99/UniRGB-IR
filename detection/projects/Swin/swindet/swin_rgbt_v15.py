"""
Implementation of Swin Transformer for RGBT (RGB-Thermal) data with feature injection.

This code extends the Swin Transformer implementation to handle both RGB and thermal inputs,
using a Spatial Prior Module (SPM) to extract multi-scale CNN features and inject them
into the Swin Transformer via cross-attention, similar to the ViTRGBT approach.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn.modules.utils import _pair
from functools import partial

from mmcv.cnn import build_activation_layer, build_norm_layer
from mmcv.cnn.bricks import DropPath
from mmengine.logging import MMLogger
from mmengine.model import BaseModule
from mmengine.runner.checkpoint import CheckpointLoader
from mmdet.registry import MODELS
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ops.modules import MSDeformAttn

_to_2tuple = nn.modules.utils._ntuple(2)

# @MODELS.register_module()
# class LN2d(nn.Module):
#     """A LayerNorm variant, popularized by Transformers, that performs
#     pointwise mean and variance normalization over the channel dimension for
#     inputs that have shape (batch_size, channels, height, width)."""

#     def __init__(self, normalized_shape, eps=1e-6):
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(normalized_shape))
#         self.bias = nn.Parameter(torch.zeros(normalized_shape))
#         self.eps = eps
#         self.normalized_shape = (normalized_shape, )

#     def forward(self, x):
#         u = x.mean(1, keepdim=True)
#         s = (x - u).pow(2).mean(1, keepdim=True)
#         x = (x - u) / torch.sqrt(s + self.eps)
#         x = self.weight[:, None, None] * x + self.bias[:, None, None]
#         return x


class Mlp(nn.Module):
    """Multilayer perceptron."""

    def __init__(
        self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

    
class WindowAttention(nn.Module):
    """Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value.
            Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):

        super().__init__()
        self.dim = dim
        # TSUIPO: Store window dimensions as tuple (height, width)
        self.window_size = window_size  # Wh, Ww
        # TSUIPO: Number of parallel attention heads
        self.num_heads = num_heads
        # TSUIPO: Calculate dimension per head by dividing input dimension by number of heads
        head_dim = dim // num_heads
        # TSUIPO: Scale factor for attention, default is 1/sqrt(head_dim)
        self.scale = qk_scale or head_dim**-0.5

        # TSUIPO: Create learnable table for relative position bias
        # TSUIPO: Size is (2*window_height-1) * (2*window_width-1) for all possible relative positions
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH

        # TSUIPO: Generate pair-wise relative position index for each token in the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        # TSUIPO: Create mesh grid of coordinates
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        # TSUIPO: Flatten spatial dimensions
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        # TSUIPO: Compute relative coordinates between each pair of positions
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        # TSUIPO: Rearrange dimensions to [Wh*Ww, Wh*Ww, 2] format
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        # TSUIPO: Shift coordinates to start from 0
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        # TSUIPO: Encode 2D relative position into 1D index
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        # TSUIPO: Final relative position index used to index the bias table
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        # TSUIPO: Register as buffer (persistent state but not model parameter)
        self.register_buffer("relative_position_index", relative_position_index)

        # TSUIPO: Linear projection for query, key, value computation (all from same input)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # TSUIPO: Dropout for attention weights
        self.attn_drop = nn.Dropout(attn_drop)
        # TSUIPO: Linear projection to combine multihead attention outputs
        self.proj = nn.Linear(dim, dim)
        # TSUIPO: Dropout for projected output
        self.proj_drop = nn.Dropout(proj_drop)

        # TSUIPO: Initialize bias table with truncated normal distribution
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        # TSUIPO: Softmax operation to normalize attention weights
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # TSUIPO: Get batch size (num_windows*B), sequence length (window_size^2), and channels
        B_, N, C = x.shape
        # TSUIPO: Project input to query, key, value vectors all at once
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        # TSUIPO: Split QKV into separate tensors
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # TSUIPO: Scale query vectors by scaling factor
        q = q * self.scale
        # TSUIPO: Compute attention scores through matrix multiplication (batch matmul)
        attn = q @ k.transpose(-2, -1)

        # TSUIPO: Retrieve relative position bias from the table using position index
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )  # Wh*Ww,Wh*Ww,nH
        # TSUIPO: Rearrange dimensions for proper broadcasting to attention scores
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # nH, Wh*Ww, Wh*Ww
        # TSUIPO: Add relative position bias to attention scores
        attn = attn + relative_position_bias.unsqueeze(0)

        # TSUIPO: Apply attention mask if provided (for shifted windows)
        if mask is not None:
            nW = mask.shape[0]
            # TSUIPO: Reshape and add mask to attention scores (mask is for shifted windows)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            # TSUIPO: Apply softmax to get normalized attention weights
            attn = self.softmax(attn)
        else:
            # TSUIPO: Apply softmax without mask for regular windows
            attn = self.softmax(attn)

        # TSUIPO: Apply dropout to attention weights
        attn = self.attn_drop(attn)

        # TSUIPO: Apply attention weights to value vectors
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        # TSUIPO: Final linear projection
        x = self.proj(x)
        # TSUIPO: Apply dropout to output
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block."""

    def __init__(
        self,
        dim,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=_to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        if drop_path > 0.0:
            from timm.models.layers import DropPath
            self.drop_path = DropPath(drop_path)
        else:
            self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop
        )

        self.H = None
        self.W = None

    def forward(self, x, mask_matrix):
        """Forward function."""
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(
            shifted_x, self.window_size
        )  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(
            -1, self.window_size * self.window_size, C
        )  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    """Patch Merging Layer."""

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """Forward function."""
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


# Components for RGBT feature extraction and injection
class ConvMixFusion(nn.Module):
    def __init__(self, kernels=(3, 3, 5, 7), groups=4, inplanes=64):
        super().__init__()
        self.groups = groups
        self.inplanes = inplanes
        assert len(kernels) == groups
        assert inplanes % groups == 0
        
        self.channel_per_group = inplanes // groups
        convs_rgb = []
        convs_ir = []
        for ks in kernels:
            convs_rgb.append(nn.Conv2d(self.channel_per_group, self.channel_per_group, kernel_size=ks, stride=1, padding=(ks-1)//2, bias=True))
            convs_ir.append(nn.Conv2d(self.channel_per_group, self.channel_per_group, kernel_size=ks, stride=1, padding=(ks-1)//2, bias=True))
        self.convs_rgb = nn.ModuleList(convs_rgb)
        self.convs_ir = nn.ModuleList(convs_ir)
        self.fc = nn.Conv2d(self.channel_per_group, self.channel_per_group, kernel_size=1, stride=1, bias=True)
        
    def forward(self, rgb, ir):
        assert rgb.shape[1] == self.inplanes
        outs = []
        for i in range(self.groups):
            partial_rgb = rgb[:, i * self.channel_per_group: (i+1) * self.channel_per_group, :, :]
            partial_rgb = self.convs_rgb[i](partial_rgb)
            partial_ir = ir[:, i * self.channel_per_group: (i+1) * self.channel_per_group, :, :]
            partial_ir = self.convs_ir[i](partial_ir)
            partial = partial_rgb + partial_ir
            
            alpha = torch.sigmoid(self.fc(partial))
            outs.append(partial_rgb * alpha + partial_ir * (1-alpha))
        
        out = torch.cat(outs, dim=1)
        return out


class SpatialPriorModule(nn.Module):
    def __init__(self, inplanes=64, embed_dim=96, fusion_kernels=(3, 3, 5, 7), fusion_groups=4):
        super().__init__()
        self.embed_dim = embed_dim
        # Stage1: stem, out_dim = inplanes
        self.stem_vis = nn.Sequential(
            nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.stem_ir = nn.Sequential(
            nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.mix_fusion = ConvMixFusion(kernels=fusion_kernels, groups=fusion_groups, inplanes=inplanes)
        
        # cumulative: 8x down
        self.conv2 = nn.Sequential(
            nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(2 * inplanes),
            nn.ReLU(inplace=True)
        )
        
        # cumulative: 16x down
        self.conv3 = nn.Sequential(
            nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(4 * inplanes),
            nn.ReLU(inplace=True)
        )
        
        # cumulative: 32x down
        self.conv4 = nn.Sequential(
            nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(4 * inplanes),
            nn.ReLU(inplace=True)
        )
        
        self.out_8 = nn.Conv2d(2 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.out_16 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.out_32 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        
        # 添加层归一化确保特征维度稳定
        self.norm_8 = nn.LayerNorm(embed_dim)
        self.norm_16 = nn.LayerNorm(embed_dim)
        self.norm_32 = nn.LayerNorm(embed_dim)

    def forward(self, vis, ir):
        bs = vis.shape[0]
        vis = self.stem_vis(vis)  # 4x down
        ir = self.stem_ir(ir)
        
        x_4 = self.mix_fusion(vis, ir)
        x_8 = self.conv2(x_4)
        x_16 = self.conv3(x_8)
        x_32 = self.conv4(x_16)
        
        # 转换为序列格式
        out_8 = self.out_8(x_8).view(bs, self.embed_dim, -1).transpose(1, 2)
        out_16 = self.out_16(x_16).view(bs, self.embed_dim, -1).transpose(1, 2)
        out_32 = self.out_32(x_32).view(bs, self.embed_dim, -1).transpose(1, 2)
        
        # 应用层归一化
        out_8 = self.norm_8(out_8)
        out_16 = self.norm_16(out_16)
        out_32 = self.norm_32(out_32)
        
        return out_8, out_16, out_32  # return sequence


# Auxiliary Functions for Deformable Attention (Used for cross-attention)
def get_reference_points(spatial_shapes, device):
    reference_points_list = []
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
            torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
        ref_y = ref_y.reshape(-1)[None] / H_
        ref_x = ref_x.reshape(-1)[None] / W_
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    reference_points = torch.cat(reference_points_list, 1)
    reference_points = reference_points[:, :, None]
    return reference_points


def deform_inputs(x):
    bs, c, h, w = x.shape

    spatial_shapes = torch.as_tensor([(h // 8, w // 8),
                                     (h // 16, w // 16),
                                     (h // 32, w // 32)],
                                    dtype=torch.long, device=x.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros(
        (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points([(h // 16, w // 16)], x.device)
    inputs = [reference_points, spatial_shapes, level_start_index]
    
    return inputs


class GRU(nn.Module):
    """Simplified GRU for prompt update mechanism."""
    def __init__(self, prompt_dim): 
        super(GRU, self).__init__()
        self.update_gate = nn.Linear(2 * prompt_dim, prompt_dim)

    def forward(self, new_prompts, last_prompts):
        B, seq_len, dim = new_prompts.shape 
        new_prompts = new_prompts.transpose(-2, -1)
        last_prompts = last_prompts.transpose(-2, -1)

        h_tilde = new_prompts.mean(dim=-1)
        h = last_prompts.mean(dim=-1)

        combined = torch.cat((h_tilde, h), dim=1)
        z = torch.sigmoid(self.update_gate(combined))

        h = (1 - z.unsqueeze(1)) * last_prompts.transpose(-2, -1) + z.unsqueeze(1) * new_prompts.transpose(-2, -1)
        
        return h 


class StageBlock(nn.Module):
    """Stage Block for feature injection into Swin Transformer."""
    def __init__(self, dim, n_levels, num_heads=6, n_points=4, deform_ratio=1.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0.,
                first_stage=False):  
        super().__init__()
        self.first_stage = first_stage 
        self.dim = dim  # 记录输入维度
        
        # 特征处理部分
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MSDeformAttn(d_model=dim, n_levels=n_levels, n_heads=num_heads,
                                n_points=n_points, ratio=deform_ratio)
        self.gamma = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
        
        # 维度对齐线性投影
        self.output_proj = nn.Linear(dim, dim)
        
    def forward(self, x, prompt, blocks, gru, feat_pool, 
                reference_points, spatial_shapes, level_start_index, H, W, attn_mask):
        B, L, C = x.shape
        orig_x = x  # 保存原始特征
        
        # 使用cross-attention获取新的prompt
        new_prompt = self.attn(self.query_norm(x), reference_points,
                        self.feat_norm(feat_pool), spatial_shapes,
                        level_start_index, None)
                        
        # 尺度因子和prompt更新
        prompt_tilda = new_prompt * self.gamma
        if not self.first_stage:
            prompt = gru(prompt_tilda, prompt)
        else:
            assert feat_pool is not None, '[error] Feature pool is None!'
            prompt = prompt_tilda
        
        # 使用投影层确保维度匹配
        prompt_proj = self.output_proj(prompt)
        
        # 特征融合，确保维度一致
        x = x + prompt_proj
        
        # 处理transformer blocks
        for blk in blocks:
            blk.H, blk.W = H, W
            x = blk(x, attn_mask)

        return x, prompt


class BasicLayer(nn.Module):
    """Basic Layer for Swin Transformer with RGBT support."""
    
    def __init__(
        self,
        dim,
        depth,
        num_heads,
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
        # For RGBT feature injection
        is_injection_layer=False,
        substage_depths=None,
        n_points=4,
        deform_num_heads=6,
        deform_ratio=1.0,
        deform_ls_init_values=0.0,
    ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.is_injection_layer = is_injection_layer
        self.substage_depths = substage_depths

        # build blocks
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        # For RGB-T feature injection
        if self.is_injection_layer and self.substage_depths:
            n_levels = 3
            self.stages = nn.ModuleList()
            self.grus = nn.ModuleList()
            
            # First substage
            self.stages.append(StageBlock(
                dim, n_levels, deform_num_heads, n_points, deform_ratio,
                norm_layer, deform_ls_init_values, True
            ))
            
            # Remaining substages
            for i in range(1, len(substage_depths)):
                if substage_depths[i] > 0:
                    self.stages.append(StageBlock(
                        dim, n_levels, deform_num_heads, n_points, deform_ratio,
                        norm_layer, deform_ls_init_values, False
                    ))
                    self.grus.append(GRU(dim))

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W, feat_pool=None, deform_inputs_=None):
        """Forward function."""
        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(
            img_mask, self.window_size
        )
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
            attn_mask == 0, float(0.0)
        )

        # For normal Swin layer without feature injection
        if not self.is_injection_layer or not feat_pool:
            for blk in self.blocks:
                blk.H, blk.W = H, W
                if self.use_checkpoint:
                    x = checkpoint.checkpoint(blk, x, attn_mask)
                else:
                    x = blk(x, attn_mask)
        # For Swin layer with feature injection
        else:
            start_idx = 0
            prompt = None
            
            for i, stage in enumerate(self.stages):
                end_idx = start_idx + self.substage_depths[i]
                
                if i == 0:  # First substage
                    x, prompt = stage(
                        x, prompt, self.blocks[start_idx:end_idx], 
                        None, feat_pool, 
                        deform_inputs_[0], deform_inputs_[1], deform_inputs_[2], 
                        H, W, attn_mask
                    )
                else:  # Subsequent substages
                    if self.substage_depths[i] > 0:
                        x, prompt = stage(
                            x, prompt, self.blocks[start_idx:end_idx], 
                            self.grus[i-1], feat_pool,
                            deform_inputs_[0], deform_inputs_[1], deform_inputs_[2], 
                            H, W, attn_mask
                        )
                
                start_idx = end_idx

        # Downsample if needed    
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W


class PatchEmbed(nn.Module):
    """Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = _to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)

        return x


@MODELS.register_module()
class SwinRGBTv15(BaseModule):
    """Swin Transformer for RGBT with feature injection.
    
    This implementation extends the Swin Transformer to handle both RGB and thermal inputs,
    similar to ViTRGBTv15, with feature injection in the third stage.
    
    Args:
        pretrain_img_size (int): Input image size for training the pretrained model,
            used in absolute position embedding. Default: 224.
        patch_size (int): Patch size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Embedding dimension. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True.
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        conv_inplane (int): Number of channels in the SPM. Default: 64.
        n_points (int): Number of points in deformable attention. Default: 4.
        deform_num_heads (int): Number of heads in deformable attention. Default: 6.
        deform_ratio (float): Ratio for deformable attention. Default: 1.0.
        adapter_dim (int): Number of channels in the adapter. Default: None.
        init_cfg (dict, optional): Initialization config dict. Default: None.
    """

    def __init__(
        self,
        pretrain_img_size=224,
        patch_size=4,
        in_chans=3,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        use_checkpoint=False,
        # For RGBT support
        conv_inplane=64,
        n_points=4,
        deform_num_heads=6,
        deform_ls_init_values=0.,
        deform_ratio=1.0,
        adapter_dim=None,
        init_cfg=None
    ):
        super().__init__(init_cfg=init_cfg)

        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        # RGB patch embedding
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )

        # absolute position embedding
        if self.ape:
            pretrain_img_size = _to_2tuple(pretrain_img_size)
            patch_size = _to_2tuple(patch_size)
            patches_resolution = [
                pretrain_img_size[0] // patch_size[0],
                pretrain_img_size[1] // patch_size[1],
            ]

            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1])
            )
            nn.init.trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule
        
        # Feature injection setup for third stage
        third_stage_idx = 2
        if depths[third_stage_idx] <= 6:
            substage_depths = [2, 2, 0, 2]
        else:
            remaining_blocks = depths[third_stage_idx] - 6
            substage_depths = [2, 2, remaining_blocks, 2]

        # 计算各阶段特征维度
        num_features = [int(embed_dim * 2**i) for i in range(self.num_layers)]
        self.num_features = num_features
        
        # 如果未指定adapter_dim，则默认使用第三阶段的特征维度
        if adapter_dim is None:
            adapter_dim = num_features[third_stage_idx]  # 默认使用stage3的维度(768 for Swin-L)
            
        # Add SPM and level embedding for RGBT - 使用指定的adapter_dim
        self.adapter_spm = SpatialPriorModule(inplanes=conv_inplane, embed_dim=adapter_dim)
        self.adapter_level_embed = nn.Parameter(torch.zeros(3, adapter_dim))

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            is_injection_layer = (i_layer == third_stage_idx)
            
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                # RGBT specific parameters
                is_injection_layer=is_injection_layer,
                substage_depths=substage_depths if is_injection_layer else None,
                n_points=n_points,
                deform_num_heads=deform_num_heads,
                deform_ratio=deform_ratio,
                deform_ls_init_values=deform_ls_init_values,
            )
            self.layers.append(layer) 

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f"norm{i_layer}"
            self.add_module(layer_name, layer)

        self._freeze_stages()
        self._out_features = ["p{}".format(i) for i in self.out_indices]
        self._out_feature_channels = {
            "p{}".format(i): self.embed_dim * 2**i for i in self.out_indices
        }
        self._out_feature_strides = {"p{}".format(i): 2 ** (i + 2) for i in self.out_indices}
        self._size_divisibility = 32

        # Initialize weights
        self.adapter_spm.apply(self._init_weights)
        nn.init.normal_(self.adapter_level_embed)
        
        # Freeze non-adapter parameters for efficient fine-tuning
        for k, p in self.named_parameters():
            if 'adapter' not in k:
                p.requires_grad = False

    def _freeze_stages(self):
        """Freeze stages."""
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self):
        """Initialize the weights in backbone."""
        logger = MMLogger.get_current_instance()
        if self.init_cfg is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            self.apply(self._init_weights)
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            ckpt = CheckpointLoader.load_checkpoint(
                self.init_cfg.checkpoint, logger=logger, map_location='cpu')
            if 'model' in ckpt:
                _state_dict = ckpt['model']
            self.load_state_dict(_state_dict, False)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        # For adapter
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
                
    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()
    
    def _add_level_embed(self, c_8, c_16, c_32):
        c_8 = c_8 + self.adapter_level_embed[0]
        c_16 = c_16 + self.adapter_level_embed[1]
        c_32 = c_32 + self.adapter_level_embed[2]
        return c_8, c_16, c_32

    @property
    def size_divisibility(self):
        """Size divisibility."""
        return self._size_divisibility

    def forward(self, vis, ir):
        """Forward function."""
        # Process RGB input
        x = self.patch_embed(vis)

        Wh, Ww = x.size(2), x.size(3)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(
                self.absolute_pos_embed, size=(Wh, Ww), mode="bicubic"
            )
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)
        
        # Extract RGBT features using SPM
        deform_inputs_ = deform_inputs(vis)
        feats = self.adapter_spm(vis, ir)
        feats = self._add_level_embed(*feats)
        feat_pool = torch.cat(feats, dim=1)

        outs = {}
        for i in range(self.num_layers):
            layer = self.layers[i]
            
            # Third layer gets feature injection
            if i == 2:  # The third stage (with feature injection)
                x_orig, H_orig, W_orig, x, Wh, Ww = layer(x, Wh, Ww, feat_pool, deform_inputs_)
                # 记录原始特征的尺寸，确保view操作时使用正确的尺寸
                self.feature_map_sizes[i] = (H_orig, W_orig)
            else:
                x_orig, H_orig, W_orig, x, Wh, Ww = layer(x, Wh, Ww)
                # 记录原始特征的尺寸，确保view操作时使用正确的尺寸
                self.feature_map_sizes[i] = (H_orig, W_orig)

            if i in self.out_indices:
                # TSUIPO: norm before passing features to FPN.
                norm_layer = getattr(self, f"norm{i}") 
                # 确保归一化层的维度与特征维度匹配
                if x_orig.size(-1) != self.num_features[i]:
                    # 创建一个与实际特征维度匹配的临时归一化层
                    temp_norm = nn.LayerNorm(x_orig.size(-1), device=x_orig.device)
                    x_out = temp_norm(x_orig)
                else:
                    x_out = norm_layer(x_orig)
                
                # 获取当前层记录的正确特征图尺寸
                if i in self.feature_map_sizes:
                    H, W = self.feature_map_sizes[i]
                    # 重新计算序列长度以确保正确性
                    B, _, C = x_out.shape
                    expected_L = H * W
                    if x_out.size(1) != expected_L:
                        # 如果序列长度与HxW不匹配，使用实际的序列长度计算新的H,W
                        actual_L = x_out.size(1)
                        sqr = int(np.sqrt(actual_L))
                        if sqr * sqr == actual_L:
                            # 如果是完美平方数
                            H, W = sqr, sqr
                        else:
                            # 找最接近的因子
                            for h in range(sqr, 0, -1):
                                if actual_L % h == 0:
                                    H, W = h, actual_L // h
                                    break
                else:
                    # 使用原始记录的H,W
                    H, W = H_orig, W_orig
                
                # 使用正确计算的H,W尺寸进行重塑
                out = x_out.view(x_out.size(0), H, W, x_out.size(-1)).permute(0, 3, 1, 2).contiguous()
                outs["p{}".format(i)] = out

        return outs
        
    def train(self, mode=True):
        """Convert the model into training mode while keeping normalization layers frozen."""
        super().train(mode)
        self._freeze_stages() 