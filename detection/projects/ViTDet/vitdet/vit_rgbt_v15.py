# Main changes: After 4x downsampling in SPM, use grouped convolutions with different kernel sizes (followed by 1x1 convolution).
# TODO: Activation and regularization are subject to discussion.
#       Use ECANet structure for channel attention (with shortcut) to avoid dimension reduction in SE
#       prompt = z * new_prompt + (1 - z) * last_prompt
#       The new prompt is the result of cross-attention between ViT feats and pool.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from collections import OrderedDict
from functools import partial
# torch.autograd.set_detect_anomaly(True)  # to close

from mmcv.cnn import build_activation_layer, build_norm_layer
from mmcv.cnn.bricks import DropPath
from mmengine.logging import MMLogger
from mmengine.model import BaseModule
from mmengine.runner.checkpoint import CheckpointLoader
from mmdet.registry import MODELS
from typing import Optional
# Deformable Attn
from ops.modules import MSDeformAttn


# Auxiliary Functions For ViT
def get_abs_pos(abs_pos, has_cls_token, hw):
    h, w = hw
    if has_cls_token:
        abs_pos = abs_pos[:, 1:]
    xy_num = abs_pos.shape[1]
    size = int(math.sqrt(xy_num))
    assert size * size == xy_num

    if size != h or size != w:
        new_abs_pos = F.interpolate(
            abs_pos.reshape(1, size, size, -1).permute(0, 3, 1, 2),
            size=(h, w),
            mode='bicubic',
            align_corners=False,
        )

        return new_abs_pos.permute(0, 2, 3, 1)
    else:
        return abs_pos.reshape(1, h, w, -1)


def get_rel_pos(q_size, k_size, rel_pos):
    """
    Get relative positional embeddings according to the relative positions
    of query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode='linear',
        )
        rel_pos_resized = rel_pos_resized.reshape(-1,
                                                  max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords -
                       k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(attn, q, rel_pos_h, rel_pos_w, q_size, k_size):
    """
    Args:
        attn (Tensor): attention map.
        q (Tensor):
            query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor):
            relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor):
            relative position embeddings (Lw, C) for width axis.
        q_size (Tuple):
            spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple):
            spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum('bhwc,hkc->bhwk', r_q, Rh)
    rel_w = torch.einsum('bhwc,wkc->bhwk', r_q, Rw)

    attn = (attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] +
            rel_w[:, :, :, None, :]).view(B, q_h * q_w, k_h * k_w)

    return attn


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size,
               window_size, C)
    windows = x.permute(0, 1, 3, 2, 4,
                        5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)  # shape of windows: (num_windows, window_size, window_size, dim)


def window_unpartition(windows, window_size, pad_hw, hw):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size,
                     window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
    
    
class Attention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=True,
                 use_rel_pos=False,
                 rel_pos_zero_init=True,
                 input_size=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(
                torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(
                torch.zeros(2 * input_size[1] - 1, head_dim))

            if not rel_pos_zero_init:
                nn.init.trunc_normal_(self.rel_pos_h, std=0.02)
                nn.init.trunc_normal_(self.rel_pos_w, std=0.02)

    def forward(self, x):
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads,
                                  -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h,
                                          self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)  # grad explosion may happen here...
        
        x = (attn @ v).view(B, self.num_heads, H, W,
                            -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks."""

    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_cfg=dict(type='GELU'),
            bias=True,
            drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = build_activation_layer(act_cfg)
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Block(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path=0.0,
        init_values=None,
        norm_cfg=dict(type='LN', eps=1e-6),
        act_cfg=dict(type='GELU'),
        use_rel_pos=False,
        rel_pos_zero_init=True,
        window_size=0,
        input_size=None,
    ):
        super().__init__()
        self.norm1 = build_norm_layer(norm_cfg, dim)[1]
        # NOTE: LayerScale is default not used, unless init_values given not None...
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else
            (window_size, window_size),
        )

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.norm2 = build_norm_layer(norm_cfg, dim)[1]
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_cfg=act_cfg)

        self.window_size = window_size

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        # NOTE: layerscale before drop path
        x = shortcut + self.drop_path(self.ls1(x))  # attn
        x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))  # mlp

        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(self,
                 kernel_size=(16, 16),
                 stride=(16, 16),
                 padding=(0, 0),
                 in_chans=3,
                 embed_dim=768):
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int):  embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding)

    def forward(self, x):
        x = self.proj(x)
        # B dim H W -> B H W dim
        x = x.permute(0, 2, 3, 1)
        return x


# Auxiliary Functions for Deformable Attention (Used for cross-attention)
def get_reference_points(spatial_shapes, device):
    reference_points_list = []
    for lvl, (H_, W_) in enumerate(spatial_shapes):  # for each scale level
        ref_y, ref_x = torch.meshgrid(  # generate H×W reference points...
            torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device), 
            torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
        ref_y = ref_y.reshape(-1)[None] / H_  # add one dimension, i.e. ref_y.reshape(-1)[None, :] / H_ and ref_y.reshape(-1).unsqueeze(0) / H_
        ref_x = ref_x.reshape(-1)[None] / W_
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    reference_points = torch.cat(reference_points_list, 1)
    reference_points = reference_points[:, :, None]
    return reference_points


def deform_inputs(x):  # multi-scale only!
    bs, c, h, w = x.shape

    spatial_shapes = torch.as_tensor([(h // 8, w // 8),
                                      (h // 16, w // 16),
                                      (h // 32, w // 32)],
                                     dtype=torch.long, device=x.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros(
        (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points([(h // 16, w // 16)], x.device)
    inputs = [reference_points, spatial_shapes, level_start_index]
    
    # spatial_shapes = torch.as_tensor([(h // 16, w // 16)], dtype=torch.long, device=x.device)
    # level_start_index = torch.cat((spatial_shapes.new_zeros(
    #     (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    # reference_points = get_reference_points([(h // 16, w // 16)], x.device)
    # inputs = [reference_points, spatial_shapes, level_start_index]
    
    return inputs


class ConvMixFusion(nn.Module):
    def __init__(self, kernels=(3, 3, 5, 7), groups=4, inplanes=64):
        super().__init__()
        convs = []
        self.groups = groups
        self.inplanes = inplanes
        assert len(kernels) == groups
        assert inplanes % groups == 0
        
        self.channel_per_group = inplanes // groups  # int
        convs_rgb = []
        convs_ir = []
        for ks in kernels:
            convs_rgb.append(nn.Conv2d(self.channel_per_group, self.channel_per_group, kernel_size=ks, stride=1, padding=(ks-1)//2, bias=True))  # same padding
            convs_ir.append(nn.Conv2d(self.channel_per_group, self.channel_per_group, kernel_size=ks, stride=1, padding=(ks-1)//2, bias=True))  # same padding
        self.convs_rgb = nn.Sequential(*convs_rgb)
        self.convs_ir = nn.Sequential(*convs_ir)
        self.gap = nn.AdaptiveAvgPool2d(1)  # global average pooling
 
        self.fc = nn.Conv2d(self.channel_per_group, self.channel_per_group, kernel_size=1, stride=1, bias=True)  # NOTE: shared fc
        
    def forward(self, rgb, ir):
        assert rgb.shape[1] == self.inplanes
        outs = []
        for i in range(self.groups):  # fuse for each group of rgb and ir
            partial_rgb = rgb[:, i * self.channel_per_group: (i+1) * self.channel_per_group, :, :]
            partial_rgb = self.convs_rgb[i](partial_rgb)
            partial_ir = ir[:, i * self.channel_per_group: (i+1) * self.channel_per_group, :, :]
            partial_ir = self.convs_ir[i](partial_ir)
            partial = partial_rgb + partial_ir  # TODO: try channel-cat
            
            alpha = torch.sigmoid(self.fc(partial))
          
            outs.append(partial_rgb * alpha + partial_ir * (1-alpha))
        
        out = outs[0]
        for i in range(1, self.groups):
            out = torch.cat((out, outs[i]), dim=1)  # channel concat
        return out
    
    
class SpatialPriorModule(nn.Module):
    def __init__(self, inplanes=64, embed_dim=768, fusion_kernels=(3, 3, 5, 7), fusion_groups=4):
        super().__init__()
        self.embed_dim = embed_dim
        # Stage1: stem, out_dim = inplanes
        self.stem_vis = nn.Sequential(*[
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
        ])
        self.stem_ir = nn.Sequential(*[
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
        ])
        
        self.mix_fusion = ConvMixFusion(kernels=fusion_kernels, groups=fusion_groups, inplanes=inplanes)
        
        # culmulative: 8x down
        self.conv2 = nn.Sequential(*[
            nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(2 * inplanes),
            nn.ReLU(inplace=True)
        ])
        
        # culmulative: 16x down
        self.conv3 = nn.Sequential(*[
            nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(4 * inplanes),
            nn.ReLU(inplace=True)
        ])
        
        # culmulative: 32x down
        self.conv4 = nn.Sequential(*[
            nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(4 * inplanes),
            nn.ReLU(inplace=True)
        ])
        
        self.out_8 = nn.Conv2d(2 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)  # 8x down
        self.out_16 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)  # 16x down
        self.out_32 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)  # 32x down

    def forward(self, vis, ir):
        bs, _, _, _ = vis.shape
        vis = self.stem_vis(vis)  # 4x down
        ir = self.stem_ir(ir)
        
        x_4 = self.mix_fusion(vis, ir)
        x_8 = self.conv2(x_4)
        x_16 = self.conv3(x_8)
        x_32 = self.conv4(x_16)
        out_8 = self.out_8(x_8).view(bs, self.embed_dim, -1).transpose(1, 2)
        out_16 = self.out_16(x_16).view(bs, self.embed_dim, -1).transpose(1, 2)
        out_32 = self.out_32(x_32).view(bs, self.embed_dim, -1).transpose(1, 2)
        
        return out_8, out_16, out_32  # return sequence

    
class GRU(nn.Module):  # simplified version of GRU, the h_tilde is already generated, so only implement the 'Update' mechanism
    """
        new prompt * z + last prompt * (1-z)
        can be considered as kind of shortcut connection or long-range memo. 
        also can be considered as spatial attention (shared by patches)
    """
    def __init__(self, prompt_dim): 
        super(GRU, self).__init__()

        self.update_gate = nn.Linear(2 * prompt_dim, prompt_dim)  # generate dim-wise mask

    def forward(self, new_prompts, last_prompts):  # shape of input: (B, seq_len, dim)
        B, seq_len, dim = new_prompts.shape 
        new_prompts = new_prompts.transpose(-2, -1)
        last_prompts = last_prompts.transpose(-2, -1)

        # -> (B, dim)
        h_tilde = new_prompts.mean(dim=-1)  # NOTE: considered to be the presentation of candidate hidden state
        h = last_prompts.mean(dim=-1)

        combined = torch.cat((h_tilde, h), dim=1)
        z = torch.sigmoid(self.update_gate(combined))  # update ratio, to what extent preserve the new prompt

        # weighted sum of new prompts and last prompts
        h = (1 - z.unsqueeze(1)) * last_prompts.transpose(-2, -1) + z.unsqueeze(1) * new_prompts.transpose(-2, -1)
        
        return h  
        

class StageBlock(nn.Module):  # Includes Transformer Blocks' Forward
    def __init__(self, dim, n_levels, num_heads=6, n_points=4, deform_ratio=1.0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0.,  # Layer Scaler
                 first_stage=False):  
        super().__init__()
        self.first_stage = first_stage 
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MSDeformAttn(d_model=dim, n_levels=n_levels, n_heads=num_heads,
                                 n_points=n_points, ratio=deform_ratio)
        # Layer Scale, for better optimization?
        # if not first_stage:
        #     self.gamma = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
        # prompt_tilda = prompt(after cross-attn) * gamma
        self.gamma = nn.Parameter(init_values * torch.ones(dim), requires_grad=True) 
        
    def forward(self, x, prompt, blocks, gru, feat_pool, 
                reference_points, spatial_shapes, level_start_index, H, W):
        # Input: sequence
        B, seq_len, dim = x.shape
        # new prompt: through cross-attn
        new_prompt = self.attn(self.query_norm(x), reference_points,
                         self.feat_norm(feat_pool), spatial_shapes,
                         level_start_index, None)

        prompt_tilda = new_prompt * self.gamma
        if not self.first_stage:
            prompt = gru(prompt_tilda, prompt)
            # prompt = prompt * (1 - self.gamma) + new_prompt * self.gamma
        else:
            assert feat_pool is not None, '[error] Feature pool is None!'
            prompt = prompt_tilda
            
        x = x + prompt 
        for blk in blocks:
            x = blk(x.reshape(B, H, W, dim)).reshape(B, seq_len, dim)

        return x, prompt    


@MODELS.register_module()
class ViTRGBTv15(BaseModule):
    
    def __init__(self,
                 method=None,
                 img_size=1024,
                 stage_ranges=[[0, 2], [3, 5], [6, 8], [9, 11]],
                 conv_inplane=64,  # For SPM
                 n_points=4,
                 deform_num_heads=6,
                 deform_ls_init_values=0.,  # deform attention's ls init_val
                 deform_ratio=1.0,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 qkv_bias=True,
                 drop_path_rate=0.0,
                 deform_norm_layer=partial(nn.LayerNorm, eps=1e-6),  # deform attn's token norm
                 norm_cfg=dict(type='LN', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 use_abs_pos=True,
                 use_rel_pos=True,
                 rel_pos_zero_init=True,
                 window_size=0,
                 window_block_indexes=(0, 1, 3, 4, 6, 7, 9, 10),
                 pretrain_img_size=224,
                 pretrain_use_cls_token=True,
                 init_cfg=None
                ):
        
        super().__init__()
        self.stage_ranges = stage_ranges  # stage splitting
        self.pretrain_use_cls_token = pretrain_use_cls_token
        self.init_cfg = init_cfg
        self.pretrain_size = _pair(pretrain_img_size)
        self.adapter_level_embed = nn.Parameter(torch.zeros(3, embed_dim))  # NOTE: level embed
        deform_norm_layer = deform_norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.deform_norm_layer = deform_norm_layer
        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim)
        
        if use_abs_pos:
            num_patches = (pretrain_img_size // patch_size) * (
                pretrain_img_size // patch_size)
            num_positions = (num_patches +
                             1) if pretrain_use_cls_token else num_patches
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_positions, embed_dim))
        else:
            self.pos_embed = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i in window_block_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size))
            for i in range(depth)
        ])
        
        self.adapter_spm = SpatialPriorModule(inplanes=conv_inplane, embed_dim=embed_dim)
        # 4 stages in total
        n_levels = 3
        self.adapter_stages = [StageBlock(embed_dim, n_levels, deform_num_heads, n_points, deform_ratio,
                       deform_norm_layer, deform_ls_init_values, True)]
        for i in range(3): 
            self.adapter_stages.append(StageBlock(embed_dim, n_levels, deform_num_heads, n_points, deform_ratio,
                                                  deform_norm_layer, deform_ls_init_values, False))
        self.adapter_stages = nn.Sequential(*self.adapter_stages)
        self.adapter_grus = nn.Sequential(*[GRU(embed_dim) for i in range(len(stage_ranges) - 1)])
        # self.grus = nn.Sequential(*[GRU(embed_dim) for i in range(len(stage_ranges))])
        # Init weights
        self.adapter_spm.apply(self._init_weights)
        self.adapter_stages.apply(self._init_weights)
        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_deform_weights)  # Init deform attn weights 
        nn.init.normal_(self.adapter_level_embed)  # init layer embedding, for multi-scale feat
            
        for k, p in self.named_parameters():
            if 'prompt' not in k and 'adapter' not in k:
                p.requires_grad = False
            # if 'gamma' in k:  # layer scaler unfreeze
            #     p.requires_grad = True  
            if use_rel_pos:  # NOTE: add relative positon bias
                if 'rel_pos' in k:  # rel_pos_h, rel_pos_w
                    p.requires_grad = True   
                    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        # For Adapter
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self):
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
            
    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()
            
    def _add_level_embed(self, c_8, c_16, c_32):
        c_8 = c_8 + self.adapter_level_embed[0]
        c_16 = c_16 + self.adapter_level_embed[1]
        c_32 = c_32 + self.adapter_level_embed[2]
        return c_8, c_16, c_32
    
    def forward(self, vis, ir):
        deform_inputs_ = deform_inputs(vis)
        
        # get multi-scale CNN feature
        feats = self.adapter_spm(vis, ir)  
        feats = self._add_level_embed(*feats)
        feat_pool = torch.cat(feats, dim=1)
        
        x = self.patch_embed(vis)  # B, C, H, W -> B, H, W, dim
        B, H, W, dim = x.shape
        # abs pos embed
        if self.pos_embed is not None:
            x = x + get_abs_pos(self.pos_embed, self.pretrain_use_cls_token,
                                (x.shape[1], x.shape[2]))    
        x = x.reshape(B, H*W, dim)  # NOTE: becomes sequence
        prompt = None 
        for i, stage in enumerate(self.adapter_stages):
            blk_indexes = self.stage_ranges[i]
            if i == 0: # w/o gru
                x, prompt = stage(x, prompt, self.blocks[blk_indexes[0]: blk_indexes[-1]+1], None, feat_pool, deform_inputs_[0], deform_inputs_[1], deform_inputs_[2], H, W)
            else:
                x, prompt = stage(x, prompt, self.blocks[blk_indexes[0]: blk_indexes[-1]+1], self.adapter_grus[i - 1], feat_pool, deform_inputs_[0], deform_inputs_[1], deform_inputs_[2], H, W)
        # B, seq_len, dim -> B, dim, seq_len -> B, dim, H, W
        x = x.transpose(1, 2).view(B, dim, H, W).contiguous()
        return x
    