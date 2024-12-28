# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import, division, print_function

import MultiScaleDeformableAttention as MSDA  # CUDA operations already built?
import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.cuda.amp import custom_bwd, custom_fwd


class MSDeformAttnFunction(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    # ctx stores some context information?
    def forward(ctx, value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights, im2col_step):
        ctx.im2col_step = im2col_step
        output = MSDA.ms_deform_attn_forward(value, value_spatial_shapes,
                                             value_level_start_index,
                                             sampling_locations,
                                             attention_weights,
                                             ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes,
                              value_level_start_index, sampling_locations,
                              attention_weights)
        return output

    @staticmethod
    @once_differentiable
    @custom_bwd
    def backward(ctx, grad_output):
        value, value_spatial_shapes, value_level_start_index, \
        sampling_locations, attention_weights = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = \
            MSDA.ms_deform_attn_backward(
                value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights, grad_output, ctx.im2col_step)

        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


def ms_deform_attn_core_pytorch(value, value_spatial_shapes,
                                sampling_locations, attention_weights):
    # for debug and test only,
    # need to use cuda version instead
    # B, seq_len, n_heads, head_dim
    N_, S_, M_, D_ = value.shape  
    # B, query_len, n_heads, n_levels, num_points, 2
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape  
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)  # 将 sequence 按照 level 进行切分, 生成若干个原 tensor 的 view
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_ * M_, D_, H_, W_)
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_, mode='bilinear',
                                          padding_mode='zeros', align_corners=False)  # 对每个 level 的特征图进行插值, 取得对应"位置"上的 feature(坐标是小数)
        sampling_value_list.append(sampling_value_l_)  # 这里只是取了一些 token 的值, 用到了采样
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.transpose(1, 2).reshape(N_ * M_, 1, Lq_, L_ * P_)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) *
              attention_weights).sum(-1).view(N_, M_ * D_, Lq_)
    return output.transpose(1, 2).contiguous()

# by tsuipo, for complexity computing
# import torch
# import torch.nn.functional as F
# from torch import nn

# class MyMSDeformAttnFunction(nn.Module):
#     def __init__(self, d_model=256, n_levels=3, n_heads=8, n_points=4, ratio=1.0):
#         super(MyMSDeformAttnFunction, self).__init__()
#         if d_model % n_heads != 0:
#             raise ValueError('d_model must be divisible by n_heads, '
#                              'but got {} and {}'.format(d_model, n_heads))

#         self.im2col_step = 64

#         self.d_model = d_model
#         self.n_levels = n_levels
#         self.n_heads = n_heads
#         self.n_points = n_points
#         self.ratio = ratio
#         self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
#         self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
#         self.value_proj = nn.Linear(d_model, int(d_model * ratio))
#         self.output_proj = nn.Linear(int(d_model * ratio), d_model)

#         self._reset_parameters()

#     def _reset_parameters(self):
#         nn.init.constant_(self.sampling_offsets.weight.data, 0.)
#         thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * torch.pi / self.n_heads)
#         grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
#         grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
#         for i in range(self.n_points):
#             grid_init[:, :, i, :] *= i + 1

#         with torch.no_grad():
#             self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
#         nn.init.constant_(self.attention_weights.weight.data, 0.)
#         nn.init.constant_(self.attention_weights.bias.data, 0.)
#         nn.init.xavier_uniform_(self.value_proj.weight.data)
#         nn.init.constant_(self.value_proj.bias.data, 0.)
#         nn.init.xavier_uniform_(self.output_proj.weight.data)
#         nn.init.constant_(self.output_proj.bias.data, 0.)

#     def forward(self, query, reference_points, input_flatten, input_spatial_shapes,
#                 input_level_start_index, input_padding_mask=None):
#         N, Len_q, _ = query.shape
#         N, Len_in, _ = input_flatten.shape
#         assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

#         value = self.value_proj(input_flatten)
#         if input_padding_mask is not None:
#             value = value.masked_fill(input_padding_mask[..., None], float(0))

#         value = value.view(N, Len_in, self.n_heads, int(self.ratio * self.d_model) // self.n_heads)
#         sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
#         attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
#         attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)

#         if reference_points.shape[-1] == 2:
#             offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
#             sampling_locations = reference_points[:, :, None, :, None, :] + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
#         elif reference_points.shape[-1] == 4:
#             sampling_locations = reference_points[:, :, None, :, None, :2] + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
#         else:
#             raise ValueError('Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))

#         output = self.ms_deform_attn_core_pytorch(value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights)
#         output = self.output_proj(output)
#         return output

#     def ms_deform_attn_core_pytorch(self, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights):
#         N_, S_, M_, D_ = value.shape
#         _, Lq_, M_, L_, P_, _ = sampling_locations.shape

#         sampling_grids = 2 * sampling_locations - 1
#         output = value.new_zeros((N_, Lq_, M_, D_ * self.ratio))
        
#         for lvl in range(self.n_levels):
#             start_idx = value_level_start_index[lvl]
#             end_idx = start_idx + (value_spatial_shapes[lvl, 0] * value_spatial_shapes[lvl, 1])
#             value_l = value[:, start_idx:end_idx, :, :].reshape(N_, value_spatial_shapes[lvl, 0], value_spatial_shapes[lvl, 1], M_, D_)
#             value_l = value_l.permute(0, 3, 4, 1, 2).contiguous().view(N_ * M_, D_, value_spatial_shapes[lvl, 0], value_spatial_shapes[lvl, 1])
#             sampling_grid_l = sampling_grids[:, :, :, lvl, :, :].view(N_ * M_, Lq_, P_, 2)
#             sampling_value_l = F.grid_sample(value_l, sampling_grid_l, mode='bilinear', padding_mode='zeros', align_corners=False)
#             sampling_value_l = sampling_value_l.view(N_, M_, D_, Lq_, P_).permute(0, 3, 1, 4, 2)
#             attention_weights_l = attention_weights[:, :, :, lvl, :].view(N_, Lq_, M_, P_)
#             output += (sampling_value_l * attention_weights_l.unsqueeze(-1)).sum(-2)

#         output = output.view(N_, Lq_, self.n_heads * int(self.ratio * self.d_model) // self.n_heads)
#         return output
    