import torch
import pdb
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

class SpatialAttnDyConv(nn.Module):
    """
        Using m Learnable 'bases' conv (out_chans x in_chans x kernel_size*kernel_size)
        to compose a dynamic conv
    """
    def __init__(self, in_chans=9, out_chans=9, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True, 
                 num_bases=6, aux_chans=9, hidden_dim=16, ):
        super().__init__()
        # defensive coding 
        assert in_chans % groups == 0 and out_chans % groups == 0, 'group should be a divisor of in_chans and out_chans!'
        
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.num_bases = num_bases
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        
        # Output shape of the first stage: B, num_bases*out_chans, H, W
        # Hence the out_chans of the Conv should be num_bases*out_chans
        self.bases_weight = nn.Parameter(torch.Tensor(num_bases * out_chans, in_chans // groups,
                                                   *self.kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_bases * out_chans))
        else:
            self.register_parameter('bias', None)
            
        # To get the 'coefficient tensor'
        self.get_per_pixel_bases_combination_weight = nn.Sequential(
            nn.Conv2d(aux_chans, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, num_bases, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
        
        self.reset_parameters()  # initialize the weights and bias of conv bases
        
    def reset_parameters(self):
        n = self.in_chans
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.bases_weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
            
    def forward(self, x, aux):  # aux is used to generate per-pixel weights of bases
        B, C, H, W = x.shape
        # shape: B, num_bases, H, W
        per_pixel_bases_combination_weight = self.get_per_pixel_bases_combination_weight(aux)  # For the origianl paper, aux here is x
                
        # shape: B, out_chans*num_bases, H, W
        attn_map = F.conv2d(aux, self.bases_weight, self.bias, self.stride, 
                     self.padding, self.dilation, self.groups).view(B, self.num_bases, self.out_chans, H, W)
        
        attn_map = torch.einsum("bnohw,bnhw->bohw", attn_map, per_pixel_bases_combination_weight)
        
        # pdb.set_trace()
        
        # element-wise product
        x = x * attn_map
        
        return x
    

class DualStreamSelectiveReconstror2(nn.Module):
    # input: B, 3, H, W
    def __init__(self):
        super().__init__()
        self.hFreqConv_vis = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)  
        self.hFreqConv_ir = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)  
        self.mFreqConv_vis = nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1)  # 2x down sample
        self.mFreqConv_ir = nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1)  # 2x down sample
        self.lFreqConv_vis = nn.Conv2d(3, 3, kernel_size=3, stride=4, padding=1)  # 4x down sample, get low freq
        self.lFreqConv_ir = nn.Conv2d(3, 3, kernel_size=3, stride=4, padding=1)  # 4x down sample, get low freq
        # params: num of kernels, scale_factor, in_planes, out_planes, ...
        self.dyConv = SpatialAttnDyConv()

    def forward(self, vis, ir):
        B, C, H, W = vis.shape
        lFreq_vis = self.lFreqConv_vis(vis)
        lFreq_ir = self.lFreqConv_ir(ir)
        mFreq_vis = self.mFreqConv_vis(vis)
        mFreq_ir = self.mFreqConv_ir(ir)
        hFreq_vis = self.hFreqConv_vis(vis) - F.interpolate(mFreq_vis, scale_factor=2, mode='bilinear', align_corners=False)
        hFreq_ir = self.hFreqConv_ir(ir) - F.interpolate(mFreq_ir, scale_factor=2, mode='bilinear', align_corners=False)
        mFreq_vis = mFreq_vis - F.interpolate(lFreq_vis, scale_factor=2, mode='bilinear', align_corners=False)
        mFreq_ir = mFreq_ir - F.interpolate(lFreq_ir, scale_factor=2, mode='bilinear', align_corners=False)

        dummy_mFreq_vis = F.interpolate(mFreq_vis, scale_factor=2, mode='bilinear', align_corners=False)
        dummy_mFreq_ir = F.interpolate(mFreq_ir, scale_factor=2, mode='bilinear', align_corners=False)
        dummy_lFreq_vis = F.interpolate(lFreq_vis, scale_factor=4, mode='bilinear', align_corners=False)
        dummy_lFreq_ir = F.interpolate(lFreq_ir, scale_factor=4, mode='bilinear', align_corners=False)
        # TODO: enhance frequency...
        vis_freq = torch.cat((dummy_lFreq_vis, dummy_mFreq_vis, hFreq_vis), dim=1)
        ir_freq = torch.cat((dummy_lFreq_ir, dummy_mFreq_ir, hFreq_ir), dim=1)
        
        return self.dyConv(ir_freq, vis_freq)  # x, ref
    
    