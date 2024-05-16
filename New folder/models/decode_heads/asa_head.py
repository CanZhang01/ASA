# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import numpy as np
import pandas as pd
import torch

from mmcv.cnn import NonLocal2d
from torch import nn
import mmcv
from ..builder import HEADS
from .fcn_head import FCNHead
import pandas as pd



class NewNonLocal2d(NonLocal2d):
    """Adaptive sparse attention module.

    Args:
        temperature (float): Temperature to adjust attention. Default: 0.05
    """

    def __init__(self, *arg, **kwargs):
        super().__init__(*arg, **kwargs)
        self.se = nn.Sequential(
            nn.Conv2d(self.in_channels, out_channels= 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.se2 = nn.Sequential(
            nn.Conv2d(self.in_channels, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
    def generate_mask(self, feature_size, neighbor_size):
        """
        Generate a square mask for the sequence.
        """
        h, w = feature_size
        hm, wm = neighbor_size
        mask = torch.zeros(h, w, h, w)
        for idx_h1 in range(h):
            for idx_w1 in range(w):
                idx_h2_start = max(idx_h1 - hm // 2, 0)
                idx_h2_end = min(idx_h1 + hm // 2 + 1, h)
                idx_w2_start = max(idx_w1 - wm // 2, 0)
                idx_w2_end = min(idx_w1 + wm // 2 + 1, w)
                mask[
                idx_h1, idx_w1, idx_h2_start:idx_h2_end, idx_w2_start:idx_w2_end
                ] = 1
        mask = mask.view(h * w, h * w).cuda()
        return mask
    def embedded_gaussian(self, theta_x, phi_x):
        pairwise_weight = torch.matmul(theta_x, phi_x)

        if self.use_scale:
            pairwise_weight /= theta_x.shape[-1]**0.5
        pairwise_weight = pairwise_weight.softmax(dim=-1)
        return pairwise_weight

    '''
        def embedded_gaussian(self, h_x, phi_x):
            """Embedded gaussian with temperature."""

            # NonLocal2d pairwise_weight: [N, HxW, HxW]
            pairwise_weight = torch.matmul(h_x, phi_x)
            if self.use_scale:

                # theta_x.shape[-1] is `self.inter_channels`
                pairwise_weight /= torch.tensor(
                    h_x.shape[-1],
                    dtype=torch.float,
                    device=pairwise_weight.device)**torch.tensor(
                        0.5, device=pairwise_weight.device)
            pairwise_weight /= torch.tensor(
                self.temperature, device=pairwise_weight.device)

            pairwise_weight = pairwise_weight.softmax(dim=-1)

            return pairwise_weight
        '''


    def forward(self, x):
        n = x.size(0)
        g_x = self.g(x).view(n,self.inter_channels,-1)
        g_x = g_x.permute(0,2,1)


        # theta_x: [N, HxW, C], phi_x: [N, C, HxW]  Q
        if self.mode == 'gaussian':
            theta_x = x.view(n, self.in_channels, -1) #Q
            theta_x = theta_x.permute(0, 2, 1)
            if self.sub_sample:
                phi_x = self.phi(x).view(n, self.in_channels, -1)  # K
            else:
                phi_x = x.view(n, self.in_channels, -1)

        elif self.mode == 'concatenation':
            theta_x = self.theta(x).view(n, self.inter_channels, -1, 1) #Q
            phi_x = self.phi(x).view(n, self.inter_channels, 1, -1)  # K

        else:
            theta_x = self.theta(x).view(n, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            phi_x = self.phi(x).view(n, self.inter_channels, -1)  #phi_x torch.Size([2, 256, 8192])


        pairwise_func = getattr(self, self.mode)
        pairwise_weight = pairwise_func(theta_x, phi_x)
        LA = self.generate_mask([x.size(2), x.size(3)], neighbor_size=[3, 3]).float()
        SLA = torch.sum(LA, dim = 1)
        LA = LA * pairwise_weight
        meansholds1 = torch.sum(LA, dim = 2)/SLA
        meansholds1 = meansholds1.unsqueeze(dim = 2)
        threshold2 = self.se2(x)
        threshold2 = threshold2.view(n, -1, 1)
        LA = LA - threshold2 * meansholds1
        LA = (1 - torch.sign(- LA))/2

        mean_sholds = torch.mean(pairwise_weight, dim=2)
        mean_sholds = mean_sholds.unsqueeze(dim=2)
        thresholds1 =self.se(x)

        thresholds1 =thresholds1.view(n, -1, 1)
        A = pairwise_weight - mean_sholds * thresholds1
        A = (1 - torch.sign(- A))/2


        A = torch.mul(A.permute(0, 2, 1), A)
        A = torch.max(A, LA)
        A = torch.mul(A, pairwise_weight)
        sum_sholds = torch.sum(A, dim = 2)
        sum_sholds = sum_sholds.unsqueeze(dim=2)
        A =A / (sum_sholds + 1e-12)
        y = torch.matmul(A, g_x)


        # y: [N, C, H, W]
        y = y.permute(0, 2, 1).contiguous().reshape(n, self.inter_channels,
                                    *x.size()[2:])




        output = x + self.conv_out(y)

        return output


@HEADS.register_module()
class ASAHead(FCNHead):
    """Adaptive Sparse Attention Net.
    """

    def __init__(self,
                 reduction=2,
                 use_scale=False,
                 mode='embedded_gaussian',
                 temperature=0.05,
                 **kwargs):
        super(ASAHead, self).__init__(num_convs=2, **kwargs)
        self.count = 0
        self.reduction = reduction
        self.use_scale = use_scale
        self.mode = mode
        self.temperature = temperature
        self.non_blocak = NewNonLocal2d(
            in_channels=self.channels,
            reduction=self.reduction,
            use_scale=self.use_scale,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            mode=self.mode,
            )

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        output = self.convs[0](x)
        output = self.non_blocak(output)
        output = self.convs[1](output)

        if self.concat_input:
            output = self.conv_cat(torch.cat([x, output], dim=1))
        output = self.cls_seg(output)

        return output
