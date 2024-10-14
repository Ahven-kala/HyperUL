# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
"""
    Function: SegFormer model (decoder).

    Reference: (1) MMSegmentation toolbox.
               (2) SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers.

    Note: revise some parts of code instead of these from MMSegmentation and MMCV.

    Date: June 22, 2021
    Update: October 14, 2021. Mapping the last outputs to Hyperbolic Space.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.wrappers import resize

import libs.hyptorch.nn as hypnn
from configs.segformer_B0_hyperul_cityscapes_config import parser


args = vars(parser.parse_args())

class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        """
        x: B x C x H x W
        """
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.proj(x)
        return x  # B x N x C


class ConvModule(nn.Module):
    """
    'conv/norm/activation'
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.norm = nn.BatchNorm2d(out_channels)   # nn.SyncBatchNorm
        # self.norm = nn.SyncBatchNorm(out_channels)

    def forward(self, x):
        x = F.relu(self.norm(self.conv(x)), inplace=True)
        return x


class SegFormerHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, feature_strides, in_channels, **kwargs):
        super(SegFormerHead, self).__init__()
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels

        self.num_classes = kwargs['num_classes']

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1
        )

        self.dropout = nn.Dropout2d(kwargs['dropout_ratio'])  # default: 0.1

        # comment the following four line code when doing segmentation on the segmentation datasets.
        # self.linear_pred_1 = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=(5, 5))
        # self.linear_pred_2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # self.linear_pred_3 = nn.Conv2d(self.num_classes, self.num_classes, kernel_size=(3, 3), stride=(2, 2))
        # self.linear_pred_4 = nn.Linear(19*12*12, 1000)

        # commented when doing classification on the ImageNet dataset.
        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=(1, 1))
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=4)

        # Hyperbolic space.
        # self.norm = nn.LayerNorm(self.num_classes)
        self.tp = hypnn.ToPoincare(c=args['c'],  # default: 1.0
                                   train_c=args['train_c'],  # store_true, default: False
                                   train_x=args['train_x'],  # store_true, default: False
                                   ball_dim=self.num_classes,  # args['dim']
                                   riemannian=args['is_rie'],  # whether use Riemannian optimizer.
                                   clip_r=args['clip_r'])  # clip_r: 2.3.
        self.mlr = hypnn.HyperbolicMLR(ball_dim=self.num_classes,  # args['dim']
                                       n_classes=self.num_classes,  # 19
                                       c=args['c'])  # default: 1.0

    def forward(self, inputs, img_height, img_width):
        # x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = inputs  # c1, c2, c3, c4: B x C x H x W

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])  # B x N x H x W
        _c4 = resize(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = resize(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))  # B x (4N) x H x W

        x = self.dropout(_c)

        # comment the following five line code when doing segmentation on the segmentation datasets.
        # x = self.linear_pred_1(x)
        # x = self.linear_pred_2(x)
        # x = self.linear_pred_3(x)
        # x = x.view(x.size(0), -1)
        # x = self.linear_pred_4(x)

        # comment the following two lines when doing classification on the ImageNet dataset.
        x = self.linear_pred(x)  # B x num_classes x H x W
        # x = x.permute(0, 2, 3, 1).contiguous()  # B x H x W x num_classes.
        # x = self.norm(x)  # layer normalization.
        # x = x.permute(0, 3, 1, 2).contiguous()
        x = self.upsampling(x)  # B x num_classes x H x W

        # Hyperbolic space.
        x = x.permute(0, 2, 3, 1).contiguous()  # B x H x W x num_classes.
        n, h, w, c = x.shape
        x = x.view(n*h*w, c)  # (B*H*W, C)
        x = self.tp(x)  # to poincare.
        x = self.mlr(x, c=self.tp.c).contiguous()
        x = x.view(n, h, w, c).permute(0, 3, 1, 2).contiguous()  # (B*H*W, C) -> B x num_classes x H x W
        return x


if __name__ == "__main__":
    dropout_ratio = 0.1
    num_classes = 19
    in_channels = [32, 64, 160, 256]
    feature_strides = [4, 8, 16, 32]
    model_name = 'mit_b0'
    decoder_params = dict(embed_dim=256)
    head = SegFormerHead(feature_strides, in_channels,
                         model_name=model_name,
                         dropout_ratio=dropout_ratio,
                         num_classes=num_classes,
                         decoder_params=decoder_params)

    for name, param in head.named_parameters():
        print(name, type(param.data), param.size())
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

