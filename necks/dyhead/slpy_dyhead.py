import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.ops as ops
import numpy as np
from ..builder import NECKS


@NECKS.register_module()
class SleepyDyHead(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers): # can add more args later
        super().__init__()
        self.layers = nn.ModuleList([
            DyHeadBlock(in_channels=in_channels, out_channels=out_channels) for i in range(num_layers)
        ])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            
        return x

    
class DyHeadBlock(nn.Module):
    '''
    Scale Aware Attention:
        Projecting the channel dimension to 1 and averaging over multiple feature maps to dynamically fuse features of 
        different scales, where the scale corresponds to the different levels of features learned from the multiple feature
        maps. We can further think of the channel dimension in tensors as a higher-dimensional understanding per pixel; scale 
        aware attention attempts to fuse these embeddings across multiple scales.
        
    Spatial Aware Attention:
        In image processing, we can intuitively think of the "spatial" aspect of images as the height and width dimensions. 
        Spatial aware attention applies attention in the form of a deformable convolution, filtering over the height and width
        of feature maps.
        
    Task Aware Attention:
        Applying fully connected linear layers to model specific connections of different representations of objects (joint learning
        and generalizable representations), and thus is intuitively applied after encoding spatial and scale-wise understanding of
        the feature maps.
        
    Implementation:
        Whereas the official paper specifies this order of transformations: task-aware(spatial-aware(scale-aware(x))),
        code implementations apply the sequence in the order task-aware(scale-aware(spatial-aware(x))), as the convolutions
        in the spatial-aware can be used for up/downsampling of feature maps for concatenation.
        
    See figure/additional notes
    '''
    def __init__(
        self,
        in_channels,
        out_channels
    ):
        super().__init__()
        
        self.spatial_offset_weights = nn.Conv2d(in_channels=in_channels, out_channels=27, kernel_size=3, stride=1, padding=1)
        self.spatial_low_conv = ops.DeformConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
        )
        self.spatial_mid_conv = ops.DeformConv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=3, 
            stride=1, 
            padding=1,
            bias=False
        )
        self.spatial_high_conv = ops.DeformConv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=3, 
            stride=1, 
            padding=1,
            bias=False
        )
        self.groupnorm = nn.GroupNorm(num_groups=16, num_channels=in_channels)
        
        self.scale_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Hardsigmoid()
        )
        
        self.task_aware_attention = DynamicReLU(in_channels=in_channels)
        
    def forward(self, x):
        '''
        Input: tuple of Tensors extracted/transformed by the neck (FPN, PAN, etc) of shape [b, c, h, w]
        Output: tuple of Tensors of shape [b, c, h, w] after applying sequence of attentions
        See figure for the tensor transformations at each iteration
        
        its offset_weights, but its not exactly offset_mask either. the 'mask' is actually delta m, or the modulation
        parameter per pixel between [0, 1] scaling the entire value to determine its importance. Thus, the modulation mechanism 
        provides the network module another dimension of freedom to adjust its spatial support regions.
        
        MMDet interpolates after spatial conv for higher feature maps, my implementation interpolate before
        '''
        out = []
        for i in range(len(x)):
            offset_weights = self.spatial_offset_weights(x[i])
            offset, weights = offset_weights[:, :18, :, :], torch.sigmoid(offset_weights[:, 18:, :, :])
            
            mid_features = self.spatial_mid_conv(x[i], offset, weights)
            mid_features = self.groupnorm(mid_features)
            out_features = mid_features * self.scale_attention(mid_features) 
            
            if i > 0:
                low_features = self.spatial_low_conv(x[i - 1], offset, weights)
                low_features = self.groupnorm(low_features)
                low_features = low_features * self.scale_attention(low_features)
                out_features = torch.stack([low_features, out_features], dim=0) # [2, b, c, h, w]
            if i < len(x) - 1:
                high_features = F.interpolate(x[i + 1], size=x[i].shape[-2:], mode='bilinear', align_corners=True)
                high_features = self.spatial_high_conv(high_features, offset, weights)
                high_features = self.groupnorm(high_features)
                high_features = high_features * self.scale_attention(high_features)
                
                if len(out_features.shape) > 4:
                    out_features = torch.cat([out_features, high_features.unsqueeze(0)], dim=0) # [3, b, c, h, w]
                else:
                    out_features = torch.stack([out_features, high_features], dim=0) # [2, b, c, h, w]
            
            out_features = torch.mean(out_features, dim=0) # [b, c, h, w]; averaging over the L dimension
            out_features = self.task_aware_attention(out_features)
            out.append(out_features)
            
        return out
    
    
class DynamicReLU(nn.Module):
    '''
    Dynamic ReLU (Task Aware Attention) for DyHead
    MMDetection uses convolutional layers, I follow the original paper and use linear layers
    '''
    def __init__(
        self, 
        in_channels, 
        reduction=4, 
        expansion=4, 
        init_alpha=[1.0, 0.0],
        init_beta=[0.0, 0.0]
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.init_alpha = init_alpha
        self.init_beta = init_beta
        self.linear1 = nn.Linear(in_features=in_channels, out_features=in_channels // reduction)
        self.linear2 = nn.Linear(in_features=in_channels // reduction, out_features=in_channels * expansion)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Hardsigmoid()
        
    def forward(self, x):
        '''
        Input: Tensor (feature map) of shape [b, c, h, w]
        Output: DyReLU'd Tensor (feature map) of shape [b, c, h, w]
        '''
        y = x
        x = self.pool(x) # [b, c, 1, 1]
        x = x.flatten(start_dim=1) # [b, c]
        x = self.relu(self.linear1(x))
        x = self.sigmoid(self.linear2(x)) # [0, 1.0]; hardsigmoid normalizes values between 0, 1
        x = (x - 0.5) * 2.0               # [-0.5, 0.5] -> [-1.0, 1.0]; shifting the values to range [-1, 1]
        x = x.unsqueeze(-1).unsqueeze(-1) # reshaping from [b, 4c] to [b, 4c, 1, 1]
        a1, a2, b1, b2 = torch.split(x, self.in_channels, dim=1) # [b, c, 1, 1],...,[b, c, 1, 1]
        a1 = a1 + self.init_alpha[0]
        a2 = a2 + self.init_alpha[1]
        b1 = b1 + self.init_beta[0]
        b2 = b2 + self.init_beta[1]
        out = torch.max(a1 * y + b1, a2 * y + b2)
        return out