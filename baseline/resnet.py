# Reference: Yulong Li for providing this pytorch conversion: https://github.com/RaindragonD/transporternet-torch
# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python
"""Resnet module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import torch
from torch import nn
import numpy as np

# Todo
# Set eager execution
# if not tf.executing_eagerly():
#   tf.compat.v1.enable_eager_execution()

class IdentityBlock(nn.Module):
    """The identity block is the block that has no conv layer at shortcut."""

    def __init__(self, 
                inplanes,
                kernel_size,
                filters,
                activation=True,
                include_batchnorm=False):
        super().__init__()
        filters1, filters2, filters3 = filters
        self.conv1 = nn.Conv2d(inplanes, filters1, kernel_size=(1,1), dilation=(1,1), bias=False)
        self.bn1 = nn.BatchNorm2d(filters1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(filters1, filters2, kernel_size=kernel_size, dilation=(1,1), bias=False)
        self.bn2 = nn.BatchNorm2d(filters2)
        self.conv3 = nn.Conv2d(filters2, filters3, kernel_size=(1,1), dilation=(1,1), bias=False)
        self.bn3 = nn.BatchNorm2d(filters3)
        self.kernel_size = kernel_size
        self.activation = activation
        self.include_batchnorm = include_batchnorm
        # self.apply(weight_init)

    def forward(self, x):
        out = self.conv1(x)
        if self.include_batchnorm:
            out = self.bn1(out)
        if self.activation:
            out = self.relu(out)
        
        # padding = same
        padding = calculate_padding(x.shape[2:], self.kernel_size, (1,1))
        out = nn.functional.pad(out, padding)
        out = self.conv2(out)
        if self.include_batchnorm:
            out = self.bn2(out)
        if self.activation:
            out = self.relu(out)
        
        out = self.conv3(out)
        if self.include_batchnorm:
            out = self.bn3(out)
            
        out = torch.add(out, x)
        if self.activation:
            out = self.relu(out)
        return out

class ConvBlock(nn.Module):
    """A block that has a conv layer at shortcut. """  

    def __init__(self,
                inplanes,
                kernel_size,
                filters,
                strides=(2, 2),
                activation=True,
                include_batchnorm=False):
        super().__init__()
        filters1, filters2, filters3 = filters
        self.conv1 = nn.Conv2d(inplanes, filters1, kernel_size=(1,1), stride=strides, dilation=(1,1), bias=False)
        self.bn1 = nn.BatchNorm2d(filters1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(filters1, filters2, kernel_size=kernel_size, dilation=(1,1), bias=False)
        self.bn2 = nn.BatchNorm2d(filters2)
        self.conv3 = nn.Conv2d(filters2, filters3, kernel_size=(1,1), dilation=(1,1), bias=False)
        self.bn3 = nn.BatchNorm2d(filters3)
        self.shortcut_conv = nn.Conv2d(inplanes, filters3, kernel_size=(1,1), stride=strides, dilation=(1,1), bias=False)
        self.shortcut_bn = nn.BatchNorm2d(filters3)
        self.kernel_size = kernel_size
        self.activation = activation
        self.include_batchnorm = include_batchnorm
        # self.apply(weight_init)

    def forward(self, x):
        out = self.conv1(x)
        if self.include_batchnorm:
            out = self.bn1(out)
        if self.activation:
            out = self.relu(out)
        
        # padding = same
        padding = calculate_padding(x.shape[2:], self.kernel_size, (1,1))
        out = nn.functional.pad(out, padding)
        out = self.conv2(out)
        if self.include_batchnorm:
            out = self.bn2(out)
        if self.activation:
            out = self.relu(out)
        
        out = self.conv3(out)
        if self.include_batchnorm:
            out = self.bn3(out)

        shortcut = self.shortcut_conv(x)
        if self.include_batchnorm:
            shortcut = self.shortcut_bn(shortcut)
        
        out = torch.add(out, shortcut)
        if self.activation:
            out = self.relu(out)
        return out

class ResNet43_8s(nn.Module):
    """Build Resent 43 8s."""

    def __init__(self,
                input_dim,
                output_dim,
                include_batchnorm=False,
                cutoff_early=False):
        super().__init__()
        self.include_batchnorm = include_batchnorm
        self.cutoff_early = cutoff_early
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) 
        # Todo: padding = "same"
        self.conv2d = nn.Conv2d(input_dim, 64, kernel_size=(3,3), stride=(1,1), bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.cutoff_convblock = ConvBlock(64, 5, [64,64,output_dim], strides=(1,1), include_batchnorm=include_batchnorm)
        self.cutoff_identityblock = IdentityBlock(output_dim*2, 5, [64,64,output_dim])
        self.conv1 = ConvBlock(64, 3, [64,64,64], strides=(1,1))
        self.identity1 = IdentityBlock(64, 3, [64,64,64])
        self.conv2 = ConvBlock(64, 3, [128,128,128], strides=(2,2))
        self.identity2 = IdentityBlock(128, 3, [128,128,128])
        self.conv3 = ConvBlock(128, 3, [256,256,256], strides=(2,2))
        self.identity3 = IdentityBlock(256, 3, [256,256,256])
        self.conv4 = ConvBlock(256, 3, [512,512,512], strides=(2,2))
        self.identity4 = IdentityBlock(512, 3, [512,512,512])
        self.conv5 = ConvBlock(512, 3, [256,256,256], strides=(1,1))
        self.identity5 = IdentityBlock(256, 3, [256,256,256])
        self.conv6 = ConvBlock(256, 3, [128,128,128], strides=(1,1))
        self.identity6 = IdentityBlock(128, 3, [128,128,128])
        self.conv7 = ConvBlock(128, 3, [64,64,64], strides=(1,1))
        self.identity7 = IdentityBlock(64, 3, [64,64,64])
        self.conv8 = ConvBlock(64, 3, [16,16,output_dim], strides=(1,1), activation=False)
        self.identity8 = IdentityBlock(output_dim, 3, [16,16,output_dim], activation=False)

    def forward(self, x):
        out = self.conv2d(x)
        if self.include_batchnorm:
            out = self.bn(out)
        out = self.relu(out)
        if self.cutoff_early:
            out = self.cutoff_convblock(out)
            out = self.cutoff_identityblock(out)
            return out
        
        out = self.conv1(out)
        out = self.identity1(out)
        out = self.conv2(out)
        out = self.identity2(out)
        out = self.conv3(out)
        out = self.identity3(out)
        out = self.conv4(out)
        out = self.identity4(out)
        out = self.conv5(out)
        out = self.identity5(out)

        out = self.upsample(out)
        out = self.conv6(out)
        out = self.identity6(out)

        out = self.upsample(out)
        out = self.conv7(out)
        out = self.identity7(out)

        out = self.upsample(out)
        out = self.conv8(out)
        out = self.identity8(out)
        return out

def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))

def calculate_padding(in_shape, kernel_size, strides):
    in_height, in_width = in_shape
    if type(kernel_size) == int:
        filter_height = kernel_size
        filter_width = kernel_size
    else:
        filter_height, filter_width = kernel_size
        
    if type(strides) == int:
        stride_height = strides
        stride_width = strides
    else:
        stride_height, stride_width = strides
    
    out_height = np.ceil(float(in_height) / float(stride_height))
    out_width  = np.ceil(float(in_width) / float(stride_width))

    if (in_height % stride_height == 0):
        pad_along_height = max(filter_height - stride_height, 0)
    else:
        pad_along_height = max(filter_height - (in_height % stride_height), 0)
    if (in_width % stride_width == 0):
        pad_along_width = max(filter_width - stride_width, 0)
    else:
        pad_along_width = max(filter_width - (in_width % stride_width), 0)

    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left

    return pad_left, pad_right, pad_top, pad_bottom