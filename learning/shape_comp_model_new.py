# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-09-06 11:35:30
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-02-22 19:20:36
# @Email:  cshzxie@gmail.com

import torch
import numpy as np

class DownSampleConv(torch.nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DownSampleConv, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv3d(in_channel, out_channel, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm3d(out_channel),
            torch.nn.Dropout(0.2),
            torch.nn.LeakyReLU(0.2),
        )
    
    def forward(self, x):
        return self.model(x)

class UpSampleConv(torch.nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UpSampleConv, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(in_channel, out_channel, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(out_channel),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU()
        )
    
    def forward(self, x):
        return self.model(x)

class ShapeCompModelNew(torch.nn.Module):
    def __init__(self, shape):
        super(ShapeCompModelNew, self).__init__()
        self.shape = np.array(shape)
        init_layer = 6
        self.init_layer = init_layer
        self.conv1 = DownSampleConv(1, init_layer)
        self.conv2 = DownSampleConv(init_layer, init_layer*2)
        self.conv3 = DownSampleConv(init_layer*2, init_layer*4)
        self.conv4 = DownSampleConv(init_layer*4, init_layer*8)
        feature_shape = self.shape//16
        self.fc5 = torch.nn.Sequential(
            torch.nn.Linear(init_layer*8*np.prod(feature_shape), 2048),
            torch.nn.Dropout(),
            torch.nn.ReLU(),
        )
        self.fc6 = torch.nn.Sequential(
            torch.nn.Linear(2048, init_layer*8*np.prod(feature_shape)),
            torch.nn.Dropout(),
            torch.nn.ReLU()
        )
        self.dconv7 = UpSampleConv(init_layer*8, init_layer*4)
        self.dconv8 = UpSampleConv(init_layer*4, init_layer*2)
        self.dconv9 = UpSampleConv(init_layer*2, init_layer)
        self.dconv10 = UpSampleConv(init_layer, init_layer)
        
        self.final_conv = torch.nn.Conv3d(init_layer, 1, kernel_size=3, padding=1)

    def forward(self, volume):
        up1 = self.conv1(volume)
        up2 = self.conv2(up1)
        up3 = self.conv3(up2)
        up4 = self.conv4(up3)
        feature_shape = self.shape//16
        features = self.fc5(up4.view(-1, self.init_layer*8*np.prod(feature_shape)))
        down4 = self.fc6(features).view(-1, self.init_layer*8, *feature_shape)
        down4 = down4 + up4
        down3 = self.dconv7(down4) + up3
        down2 = self.dconv8(down3) + up2
        down1 = self.dconv9(down2) + up1
        output = self.dconv10(down1) + volume
        output = self.final_conv(output)
        return output

# shape = (128,128,128)
# model = ShapeCompModelNew(shape)
# x = torch.rand(1,1,*shape)
# out = model(x)
# print(out.shape)
