from torch import nn
import torch
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1, norm=True, relu=True, pool=False, upsm=False):
        super().__init__()

        self.conv = nn.Conv3d(inplanes, planes, kernel_size=3, stride=stride,
                              padding=dilation, dilation=dilation, bias=not norm)
        self.norm = nn.BatchNorm3d(planes) if norm else None
        self.relu = nn.LeakyReLU(inplace=True) if relu else None
        self.pool = nn.MaxPool3d(
            kernel_size=3, stride=2, padding=1) if pool else None
        self.upsm = upsm

    def forward(self, x):
        out = self.conv(x)

        out = out if self.norm is None else self.norm(out)
        out = out if self.relu is None else self.relu(out)
        out = out if self.pool is None else self.pool(out)
        out = out if not self.upsm else F.interpolate(
            out, scale_factor=2, mode='trilinear', align_corners=True)

        return out


class ResBlock3D(nn.Module):
    def __init__(self, inplanes, planes, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv3d(
            inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class VolumeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        input_channel = 1
        self.conv00 = ConvBlock3D(
            input_channel, 16, stride=2, dilation=1)  # 16X16X16

        self.conv10 = ConvBlock3D(16, 32, stride=2, dilation=1)  # 8X8X8
        self.conv11 = ConvBlock3D(32, 32, stride=1, dilation=1)
        self.conv12 = ConvBlock3D(32, 32, stride=1, dilation=1)
        self.conv13 = ConvBlock3D(32, 32, stride=1, dilation=1)

        self.conv20 = ConvBlock3D(32, 64, stride=2, dilation=1)  # 4X4X4
        self.resn21 = ResBlock3D(64, 64)
        self.resn22 = ResBlock3D(64, 64)

    def forward(self, x):
        x0 = self.conv00(x)

        x1 = self.conv10(x0)
        x1 = self.conv11(x1)
        x1 = self.conv12(x1)
        x1 = self.conv13(x1)

        x2 = self.conv20(x1)
        x2 = self.resn21(x2)
        x2 = self.resn22(x2)
        x2 = x2
        return x2, (x1, x0)


class FeatureDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv00 = ConvBlock3D(64, 32, upsm=True)  # 8X8X8
        self.conv01 = ConvBlock3D(32, 32)

        self.conv10 = ConvBlock3D(32*2, 16, upsm=True)  # 16X16X16
        self.conv11 = ConvBlock3D(16, 16)

        self.conv20 = ConvBlock3D(16*2, 8, upsm=True)  # 32X32X32
        self.conv21 = ConvBlock3D(8, 1)

    def forward(self, x, cache):
        m0, m1 = cache
        
        x0 = self.conv00(x)
        x0 = self.conv01(x0)
        
        x1 = self.conv10(torch.cat([x0, m0], dim=1))
        x1 = self.conv11(x1)

        x2 = self.conv20(torch.cat([x1, m1], dim=1))
        x2 = self.conv21(x2)

        return x2


class ShapeCompModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.ve = VolumeEncoder()
        self.fd = FeatureDecoder()
    
    def forward(self, x):
        return self.fd(*self.ve(x))
        