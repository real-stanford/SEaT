from torch import nn
from torch.nn import functional as F

class Conv3DChannelUp(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size=7, padding=3) -> None:
        super().__init__()
        self.conv0 = nn.Conv3d(inplanes, outplanes // 2, kernel_size=kernel_size, stride=1, padding=padding)
        self.bn0 = nn.BatchNorm3d(outplanes // 2)
        self.conv1 = nn.Conv3d(outplanes // 2, outplanes, kernel_size=kernel_size, stride=1, padding=padding)
        self.bn1 = nn.BatchNorm3d(outplanes)
    
    def forward(self, x):
        x = F.leaky_relu(self.bn0(self.conv0(x)))
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        return x

class ResBlock3D(nn.Module):
    def __init__(self, inplanes, kernel_size=7, padding=3) -> None:
        super().__init__()
        self.conv0 = nn.Conv3d(inplanes, inplanes, kernel_size=kernel_size, stride=1, padding=padding)
        self.bn0 = nn.BatchNorm3d(inplanes)

        self.conv1 = nn.Conv3d(inplanes, inplanes, kernel_size=kernel_size, stride=1, padding=padding)
        self.bn1 = nn.BatchNorm3d(inplanes)

    def forward(self, x):
        residual = x
        x = nn.functional.leaky_relu(self.bn0(self.conv0(x)))
        x = self.bn1(self.conv1(x))
        x = x + residual
        x = nn.functional.leaky_relu(x)
        return x

class DownSamplingBlock(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size=3) -> None:
        super().__init__()
        self.cu0 = Conv3DChannelUp(inplanes, outplanes)
        self.res1 = ResBlock3D(outplanes)
        self.mp2 = nn.MaxPool3d(kernel_size=kernel_size, stride=2, padding=1)
    
    def forward(self, x):
        x = self.cu0(x)
        x = self.res1(x)
        x = self.mp2(x)
        return x
