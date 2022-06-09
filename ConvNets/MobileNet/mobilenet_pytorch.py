import torch
import torch.nn as nn
from torch import Tensor

class ConvBlock(nn.Module):
    """ Convolution Block with Conv2d layer, Batch Normalization and ReLU. """
    def __init__(
        self,
        in_channels : int,
        out_channels : int,
        kernel_size : int,
        stride : int,
        padding : int, 
        groups = 1,
        bias=False,     
        ):

        super().__init__()

        self.c = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.relu(self.bn(self.c(x)))

class DepthwiseConvBlock(nn.Module):
    """ Depthwise Separable Convolution Block. """ 
    def __init__(
        self,
        in_channels : int,
        out_channels : int,
        stride : int,
        ):
        
        super().__init__()

        self.depthwise = ConvBlock(in_channels, in_channels, 3, stride, 1, in_channels)
        self.pointwise = ConvBlock(in_channels, out_channels, 1, 1, 0)
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class MobileNet(nn.Module):
    """ Baseline MobileNet model that takes in width(alpha)"""
    def __init__(
        self,
        alpha : float,
        in_channels=3,
        classes=1000
        ):    

        super().__init__()
        
        """ List of strides and channels for Depthwise Blocks """
        strides = [1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1]
        channels = [32, 64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 1024, 1024]

        if alpha < 1:
            channels = [int(channel * alpha) for channel in channels]

        """ List of Depthwise Blocks. """
        self.blocks = nn.ModuleList([])
        for i, stride in enumerate(strides):
            self.blocks.append(DepthwiseConvBlock(channels[i], channels[i+1], stride))

        self.conv1 = ConvBlock(in_channels, channels[0], 3, 2, 1)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(channels[-1], classes)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        for block in self.blocks:
            x = block(x)
        
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    rho = 1
    alpha = 1
    res = int(224 * rho)

    net = MobileNet(alpha)
    print(net(torch.rand(1, 3, res, res)).shape)