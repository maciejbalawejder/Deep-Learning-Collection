import torch
import torch.nn as nn
from torch import Tensor

class ConvBlock(nn.Module):
    """ Convolution Block with Conv2d layer, Batch Normalization and ReLU. Act input defines whetver to apply activation or not. """
    def __init__(
        self,
        in_channels : int,
        out_channels : int,
        kernel_size : int,
        stride : int,
        padding : int, 
        groups = 1,
        act=True,
        bias=False     
        ):

        super().__init__()

        self.c = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        
        if act:
            self.relu = nn.ReLU6()
        else:
            self.relu = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.relu(self.bn(self.c(x)))

class InvertedResBlock(nn.Module):
    """ Inverted Residual Block with expansion(exp) parameter. """ 
    def __init__(
        self,
        in_channels : int,
        out_channels : int,
        stride : int,
        exp : int
        ):
        
        super().__init__()

        self.add = True if in_channels == out_channels and stride == 2 else False    
        exp_channels = in_channels * exp

        """ 3 Convolutions : Expansion, Depthwise, Pointwise. """
        self.expansion = ConvBlock(in_channels, exp_channels, 1, 1, 0, act=True)
        self.dwise = ConvBlock(exp_channels, exp_channels, 3, stride, 1, groups=exp_channels)
        self.pwise = ConvBlock(exp_channels, out_channels, 1, 1, 0, act=False)
    
        
    def forward(self, x: Tensor) -> Tensor:
        res = x
        x = self.expansion(x)
        x = self.dwise(x)
        x = self.pwise(x)
        if self.add:
            x = x + res

        return x

class MobileNetv2(nn.Module):
    """ Baseline MobileNet model that takes in width(alpha)"""
    def __init__(
        self,
        alpha : float,
        in_channels=3,
        classes=1000,
        show=False
        ):    

        super().__init__()
        self.show = show # prints shapes of outputs between layers
        """ List of strides(s) and channels(c), expansion(t) for Inverted Residual Blocks and how many times they are repeated(n). """
        s = [1, 2, 2, 2, 1, 2, 1]
        n = [1, 2, 3, 4, 3, 3, 1]
        t = [1, 6, 6, 6, 6, 6, 6]
        c = [32, 16, 24, 32, 64, 96, 160, 320, 1280]

        if alpha > 1:
            c = [int(channel * alpha) for channel in c]
        
        elif alpha < 1:
            c = [int(channel * alpha) for channel in c[:-1]]

        """ List of Inverted Residual Blocks. """
        self.blocks = nn.ModuleList([])
        for i in range(len(s)):
            self._add_layer(c[i], c[i+1], s[i], n[i], t[i])

        self.conv1 = nn.Conv2d(in_channels, c[0], 3, 2, 1)

        self.classifier = nn.Sequential(
            nn.Conv2d(c[-2], c[-1], 1, 1, 0),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(c[-1], classes)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        for block in self.blocks:
            x = block(x)
            if self.show : print(x.shape)
        
        x = self.classifier(x)
        return x

    def _add_layer(self, in_channels, out_channels, stride, n, t):
        """ Add layer function which takes input and output channels, stride, repeats, expansion. """
        if n == 1:
            self.blocks.append(InvertedResBlock(in_channels, out_channels, stride, t))
        else:
            self.blocks.append(InvertedResBlock(in_channels, in_channels, 1, t))
            for _ in range(n-2):
                self.blocks.append(InvertedResBlock(in_channels, in_channels, 1, t))

            self.blocks.append(InvertedResBlock(in_channels, out_channels, stride, t))


if __name__ ==  "__main__":
    """ Second hyperparameter is resolution multiplayer(rho). """
    """ Baseline configuration is alpha=1, rho=1. """

    rho = 1
    alpha = 1
    res = int(224 * rho)

    net = MobileNetv2(alpha)
    print(net(torch.rand(1, 3, res, res)).shape)