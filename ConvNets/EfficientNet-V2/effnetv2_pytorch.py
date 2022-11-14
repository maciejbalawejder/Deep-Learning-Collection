import torch
import torch.nn as nn
from torch import Tensor
from stochastic_depth import StochasticDepth

class ConvBlock(nn.Module):
    """Convolution Block with Batch Normalization and Activation Function

    Parameters:
        in_channels : int - number of input channels
        out_channels : int - number of output channels
        kernel_size : int - int value of kernel 
        stride : int - single number
        groups : int - number of groups the input channels are split into
        act : bool - defines if there is activation function
        bias : bool - defines if convolution has bias

    Attributes:
        padding : int - number of padding applied to the input
        c : nn.Conv2d - convolution layer
        bn : nn.BatchNorm2d - batch normalization
        silu : nn.SiLU - silu activation function
    
    """
    def __init__(
                self, 
                in_channels, 
                out_channels, 
                kernel_size, 
                stride, 
                groups=1, 
                act=True, 
                bias=False
                ):

        super().__init__()

        # If kernel_size = 1 -> padding = 0, kernel_size = 3 -> padding = 1, kernel_size = 5, padding = 2.
        padding = kernel_size // 2
        self.c = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x) -> Tensor:
        """Forward pass.

        Parameters:
            x : torch.Tensor - input image with shape (batch_size, in_channels, height, width).

        Returns :
            torch.Tensor - output tensor of shape (batch_size, out_channels, height, width).

        """
        
        x = self.silu(self.bn(self.c(x)))
        return x

# Squeeze-and-Excitation Block
class SeBlock(nn.Module):
    def __init__(self, in_channels, r):
        super().__init__()
        C = in_channels
        self.globpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(C, C//r, bias=False)
        self.fc2 = nn.Linear(C//r, C, bias=False)
        self.silu = nn.SiLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: [batch, channels, height, width]. 
        f = self.globpool(x)
        f = torch.flatten(f,1)
        f = self.silu(self.fc1(f))
        f = self.sigmoid(self.fc2(f))
        f = f[:,:,None,None]
        # f shape: [batch, channels, 1, 1]

        scale = x * f
        return scale

class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, exp, r, p):
        super().__init__()
        exp_channels = in_channels * exp
        self.add = in_channels == out_channels and stride == 1
        self.c1 = ConvBlock(in_channels, exp_channels, 1, 1) if exp > 1 else nn.Identity()
        self.c2 = ConvBlock(exp_channels, exp_channels, kernel_size, stride, exp_channels)
        self.se = SeBlock(exp_channels, r)
        self.c3 = ConvBlock(exp_channels, out_channels, 1, 1, act=False)

        # Stochastic Depth module with default survival probability 0.5
        self.sd = StochasticDepth()

    def forward(self, x):
        f = self.c1(x)
        f = self.c2(f)
        f = self.se(f)
        f = self.c3(f)

        if self.add:
            f = x + f

        f = self.sd(f)

        return f