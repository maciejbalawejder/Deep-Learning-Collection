import torch
import torch.nn as nn
from torch import Tensor
from stochastic_depth import StochasticDepth
from effnetv2_configs import get_config

class ConvBlock(nn.Module):
    """Convolution Block with Batch Normalization and Activation Function

    Parameters
    ----------
        in_channels : int 
            number of input channels

        out_channels : int
            number of output channels

        kernel_size : int
            int value of kernel 
            
        stride : int
            stride size

        groups : int
            number of groups the input channels are split into

        act : bool
            defines if there is activation function

        bias : bool
            defines if convolution has bias parameter

    Attributes
    ----------
        c : nn.Conv2d
            convolution layer

        bn : nn.BatchNorm2d
            batch normalization

        silu : nn.SiLU
            silu activation function
    
    """
    def __init__(
                self, 
                in_channels, 
                out_channels, 
                kernel_size, 
                stride, 
                groups=1, 
                act=True, 
                bias=None
                ):

        super().__init__()

        # If kernel_size = 1 -> padding = 0, kernel_size = 3 -> padding = 1, kernel_size = 5, padding = 2.
        padding = kernel_size // 2
        self.c = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x) -> Tensor:
        """Forward pass.

        Parameters
        ----------
            x : torch.Tensor
                Input tensor with shape (batch_size, in_channels, height, width).

        Returns
        -------
            ret : torch.Tensor
                Output tensor of shape (batch_size, out_channels, height, width).

        """
        
        x = self.silu(self.bn(self.c(x)))
        return x

class SeBlock(nn.Module):
    """Squeeze-and-Excitation Block

    Parameters
    ----------
        in_channels : int
            number of input channels

        r : int
            reduction ratio [0,1]

    Attributes
    ----------
        globpool : nn.AdaptiveAvgPool2d 
            global pooling operation(squeeze) that brings down spatial dimensions to (1,1) for all channels

        fc1 : nn.Conv2d
            first linear/convolution layer that brings down the number of channels the reduction space

        fc2 : nn.Conv2d
            second linear/convolution layer that brings up the number of channels the input space(excitation)

        silu : nn.SiLU
            silu activation function

        sigmoid : nn.Sigmoid
            sigmoid activation function

    """

    def __init__(
                self, 
                in_channels, 
                r
                ):

        super().__init__()

        C = in_channels
        sqeeze_channels = max(1, int(C*r))
        self.globpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Conv2d(C, sqeeze_channels, 1)
        self.fc2 = nn.Conv2d(sqeeze_channels, C, 1)
        self.silu = nn.SiLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x) -> Tensor:
        """Forward pass.

        Parameters
        ----------
            x : torch.Tensor
                Input tensor with shape (batch_size, in_channels, height, width).

        Returns
        -------
            ret : torch.Tensor
                Output tensor of shape (batch_size, out_channels, height, width).

        """
        f = self.globpool(x)
        f = self.silu(self.fc1(f))
        f = self.sigmoid(self.fc2(f))
        scale = x * f
        return scale

class MBConv(nn.Module):
    """MBConv Block

    Parameters
    ----------
        in_channels : int
            number of input channels

        out_channels : int
            number of output channels

        kernel_size : int
            int value of kernel 

        stride : int
            single number

        exp : int
            expansion ratio

        r : int
            reduction ratio for SEBlock

        p_sd : float
            stochastic depth probablity

    Attributes
    ----------
        use_connection : bool
            defines whether use residual connection or not

        block : nn.Sequential
            collection of expansion operation, depthwise conv, seblock and reduction convolution

        sd : nn.Module
            stochastic depth operation that randomly drops a examples in the batch
    """

    def __init__(
                self, 
                in_channels, 
                out_channels, 
                kernel_size, 
                stride, 
                exp, 
                r, 
                p_sd
                ):

        super().__init__()

        exp_channels = in_channels * exp
        self.use_connection = in_channels == out_channels and stride == 1
        self.block = nn.Sequential(
            ConvBlock(in_channels, exp_channels, 1, 1) if exp > 1 else nn.Identity(),
            ConvBlock(exp_channels, exp_channels, kernel_size, stride, groups=exp_channels),
            SeBlock(exp_channels, r),
            ConvBlock(exp_channels, out_channels, 1, 1, act=False)
        )
        
        self.sd = StochasticDepth(p_sd)

    def forward(self, x) -> Tensor:
        """Forward pass.

        Parameters
        ----------
            x : torch.Tensor
                Input tensor of shape (batch_size, in_channels, height, width).

        Returns
        -------
            ret : torch.Tensor
                Output tensor of shape (batch_size, out_channels, height, width).

        """
        f = self.block(x)

        if self.use_connection:
            f = self.sd(f)
            f = x + f

        return f

class FusedMBConv(nn.Module):
    """Fused-MBConv Block
        I removed the SEBlock since it doesn't exist in any of the configurations.

    Parameters
    ----------
        in_channels : int
            number of input channels

        out_channels : int
            number of output channels

        kernel_size : int
            int value of kernel 

        stride : int
            single number

        exp : int
            expansion ratio

        p_sd : float
            stochastic depth probablity

    Attributes
    ----------
        use_connection : bool
            defines whether use residual connection or not

        block : nn.Sequential
            collection of "fused" convolution and reduction convolution

        sd : nn.Module
            stochastic depth operation that randomly drops a examples in the batch
    """

    def __init__(
                self, 
                in_channels, 
                out_channels, 
                kernel_size, 
                stride, 
                exp, 
                p_sd
                ):

        super().__init__()

        exp_channels = in_channels * exp
        self.use_connection = in_channels == out_channels and stride == 1
        self.block = nn.Sequential(
            ConvBlock(in_channels, exp_channels, kernel_size, stride),
            ConvBlock(exp_channels, out_channels, 1, 1, act=False)
        ) if exp > 1 else ConvBlock(in_channels, out_channels, kernel_size, stride)
        
        self.sd = StochasticDepth(p_sd)

    def forward(self, x) -> Tensor:
        """Forward pass.

        Parameters
        ----------
            x : torch.Tensor
                Input tensor of shape (batch_size, in_channels, height, width).

        Returns
        -------
            ret : torch.Tensor
                Output tensor of shape (batch_size, out_channels, height, width).

        """
        f = self.block(x)

        if self.use_connection:
            f = self.sd(f)
            f = x + f

        return f

class EfficientNetV2(nn.Module):
    """EfficientNetV2 architecture
    
    Parameters
    ----------
        config_name : str
            name of the configuration, there are available 5 options : Base, S, M, L, XL

    Attributes
    ----------
        use_connection : bool

    
    """

    def __init__(
            self, 
            config_name,
            in_channels=3,
            classes=1000,
            add_head=True,
            ):

        super().__init__()

        self.add_head = add_head
        config = get_config(config_name)
        self.blocks = nn.ModuleList([])
        p_sd = 0.5

        for n, stage in enumerate(config):
            r, k, s, e, i, o, c = stage.split("_") # c -> fuse block or se
            r = int(r[1:])

            # Only first MBConv has stride 2 or 1
            if "c" in c:
                self.blocks.append(FusedMBConv(int(i[1:]), int(o[1:]), int(k[1:]), int(s[1:]), int(e[1:]), p_sd))
            else:
                self.blocks.append(MBConv(int(i[1:]), int(o[1:]), int(k[1:]), int(s[1:]), int(e[1:]), float(c[-4:]), p_sd))

            if r > 1:
                for _ in range(r-1):
                    if "c" in c:
                        self.blocks.append(FusedMBConv(int(o[1:]), int(o[1:]), int(k[1:]), 1, int(e[1:]), p_sd))
                    else:
                        self.blocks.append(MBConv(int(o[1:]), int(o[1:]), int(k[1:]), 1, int(e[1:]), float(c[-4:]), p_sd))

            if n == 0:
                first_in_channel = int(i[1:])

            if n == len(config) - 1:
                last_out_channel = int(o[1:])

        self.stem = ConvBlock(
            in_channels=in_channels, 
            out_channels=first_in_channel, 
            kernel_size=3,
            stride=1
        )

        self.final_conv = nn.Conv2d(last_out_channel, 1280, 1, 1, 0)

        self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1,1)),
                nn.Flatten(start_dim=1,end_dim=3),
                nn.Linear(1280, classes)
        )


            
    def forward(self, x) -> Tensor:
        """Forward pass.

        Parameters
        ----------
            x : torch.Tensor
                Input tensor of shape (batch_size, in_channels, height, width).

        Returns
        -------
            ret : torch.Tensor
                Output tensor of shape (batch_size, classes).

        """
        x = self.stem(x)
        print(x.shape)
        for block in self.blocks:
            x = block(x)
            print(x.shape)

        x = self.final_conv(x)
        print(x.shape)
        if self.add_head:
            x = self.head(x)
        return x


# sanity check
if __name__ == "__main__":
    effnet = EfficientNetV2("Base")
    img = torch.rand((1, 3, 224, 224))
    print(effnet(img).shape)
    