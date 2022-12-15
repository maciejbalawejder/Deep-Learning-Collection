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

        activation : nn.SiLU or nn.Identity()
            silu activation function or identity function(no operation)
    
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
        self.activation = nn.SiLU(inplace=True) if act else nn.Identity()

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
        
        x = self.activation(self.bn(self.c(x)))
        return x

class SeBlock(nn.Module):
    """Squeeze-and-Excitation Block

    Parameters
    ----------
        in_channels : int
            number of input channels

        squeeze_channels : int
            number of channels to squeeze to

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
                squeeze_channels
                ):

        super().__init__()

        C = in_channels
        self.globpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Conv2d(C, squeeze_channels, 1)
        self.fc2 = nn.Conv2d(squeeze_channels, C, 1)
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
                p_sd
                ):

        super().__init__()

        squeeze_channels = max(1, in_channels//4)
        exp_channels = in_channels * exp
        self.use_connection = in_channels == out_channels and stride == 1
        self.block = nn.Sequential(
            ConvBlock(in_channels, exp_channels, 1, 1) if exp > 1 else nn.Identity(),
            ConvBlock(exp_channels, exp_channels, kernel_size, stride, groups=exp_channels),
            SeBlock(exp_channels, squeeze_channels),
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

        in_channels : int 
            number of input channels, default 3 for image

        out_channels : int
            number of output channels, default 1000 classes for imagenet

        p_sd : float
            stochastic depth probability

        p_drop : float
            dropout probability

        add_pretrain_head : bool 
            defines whether to add pretrained head or new one

    Attributes
    ----------
        add_pretrain_head : bool
            contains add add_pretrain_head parameters

        blocks : nn.ModuleList
            container for all stages with MBConv and FusedMbConv

        stem : ConvBlock
            first convolution block in the network

        final_conv : ConvBlock
            final convolution block before classification head

        avgpool : nn.AdaptiveAvgPool2d
            adaptive average pooling that squeezes all the channels

        head : nn.Sequential
            the final layers of the model with Dropout and classification layer
        
    
    """

    def __init__(
            self, 
            config_name,
            in_channels=3,
            out_channels=1000,
            p_sd=0.2,
            p_drop=0.5,
            add_pretrain_head=True
            ):

        super().__init__()

        self.add_pretrain_head = add_pretrain_head
        config = get_config(config_name)
        self.blocks = nn.ModuleList([])
        total_no_blocks = sum([int(i.split("_")[0][1:]) for i in config]) # number of all blocks
        stage_block_id = 0


        for n, stage_config in enumerate(config):
            stage = [] # list of blocks
            r, k, s, e, i, o, c = stage_config.split("_") # c -> fuse block or se
            r = int(r[1:])

            stage_block_id += 1
            sd_prob = p_sd * float(stage_block_id) / total_no_blocks # adaptive stochastic depth probability which increases with depth

            # Only first MBConv has stride 2 or 1
            if "c" in c:
                stage.append(FusedMBConv(int(i[1:]), int(o[1:]), int(k[1:]), int(s[1:]), int(e[1:]), sd_prob))
            else:
                stage.append(MBConv(int(i[1:]), int(o[1:]), int(k[1:]), int(s[1:]), int(e[1:]), sd_prob))

            if r > 1:
                for _ in range(r-1):
                    stage_block_id += 1
                    sd_prob = p_sd * float(stage_block_id) / total_no_blocks

                    if "c" in c:
                        stage.append(FusedMBConv(int(o[1:]), int(o[1:]), int(k[1:]), 1, int(e[1:]), sd_prob))
                    else:
                        stage.append(MBConv(int(o[1:]), int(o[1:]), int(k[1:]), 1, int(e[1:]), sd_prob))

            if n == 0:
                first_in_channel = int(i[1:])

            if n == len(config) - 1:
                last_out_channel = int(o[1:])

            self.blocks.append(nn.Sequential(*stage))

        self.stem = ConvBlock(
            in_channels=in_channels, 
            out_channels=first_in_channel, 
            kernel_size=3,
            stride=2
        )

        self.final_conv = ConvBlock(last_out_channel, 1280, 1, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.head = nn.Sequential(
            nn.Dropout(p=p_drop, inplace=True),
            nn.Linear(1280, out_channels)
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
        for block in self.blocks:
            x = block(x)

        x = self.final_conv(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x


# sanity check
if __name__ == "__main__":
    models = ["Base", "S", "M", "L", "XL"]
    for model in models[2:3]:
        effnet = EfficientNetV2(model)
        print(" ### MODEL ",model, "###")
        for i, (k,v) in enumerate(effnet.state_dict().items()):
            if i < 500:
                print(k, " -> ", v.shape)
        img = torch.rand((1, 3, 224, 224))
        print(effnet(img).shape)
        