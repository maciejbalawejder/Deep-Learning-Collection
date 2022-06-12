import torch 
import torch.nn as nn

# ConvBlock
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, bias=False):
        super().__init__()
        self.c = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        return self.bn(self.c(x))

# Squeeze-and-Excitation Block
class SeBlock(nn.Module):
    def __init__(self, in_channels, r):
        super().__init__()
        C = in_channels
        self.globpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(C, C//r, bias=False)
        self.fc2 = nn.Linear(C//r, C, bias=False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: [N, C, H, W]
        f = self.globpool(x)
        f = torch.flatten(f,1)
        f = self.relu(self.fc1(f))
        f = self.sigmoid(self.fc2(f))
        f = f[:,:,None,None]
        # f shape: [N, C, 1, 1]

        scale = x * f
        return scale

# Bottleneck ResNet ResidualBlock + Squeeze-and-Excitation
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, r, first=False):
        super().__init__()
        res_channels = in_channels // 4
        stride = 1

        self.projection = in_channels!=out_channels
        if self.projection:
            self.p = ConvBlock(in_channels, out_channels, 1, 2, 0)
            stride = 2
            res_channels = in_channels // 2

        if first:
            self.p = ConvBlock(in_channels, out_channels, 1, 1, 0)
            stride = 1
            res_channels = in_channels


        self.c1 = ConvBlock(in_channels, res_channels, 1, 1, 0) 
        self.c2 = ConvBlock(res_channels, res_channels, 3, stride, 1)
        self.c3 = ConvBlock(res_channels, out_channels, 1, 1, 0)
        self.relu = nn.ReLU()
        self.seblock = SeBlock(out_channels, r=r)


    def forward(self, x):
        f = self.relu(self.c1(x))
        f = self.relu(self.c2(f))
        f = self.c3(f)
        f = self.seblock(f)

        if self.projection:
            x = self.p(x)

        h = self.relu(torch.add(f, x))
        return h
         
# Bottleneck ResNeXt ResidualBlock + Squeeze-and-Excitation
class ResNeXtBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, r, first=False, cardinality=32):
        super().__init__()
        self.C = cardinality
        self.downsample = stride==2 or first
        res_channels = out_channels // 2
        self.c1 = ConvBlock(in_channels, res_channels, 1, 1, 0)
        self.c2 = ConvBlock(res_channels, res_channels, 3, stride, 1, self.C)
        self.c3 = ConvBlock(res_channels, out_channels, 1, 1, 0)
        self.seblock = SeBlock(out_channels, r=r)

        self.relu = nn.ReLU()

        if self.downsample:
            self.p = ConvBlock(in_channels, out_channels, 1, stride, 0)


    def forward(self, x):
        f = self.relu(self.c1(x))
        f = self.relu(self.c2(f))
        f = self.c3(f)
        f = self.seblock(f)

        if self.downsample:
            x = self.p(x)

        h = self.relu(torch.add(f,x))

        return h


# SE-ResNet
class SEResNet(nn.Module):
    def __init__(
        self, 
        config_name : int, 
        in_channels : int = 3, 
        classes : int = 1000,
        r : int = 16
        ):
        super().__init__()

        configurations = {
            50 : [3, 4, 6, 3],
            101 : [3, 4, 23, 3],
            152 : [3, 8, 36, 3]
        }    

        no_blocks = configurations[config_name]
        
        out_features = [256, 512, 1024, 2048]
        self.blocks = nn.ModuleList([ResNetBlock(64, 256, r=r, first=True)])

        for i in range(len(out_features)):
            if i > 0:
                self.blocks.append(ResNetBlock(out_features[i-1], out_features[i], r=r))
            for _ in range(no_blocks[i]-1):
                self.blocks.append(ResNetBlock(out_features[i], out_features[i], r=r))
        
        self.conv1 = ConvBlock(in_channels, 64, 7, 2, 3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(2048, classes)

        self.relu = nn.ReLU()

        self.init_weight()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        for block in self.blocks:
            x = block(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def init_weight(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)


# SE-ResNeXt
class SEResNeXt(nn.Module):
    def __init__(
        self, 
        config_name : int, 
        in_channels : int = 3, 
        classes : int = 1000,
        C : int = 32, # cardinality
        r : int = 16
        ):
        super().__init__()

        configurations = {
            50 : [3, 4, 6, 3],
            101 : [3, 4, 23, 3],
            152 : [3, 8, 36, 3]
        }    

        no_blocks = configurations[config_name]

        out_features = [256, 512, 1024, 2048]
        self.blocks = nn.ModuleList([ResNeXtBlock(64, 256, 1, first=True, r=r, cardinality=C)])

        for i in range(len(out_features)):
            if i > 0:
                self.blocks.append(ResNeXtBlock(out_features[i-1], out_features[i], 2, cardinality=C, r=r))
            for _ in range(no_blocks[i]-1):
                self.blocks.append(ResNeXtBlock(out_features[i], out_features[i], 1, cardinality=C, r=r))
        
        self.conv1 = ConvBlock(in_channels, 64, 7, 2, 3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(2048, classes)

        self.relu = nn.ReLU()

        self.init_weight()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        for block in self.blocks:
            x = block(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def init_weight(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)


if __name__ == "__main__":
    config_name = 50
    se_resnext = SEResNeXt(config_name)
    image = torch.rand(1, 3, 224, 224)
    print(se_resnext(image).shape)

    se_resnet = SEResNet(config_name)
    print(se_resnet(image).shape)