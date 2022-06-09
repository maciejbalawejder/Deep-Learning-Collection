import torch
import torch.nn as nn

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU()
        # Different projections
        self.p1 = nn.Sequential(*[nn.Conv2d(in_channels, out_channels[0], kernel_size=1, padding=0, stride=1), self.relu])
        self.p2 = nn.Sequential(*[nn.Conv2d(in_channels, out_channels[1], kernel_size=1, padding=0, stride=1), self.relu, nn.Conv2d(out_channels[1], out_channels[2], kernel_size=3, padding=1, stride=1), self.relu])
        self.p3 = nn.Sequential(*[nn.Conv2d(in_channels, out_channels[3], kernel_size=1, padding=0, stride=1), self.relu, nn.Conv2d(out_channels[3], out_channels[4], kernel_size=5, padding=2, stride=1), self.relu])
        self.p4 = nn.Sequential(*[nn.MaxPool2d(kernel_size=3, padding=1, stride=1), nn.Conv2d(in_channels, out_channels[5], kernel_size=1, padding=0, stride=1)])


    def forward(self, x):
        o1 = self.p1(x)
        o2 = self.p2(x)
        o3 = self.p3(x)
        o4 = self.p4(x)

        return torch.cat((o1,o2,o3,o4), axis=1)

class AuxClassifier(nn.Module):
    def __init__(self, in_channels, classes):
        super().__init__()
        in_features = 4 * 4 * 128
        self.avg_pool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv1x1 = nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.fc1 = nn.Linear(in_features=in_features, out_features=in_features)
        self.fc2 = nn.Linear(in_features=in_features, out_features=classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.7)
    
    def forward(self, x):
        x = self.avg_pool(x)
        x = self.relu(self.conv1x1(x))
        x = torch.flatten(x, 1)
        x = self.relu(self.dropout(self.fc1(x)))
        x = self.fc2(x)
        return x

class GoogLeNet(nn.Module): 
    def __init__(self, in_depth=3, classes=1000):
        super().__init__()

        in_channels = [192, 256, 480, 512, 512, 512, 528, 832, 832, 1024]
        feature_maps = [[64, 96, 128, 16, 32, 32],
                        [128, 128, 192, 32, 96, 64],
                        [192, 96, 208, 16, 48, 64],
                        [160, 112, 224, 24, 64, 64],
                        [128, 128, 256, 24, 64, 64],
                        [112, 144, 288, 32, 64, 64],
                        [256, 160, 320, 32, 128, 128],
                        [256, 160, 320, 32, 128, 128],
                        [384, 192, 384, 48, 128, 128]
                    ]
    
        self.AuxClass1 = AuxClassifier(512, classes)
        self.AuxClass2 = AuxClassifier(528,classes)
        self.Blocks = nn.ModuleList([InceptionBlock(in_channels[i], feature_maps[i]) for i in range(len(feature_maps))])
        
        # Rest of the model
        self.Conv7k = nn.Conv2d(in_channels=in_depth, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.Conv1k = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.Conv3k = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.MaxPool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.LocalNorm = nn.LocalResponseNorm(size=5)
        self.AvgPool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.FC = nn.Linear(1024, classes)
        self.Dropout = nn.Dropout(0.4)
        self.relu = nn.ReLU()

        #print(next(iter(self.modules())))
    

    def forward(self, x):
        outputs = []
        # x shape : [batch, 3, 224, 224]
        x = self.relu(self.Conv7k(x))
        x = self.LocalNorm(self.MaxPool(x))
        x = self.relu(self.Conv1k(x))
        x = self.LocalNorm(self.relu(self.Conv3k(x)))
        x = self.MaxPool(x)
        for i, block in enumerate(self.Blocks):
            if i == 2 or i==7:
                x = self.MaxPool(x)
            elif i == 3:
                outputs.append(self.AuxClass1(x))
            elif i == 6:
                outputs.append(self.AuxClass2(x))

            x = block(x)
        
        x = self.Dropout(self.AvgPool(x))
        x = torch.flatten(x,1)
        x = self.FC(x)
        outputs.append(x)
        return outputs

if __name__ == "__main__":
    Inception = GoogLeNet()
    outs = Inception(torch.rand(1, 3, 224, 224))
    for out in outs:
        print(out.shape)