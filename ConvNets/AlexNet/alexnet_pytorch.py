import torch 
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, in_channels=3, classes=1000):
        super().__init__()
        self.c1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0)
        self.c2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.c3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.c4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.c5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(6*6*256, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, classes)

        self.localnorm = nn.LocalResponseNorm(size=5)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

        self.init_weight()

    def forward(self, x):
        # x shape: [batch, 3, 227, 227]
        x = self.relu(self.c1(x))
        # x shape: [batch, 96, 55, 55]
        x = self.maxpool(self.localnorm(x))
        # x shape: [batch, 96, 27, 27]
        x = self.relu(self.c2(x))
        # x shape: [batch, 256, 27, 27]
        x = self.maxpool(self.localnorm(x))
        # x shape: [batch, 256, 13, 13]
        x = self.relu(self.c3(x))
        # x shape: [batch, 384, 13, 13]
        x = self.relu(self.c4(x))
        # x shape: [batch, 384, 13, 13]
        x = self.maxpool(self.relu(self.c5(x)))
        # x shape: [batch, 256, 6, 6]
        x = torch.flatten(x,1)
        # x shape: [batch, 256*6*6]
        x = self.relu(self.dropout(self.fc1(x)))
        # x shape: [batch, 4096]
        x = self.relu(self.dropout(self.fc2(x)))
        # x shape: [batch, 4096]
        x = self.fc3(x)
        # x shape: [batch, classes]
        return x
    
    def init_weight(self):
        bias = [1,3,4,5,6,7]
        for i,layer in enumerate(self.modules()):
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                if i in bias:
                    nn.init.constant_(layer.bias, 1)
                else:
                    nn.init.constant_(layer.bias, 0)
                
                nn.init.normal_(layer.weight, mean=0, std=0.01)

if __name__ == "__main__":
    net = AlexNet()
    print(net(torch.rand(1,3,227,227)).shape)