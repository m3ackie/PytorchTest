import torch
import torch.nn as nn
import torch.nn.functional as f


class My2DCNN(nn.Module):

    def __init__(self, in_channels):
        super(My2DCNN, self).__init__()
        self.maxpooling = nn.MaxPool2d(kernel_size=2, stride=2)
        # input 3*64*64 三切片
        # output 64*32*32
        self.layer_1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            self.maxpooling
        )
        # input 64*32*32
        # output 128*16*16
        self.layer_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            self.maxpooling
        )
        # input 128*16*16
        # output 256*8*8
        self.layer_3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            self.maxpooling
        )
        self.fc = nn.Sequential(
            nn.Linear(256*8*8, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 2)
        )

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    input = torch.randn(4, 3, 64, 64)
    model =My2DCNN(3)
    y = model(input)
    print(y)
