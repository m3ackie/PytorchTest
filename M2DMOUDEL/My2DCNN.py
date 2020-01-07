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


class Conv2d_simple(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, flag_bn, **kwargs):
        super(Conv2d_simple, self).__init__()
        self.flag = flag_bn
        self.pad = int((ksize - 1)/2)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, ksize), padding=(0, self.pad), bias=False)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=(ksize, 1), padding=(self.pad, 0), bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.PReLU(out_channels)
    def forward(self, x):
        out = self.conv_1(x)
        out = self.conv_2(out)

        if self.flag:
            out = self.bn(out)
        out = self.act(out)
        return out


class MyGoogleNet(nn.Module):
    def __init__(self, num_classes):
        super(MyGoogleNet, self).__init__()
        self.layer_1_1 = nn.Conv2d(3, 32, 1, padding=0, bias=False)
        self.layer_1_3 = Conv2d_simple(1, 32, 3, True)
        self.layer_1_7 = Conv2d_simple(1, 32, 7, True)

        self.layer_2_1 = nn.Conv2d(97, 128, 1, padding=0, bias=False)
        self.layer_2_3 = Conv2d_simple(97, 128, 3, True)
        self.layer_2_7 = Conv2d_simple(97, 128, 7, True)

        self.layer_3_1 = nn.Conv3d(481, 128, 1, padding=0, bias=False)
        self.layer_3_3 = Conv2d_simple(481, 128, 3, False)
        self.layer_3_7 = Conv2d_simple(481, 128, 7, False)

        self.layer_4_1 = nn.Conv3d(865, 128, 1, padding=0, bias=False)
        self.layer_4_3 = Conv2d_simple(865, 128, 3, False)

        self.layer_5_1 = nn.Conv3d(1121, 64, 1, padding=0, bias=False)
        self.layer_5_3 = Conv2d_simple(1121, 64, 3, False)

        self.layer_6_3 = Conv2d_simple(1249, 128, 3, False)

        self.layer_7_3 = Conv2d_simple(128, 128, 3, False)

        self.layer_8_3 = Conv2d_simple(128, 128, 3, False)

        self.drop = nn.Dropout2d(p=0.25, inplace=False)

        self.out = nn.Linear(128, num_classes)
        self.pool_simple = nn.AvgPool2d((1, 2, 2))
        self.pool_full = nn.AvgPool2d((2, 2))

    def forward(self, input):
        l1_1 = self.layer_1_1(input)
        l1_3 = self.layer_1_3(input)
        l1_7 = self.layer_1_7(input)
        l1_out = torch.cat((l1_1, l1_3, l1_7, input), 1)
        # 97 x 64 x 64
        l1_out = self.pool_full(l1_out)
        # 97 x 32 x 32
        l2_1 = self.layer_2_1(l1_out)
        l2_3 = self.layer_2_3(l1_out)
        l2_7 = self.layer_2_7(l1_out)
        l2_out = torch.cat((l2_1, l2_3, l2_7, l1_out), 1)
        # 481 x 32 x 32
        l2_out = self.pool_full(l2_out)
        # 481 x 16 x 16
        l3_1 = self.layer_3_1(l2_out)
        l3_3 = self.layer_3_3(l2_out)
        l3_7 = self.layer_3_7(l2_out)
        l3_out = torch.cat((l3_1, l3_3, l3_7, l2_out), 1)
        # 865 x 32 x 32
        # l3_out = self.pool_full(l3_out)
        # 865 x 32 x 32
        l4_1 = self.layer_4_1(l3_out)
        l4_3 = self.layer_4_3(l3_out)
        l4_out = torch.cat((l4_1, l4_3, l3_out), 1)
        # 1121 x 16 x 16
        l4_out = self.pool_full(l4_out)
        # 1121 x 8 x 8
        l5_1 = self.layer_5_1(l4_out)
        l5_3 = self.layer_5_3(l4_out)
        l5_out = torch.cat((l5_1, l5_3, l4_out), 1)
        # 1249 x 8 x 8
        l6 = self.layer_6_3(l5_out)
        # 128 x  8 x 8
        l6 = self.pool_full(l6)
        # 128 x 4 x 4
        l7 = self.layer_7_3(l6)
        l7 = self.pool_full(l7)
        # 128 x 2 x 2
        l8 = self.layer_8_3(l7)
        l8 = self.pool_full(l8)
        # 128 x 1 x 1
        l8 = l8.view(input.shape[0], -1)

        out = self.out(l8)
        return out
if __name__ == "__main__":
    input = torch.randn(4, 3, 64, 64)
    model =My2DCNN(3)
    y = model(input)
    print(y)
