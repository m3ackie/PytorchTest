import numpy as np
import torch
import torch.nn as nn

def conv_block(in_channel, out_channel):
    layer = nn.Sequential(
        nn.BatchNorm3d(in_channel),
        nn.ReLU(True),
        nn.Conv3d(in_channel, out_channel, 3, padding=1, bias=False)
    )
    return layer

class dense_block(nn.Module):
    def __init__(self, in_channel, growth_rate, num_layers):
        super(dense_block, self).__init__()
        block = []
        channel = in_channel
        for i in range(num_layers):
            block.append(conv_block(channel, growth_rate))
            channel += growth_rate
        self.net = nn.Sequential(*block)
    def forward(self, x):
        for layer in self.net:
            out = layer(x)
            x = torch.cat((out, x), dim=1)
        return x


def transition(in_channel, out_channel):
    trans_layer = nn.Sequential(
        nn.BatchNorm3d(in_channel),
        nn.ReLU(True),
        nn.Conv3d(in_channel, out_channel, 1),
        nn.AvgPool3d(2, 2)
    )
    return trans_layer


class densenet(nn.Module):
    def __init__(self, in_channel, num_classes, growth_rate=32, block_layers=[6, 12, 24, 16]):
        super(densenet, self).__init__()
        # 初始模块
        self.block1 = nn.Sequential(
            nn.Conv3d(in_channel, 64, 7, 2, 3),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            nn.MaxPool3d(3, 2, padding=1)
        )
        # dense_block模块
        channels = 64
        block = []
        for i, layers in enumerate(block_layers):
            block.append(dense_block(channels, growth_rate, layers))
            channels += layers * growth_rate
            if i != len(block_layers) - 1:
                block.append(transition(channels, channels // 2))
                channels = channels // 2
        self.block2 = nn.Sequential(*block)
        # 其他层
        self.block2.add_module('bn', nn.BatchNorm3d(channels))
        self.block2.add_module('relu', nn.ReLU(True))
        self.block2.add_module('avg_pool', nn.AvgPool3d(3))
        self.classifier = nn.Linear(channels, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x

