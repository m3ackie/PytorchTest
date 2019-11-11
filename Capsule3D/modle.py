import cv2
import numpy as np
import matplotlib.pylab as plt
import pydicom as dicom
import os
import glob

import pandas as pd
import scipy.ndimage

import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

import torch.nn.functional as F
import torchvision.models as models

NUM_ROUTING_ITERATIONS = 3
NUM_CLASSES = 2


def softmax(input, dim=1):
    transposed_input = input.transpose(dim, len(input.size()) - 1)
    softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=-1)
    return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)


def augmentation(x, max_shift=2):
    _, _, height, width = x.size()

    h_shift, w_shift = np.random.randint(-max_shift, max_shift + 1, size=2)
    source_height_slice = slice(max(0, h_shift), h_shift + height)
    source_width_slice = slice(max(0, w_shift), w_shift + width)
    target_height_slice = slice(max(0, -h_shift), -h_shift + height)
    target_width_slice = slice(max(0, -w_shift), -w_shift + width)

    shifted_image = torch.zeros(*x.size())
    shifted_image[:, :, source_height_slice, source_width_slice] = x[:, :, target_height_slice, target_width_slice]
    return shifted_image.float()


class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, kernel_size=None, stride=None,
                 num_iterations=NUM_ROUTING_ITERATIONS):
        """

        :param num_capsules: how long capsule we wanna get
        :param num_route_nodes: -1 means dont need routing, else the param is the summation of features a capsule has
        eg: 32 * 6 * 6
        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param stride:
        :param num_iterations:
        """
        super(CapsuleLayer, self).__init__()

        self.num_route_nodes = num_route_nodes
        self.num_iterations = num_iterations
        self.num_capsules = num_capsules

        if num_route_nodes != -1:
            # capsule to capsule
            # 输出胶囊数，输入胶囊数，输入胶囊维度，输出胶囊维度
            self.route_weights = nn.Parameter(
                0.01 * torch.randn(num_capsules, num_route_nodes, in_channels, out_channels))
        else:
            # conv to capsule
            self.capsules = nn.ModuleList(
                [nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0) for _ in
                 range(num_capsules)])

    def squash(self, tensor, dim=-1):
        # 张量的范数||Sj||
        squared_norm = torch.norm(tensor, p=2, dim=dim, keepdim=True)
        # squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        # 1e-8 防止分母为0
        scale = squared_norm ** 2 / (1 + squared_norm ** 2) / (squared_norm + 1e-8)
        return scale * tensor

    def forward(self, x):
        if self.num_route_nodes != -1:
            # priors : [batch, out_num_capsules, in_num_capsules(32 * 10 * 10 * 10), 1, out_dim]
            priors = x[:, None, :, None, :] @ self.route_weights[None, :, :, :, :]
            priors_detach = priors.detach()
            # logits : (4, 2, 32000, 1, 16)
            logits = Variable(torch.zeros(*priors.size())).cuda()
            for i in range(self.num_iterations):
                # softmax 生成概率[4,2,32000,1,16]
                probs = softmax(logits, dim=1)
                if i == self.num_iterations - 1:
                    # 计算出每个胶囊与对应权值的积 输出胶囊 outputs : (4, 2, 1, 1, 16)
                    outputs = self.squash((probs * priors).sum(dim=2, keepdim=True))
                # 更新权重
                else:
                    outputs = self.squash((probs * priors_detach).sum(dim=2, keepdim=True))
                    # delta_logits : (4, 2, 32000, 1, 1)
                    delta_logits = (priors_detach * outputs).sum(dim=-1, keepdim=True)
                    logits = logits + delta_logits
        else:
            # 循环生成列表（长度为一个胶囊的维度，即num_capsules）capsule(x) eg: batchsize*outchannels*D*H*W 2*32*(10*10*10)*1
            outputs = [capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules]
            # cat把output合并(reshape)为一个胶囊  outputs 2*32*(10*10*10)*8D(num_capsule)
            outputs = torch.cat(outputs, dim=-1)
            # 挤压squash操作
            outputs = self.squash(outputs)
            # print('8d capsule')
            # print(outputs)
        # print(self.num_route_nodes, outputs.shape)

        return outputs


class CapsuleNet_3D(nn.Module):
    def __init__(self):
        super(CapsuleNet_3D, self).__init__()
        self.features = nn.Sequential(
            # 1 low features conv layer
            nn.Conv3d(in_channels=1, out_channels=128, kernel_size=3, stride=1),
            # 2 primary capsule layer[1](feature to capsule)
            CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=128, out_channels=16, kernel_size=9, stride=2),
            # 3 classification capsule layer(capsule to capsule)
            CapsuleLayer(num_capsules=NUM_CLASSES, num_route_nodes=16 * 10 * 10 * 10, in_channels=8, out_channels=16)
        )
        # 4 full conv decoder
        self.decoder = nn.Sequential(
            nn.Linear(16 * NUM_CLASSES, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 10240),
            nn.ReLU(inplace=True),
            nn.Linear(10240, 27000),
            nn.Sigmoid()
        )

    def forward(self, x, y=None):
        x = F.relu(self.features[0](x), inplace=True)

        x = self.features[1](x)
        # print(x.shape)
        x = self.features[2](x).squeeze()
        print('xxxxxxxxxxxxxx')
        print(x.shape)
        # capsule length & classes is y_pre
        classes = x.norm(dim=-1)  # (x ** 2).sum(dim=-1) ** 0.5
        print(classes)
        # classes = F.softmax(classes, dim=-1)

        if y is None:
            # In all batches, get the most active capsule.
            _, max_length_indices = classes.max(dim=1)
            # y = Variable(torch.eye(NUM_CLASSES)).cuda().index_select(dim=0, index=max_length_indices.data)
            y = Variable(torch.zeros(classes.size()).scatter_(1, max_length_indices.view(-1, 1).cpu().data, 1.).cuda())

        reconstructions = self.decoder((x * y[:, :, None]).view(x.size(0), -1))

        return classes, reconstructions


class CapsuleLoss(nn.Module):
    def __init__(self):
        super(CapsuleLoss, self).__init__()
        self.reconstruction_loss = nn.MSELoss(size_average=True)

    def forward(self, images, labels, classes, reconstructions):
        margin_loss = labels * torch.clamp(0.9 - classes, min=0.) ** 2 + \
                      0.5 * (1 - labels) * torch.clamp(classes - 0.1, min=0.) ** 2
        margin_loss = margin_loss.sum(dim=1).mean()

        assert torch.numel(images) == torch.numel(reconstructions)
        images = images.view(reconstructions.size()[0], -1)
        reconstruction_loss = self.reconstruction_loss(reconstructions, images)
        print(margin_loss)
        print(reconstruction_loss)

        return margin_loss + 0.0005 * reconstruction_loss * 27000

# if __name__ == "__main__":
#     test = CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=128, out_channels=32, kernel_size=9, stride=2)
#     torch.randn(28, 28, 28, 128)
