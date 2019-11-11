import cv2
import numpy as np
import matplotlib.pylab as plt
import pydicom as dicom
import os
import glob

import pandas as pd
import scipy.ndimage

from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch.nn.functional as F
import torchvision.models as models


def squash(tensor, dim=-1):
    # 张量的范数||Sj||
    squared_norm = torch.norm(tensor, p=2, dim=dim, keepdim=True)
    # squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
    # 1e-8 防止分母为0
    scale = squared_norm ** 2 / (1 + squared_norm ** 2) / (squared_norm + 1e-8)
    return scale * tensor

x = torch.randn(2, 16)
y = torch.randn(4, 2, 1, 1, 16)
z = torch.randn(2, 16)
a = torch.tensor( [ 2.4372e-05,  3.9567e-05,  4.4267e-06, -9.2526e-05,  2.1147e-06,
          -4.1276e-05,  3.8939e-05,  1.0780e-05, -3.6601e-05,  2.0521e-05,
          -5.4929e-05,  9.4780e-06, -5.9110e-05,  1.3187e-05,  7.4908e-05,
          -9.9001e-08]
                 )
# print(squash(z).norm(dim=1))
# print(a.norm(dim=1))
print((a ** 2).sum() ** 0.5)
# print((z * x).shape)
# print(x, z, x+z)
# print(z)
# print(F.softmax(x, dim=2))
# print(((x*y).sum(dim=-1, keepdim=True) + z).shape)
# def softmax(input, dim=1):
#     transposed_input = input.transpose(dim, len(input.size()) - 1)
#     softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=-1)
#     return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)
# x = torch.randn(2, 30, 8)
# y = torch.randn(10, 30, 8, 16)
# x2 = x[None, :, :, None, :]
# # y = y[:, None, :, :, :]
# x1 = x[:, None, :, :, None]
# y1 = torch.randn(10, 30, 16, 8)
# # z = x2 @ y
# z1 = torch.squeeze(torch.matmul(y1, x1), dim=-1)
# print(z1.shape)
# # z = torch.squeeze(z, dim=-2)
# # priors : [out_num_capsules, batch, in_num_capsules(32 * 10 * 10 * 10), 1, out_dim]
# priors = x[None, :, :, None, :] @ y[:, None, :, :, :]
# #
# logits = Variable(torch.zeros(*priors.size())).cuda()
# for i in range(3):
#     probs = softmax(logits, dim=1)
#     print(probs.shape)
#     outputs = self.squash((probs * priors).sum(dim=2, keepdim=True))
#
#     if i != self.num_iterations - 1:
#         delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
#         logits = logits + delta_logits
# print(x1.shape, y1.shape, torch.matmul(y1, x1).shape)
# print(x2.shape, y.shape, z.shape)