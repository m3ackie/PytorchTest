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
a = torch.tensor([[0.1342, 0.1391],
        [0.1338, 0.0974],
        [0.0778, 0.0631],
        [0.0635, 0.0518]]
                 )
b = torch.tensor(251)
print(b)