import cv2
import numpy as np

# from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import torch
import torch.nn as nn
import  torch.nn.functional as f
import sklearn.metrics as skl
from torch.autograd import Variable
BATCH_SIZE = 2
learning_rate = 1e-5
num_epoches = 10
NUM_ROUTING_ITERATIONS = 3
a = torch.tensor([[1, 0],
                  [1, 1],
                  [0, 1]])
b = torch.tensor([[0, 1],
                  [0, 0],
                  [0, 1]])
targets = ['Fasle', 'True']
a = a.view(-1).numpy()
b = b.view(-1).numpy()
print(skl.classification_report(a, b, target_names=targets))
print(skl.confusion_matrix(a, b))
# print(torch.cat((a.view(-1), b.view(-1))))
# a = a.view((1, -1))
# b = b.view((1, -1))
# print(torch.cat((a, b), dim=1))


