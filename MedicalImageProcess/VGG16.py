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

import LABLE_MAEKER.makelable as ml
import Capsule3D.modle as md

BATCH_SIZE = 2
learning_rate = 1e-3
num_epoches = 10
NUM_ROUTING_ITERATIONS = 3


class MyDataset(Dataset):
    """
    root: 图像根目录
    augment:是否需要图像增强
    """
    def __init__(self, mylable, augment=None):

        self.lable = []
        self.datapath = []
        self.lymphpos = []
        self.lymphlable =[]

        with open(mylable, 'r') as fopen:
            for line in fopen:
                line = line.strip('\n')
                line = line.split('@')
                datapath, lymphpos, lymphlable = line[0], line[1], line[2]
                self.datapath.append(datapath)
                # position
                lymphpos = lymphpos.split('  ')
                self.lymphpos.append(lymphpos)
                # lable
                lymphlable = lymphlable.strip(' ')
                self.lymphlable.append(lymphlable)

                # print(datapath, lablepath)
        self.augment = augment

    def __getitem__(self, index):
        # slices is 3d d*512*512
        itlable = np.asarray(int(self.lymphlable[index])).astype(np.int64)
        itpos = np.asarray(self.lymphpos[index]).astype(np.float64)
        self.lstFilesDCM = []
        for dirName, subdirList, fileList in os.walk(self.datapath[index]):
            for filename in fileList:
                if ".dcm" in filename.lower():
                    self.lstFilesDCM.append(os.path.join(dirName, filename))

        self.lstFilesDCM = np.array(self.lstFilesDCM)

        if self.augment:
            slices = [dicom.read_file(s) for s in self.lstFilesDCM]
            slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))
            slices = self.set_slice_thickness(slices)
            firstslice = slices[0]
            pixelposition = self.getpixelposition(firstslice, itpos)
            print(pixelposition, len(slices))
            if pixelposition[2] >= 15:
                lymphslices = slices[pixelposition[2] - 15:pixelposition[2] + 15]
                if len(slices) - pixelposition[2] < 15:
                    lymphslices = slices[len(slices) - 30:len(slices)]
            else:
                lymphslices = slices[0:30]
            # print(lymphslices)
            slices = self.get_pixels_hu(lymphslices, pixelposition)
            image = self.setDicomCenWid(slices, "abdomen")
            image = image[np.newaxis, :, :, :]
            print(image.shape)
            # dataimage = self.resample(image, firstslice)
        else:
            slices = [dicom.read_file(s) for s in self.lstFilesDCM]
            slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))
            image = np.array(slices, dtype=np.float32)
        return torch.from_numpy(image), torch.from_numpy(itlable), itpos

    def __len__(self):
        # 返回图像的数量
        return len(self.datapath)

    def getpixelposition(self, firstslice, position):
        x, y, z = firstslice.ImagePositionPatient[0], firstslice.ImagePositionPatient[1], \
                  firstslice.ImagePositionPatient[2]
        spacing = firstslice.PixelSpacing[0]
        thickness = firstslice.SliceThickness
        imageposition = [round((float(position[0]) - x) / spacing), round((float(position[1]) - y) / spacing),
                         round((float(position[2]) - z) / thickness)]
        # print(imageposition)

        return imageposition

    def set_slice_thickness(self, slices):
        try:
            slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
        except:
            slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        for s in slices:
            s.SliceThickness = slice_thickness
            # print(s.SliceThickness)
        return slices

    def get_pixels_hu(self, slices, positioninslice):

        image = np.stack([np.pad(s.pixel_array, ((15, 15), (15, 15)), 'linear_ramp')[
                          positioninslice[0]:positioninslice[0]+30,
                          positioninslice[1]: positioninslice[1] + 30] for s in slices])
        print(image.shape)
        image = image.astype(np.int16)
        image[image == -2000] = 0

        for slice_number in range(len(slices)):
            intercept = slices[slice_number].RescaleIntercept
            slope = slices[slice_number].RescaleSlope
            if slope != 1:
                image[slice_number] = slope * image[slice_number].astype(np.float64)
                image[slice_number] = image[slice_number].astype(np.int16)
            image[slice_number] += np.int16(intercept)
        # print(image.shape)
        return np.array(image, dtype=np.float32)

    def get_window_size(self, organ_name):
        if organ_name == 'lung':
            # 肺部 ww 1500-2000 wl -450--600
            center = -500
            width = 2000
        elif organ_name == 'abdomen':
            # 腹部 ww 300-500 wl 30-50
            center = 40
            width = 500
        elif organ_name == 'bone':
            # 骨窗 ww 1000-1500 wl 250-350
            center = 300
            width = 2000
        elif organ_name == 'lymph':
            # 淋巴、软组织 ww 300-500 wl 40-60
            center = 45
            width = 10
        elif organ_name == 'mediastinum':
            # 纵隔 ww 250-350 wl 250-350
            center = 40
            width = 350

        return center, width

    def setDicomCenWid(self, slices, organ_name):
        img = slices
        center, width = self.get_window_size(organ_name)
        min = (2 * center - width) / 2.0 + 0.5
        max = (2 * center + width) / 2.0 + 0.5

        dFactor = 255.0 / (max - min)
        d, h, w = np.shape(img)
        for n in np.arange(d):
            for i in np.arange(h):
                for j in np.arange(w):
                    img[n, i, j] = int((img[n, i, j] - min) * dFactor)

        min_index = img < 0
        img[min_index] = 0
        max_index = img > 255
        img[max_index] = 255

        return img

    def resample(self, image, slice, new_spacing=[1, 1, 1]):
        spacing = map(float, ([slice.SliceThickness] + [slice.PixelSpacing[0], slice.PixelSpacing[1]]))
        spacing = np.array(list(spacing))
        resize_factor = spacing / new_spacing
        new_real_shape = image.shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / image.shape
        new_spacing = spacing / real_resize_factor
        image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
        print(image.shape)

        return image, new_spacing


if __name__ == "__main__":
    dataPath = r"F:\lymph_dataset\CT Lymph Nodes"
    lablePath = r"F:\lymph_dataset\MED_ABD_LYMPH_ANNOTATIONS"
    candidatePath = r"F:\lymph_dataset\MED_ABD_LYMPH_CANDIDATES"
    datalabletxt = 'F:\lymph_dataset\lable.txt'
    mylable = 'F:\lymph_dataset\mylable.txt'

    patientDirlist = os.listdir(dataPath)
    lableDirlist = os.listdir(lablePath)
    candidateDirlist = os.listdir(candidatePath)
    # print(patientDirlist)
    # 获取指定病人DICOM数据集
    allPPathList = [dataPath + "\\" + patientNum for patientNum in patientDirlist]
    candidatePathlist = [candidatePath + "\\" + lableNum for lableNum in candidateDirlist]
    ml.maketxtfile(allPPathList, candidatePathlist)
    ml.makelable(datalabletxt, mylable)
    datacount = len(open(mylable, 'rU').readlines())
    Patientdata = MyDataset(mylable, augment=True)
    train_loader = DataLoader(Patientdata, batch_size=BATCH_SIZE, num_workers=0)

    model = md.CapsuleNet_3D()
    use_gpu = torch.cuda.is_available()

    if use_gpu:
        model = model.cuda()

    criterion = md.CapsuleLoss()
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epoches):
        print('*' * 25, 'epoch{}'.format(epoch + 1), '*' * 25)
        running_loss = 0.0
        running_acc = 0.0
        for i, data in enumerate(train_loader, start=1):
            image, label, pos = data[0], data[1], data[2]
            # label = label.long()
            label = torch.zeros(label.size(0), 2).scatter_(1, label.view(-1, 1), 1.)  # change to one-hot coding
            if use_gpu:
                image = Variable(image).cuda()
                label = Variable(label).cuda()
            else:
                image = Variable(image)
                label = Variable(label)
            # train
            if i < int(datacount/BATCH_SIZE*0.9):
                print(i)
                # set gradients of optimizer to zero
                optimizer.zero_grad()
                # forward
                classes, reconstructions = model(image, label)
                loss = criterion(image, label, classes, reconstructions)
                loss.backward()
                running_loss += loss.data[0] * image.size(0)  # record the batch loss
                optimizer.step()  # update the trainable parameters with computed gradients

            else:
                print(i)
                classes, reconstructions = model(image)
                loss = criterion(image, label, classes, reconstructions)
                print('LOSS : %.4f   Accuracy: %.4f  ' % (loss, classes))




    #
    #

    # if use_gpu:
    #     model = model.cuda()
    #
    #
    # '''定义loss和optimizer'''
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    #
    # '''train'''
    # for epoch in range(num_epoches):
    #     print('*' * 25, 'epoch{}'.format(epoch + 1), '*' * 25)
    #     running_loss = 0.0
    #     running_acc = 0.0
    #     for i, data in tqdm(enumerate(train_loader, start=1)):





