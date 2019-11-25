import cv2
import numpy as np
import matplotlib.pylab as plt
import pydicom as dicom
import os
import glob

import pandas as pd
import scipy.ndimage

# from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
# from tqdm import tqdm
import torch.nn.functional as F
import torchvision.models as models


def maketxtfile(imagepath, lablepath):
    lstFilesDCM = []
    lstFilesLAB = []
    # for dirName, subdirList, fileList in os.walk(lablepath):
    #     for filename in fileList:
    #         if ".txt" in filename.lower():
    #             lstFilesLAB.append(os.path.join(dirName, filename))
    # print(len(lablepath), len(imagepath))
    if os.path.exists('../lymph_dataset/lable.txt'):
        os.remove('../lymph_dataset/lable.txt')
    if os.path.exists('../lymph_dataset/mylable.txt'):
        os.remove('../lymph_dataset/mylable.txt')
    for i in np.arange(len(imagepath)):
        row = imagepath[i] + '  ' + lablepath[i] + '\n'
        with open('../lymph_dataset/lable.txt', 'a') as f:
            f.write(row)


def make_traintest_lable(totallable, trainpath, testpath):
    if os.path.exists(trainpath):
        os.remove(trainpath)
    if os.path.exists(testpath):
        os.remove(testpath)
    datacount = len(open(totallable, 'rU').readlines())
    i = 0
    with open(totallable, 'r') as fopen:
        for line in fopen:
            if i < int(datacount * 0.8):
                with open(trainpath, 'a') as topen:
                    topen.write(line)
            else:
                with open(testpath, 'a') as teopen:
                    teopen.write(line)
            i = i + 1
    print(datacount)


def makelable(txt, mylable):
    lstFilesDCM = []
    lablefile = []
    dcm_txt = []
    m = 0
    with open(txt, 'r') as fopen:
        for line in fopen:
            line = line.strip('\n')
            line = line.split('  ')
            datapath, lablepath = line[0], line[1]
            # print(datapath, lablepath)
            allLymphPos = []
            allLymphSize = []
            newlable = []
            i = 0

            for dirName, subdirList, fileList in os.walk(lablepath):
                for filename in fileList:
                    if "negCADe_physicalPoints.txt" in filename:
                        lable = "0"
                        lablefile.append(os.path.join(dirName, filename))
                        with open(os.path.join(dirName, filename), 'r') as mfopen:
                            for line_ in mfopen:
                                line_ = line_.strip('\n')
                                line_ = line_.split(' ')
                                sagittal, coronal, axial = line_[0], line_[1], line_[2]
                                anegLymphPos = [sagittal, coronal, axial, lable]
                                if i % 3 == 0:
                                    allLymphPos.append(anegLymphPos)
                                    m += 1
                                i += 1

                    if "posCADe_physicalPoints.txt" in filename:
                        lable = "1"
                        lablefile.append(os.path.join(dirName, filename))
                        with open(os.path.join(dirName, filename), 'r') as mfopen:
                            for line_ in mfopen:
                                line_ = line_.strip('\n')
                                line_ = line_.split(' ')
                                sagittal, coronal, axial = line_[0], line_[1], line_[2]
                                aposLymphPos = [sagittal, coronal, axial, lable]
                                allLymphPos.append(aposLymphPos)

                    if "sizes.txt" in filename.lower():
                        with open(os.path.join(dirName, filename), 'r') as mfopen:
                            for line_ in mfopen:
                                line_ = line_.strip('\n')
                                line_ = line_.split(' ')
                                if len(line_) == 2:
                                    long = line_[1]
                                    aLymphSize = [long]
                                    allLymphSize.append(aLymphSize)
                                elif len(line_) == 1:
                                    long = line_[0]
                                    aLymphSize = [long]
                                    allLymphSize.append(aLymphSize)

                    # for _ in np.arange(len(allLymphPos)):
                    #     newlable.append(allLymphPos[_])
                    # print(newlable)
                with open(mylable, 'a') as mfopen:
                    for _ in allLymphPos:
                        mfopen.write(datapath + '@')
                        for i in _:
                            if _.index(i) == 2:
                                mfopen.write(i + '@')
                            else:
                                mfopen.write(i + '  ')
                        mfopen.write('\n')

            # for dirName, subdirList, fileList in os.walk(datapath):
            #     for filename in fileList:
            #         if ".dcm" in filename.lower():
            #             # self.lstFilesDCM.append(os.path.join(dirName, filename))
            #             with open(mylable, 'a') as mfopen:
            #                 mfopen.write(os.path.join(dirName, filename) + '  ')
            #                 # x = [i[2] for i in newlable]
            #                 is_exist = 0
            #                 for _ in np.arange(len(newlable)):
            #                     if filename.find(newlable[_][2]) != -1:
            #                         is_exist = is_exist + 1
            #                 mfopen.write(str(is_exist) + '@' + '\n')
        print(m)
        # print(filename)
    # print(lstFilesDCM)


if __name__ == "__main__":
    dataPath = r"../lymph_dataset/CT Lymph Nodes"
    lablePath = r"../lymph_dataset/MED_ABD_LYMPH_ANNOTATIONS"
    candidatePath = r"../lymph_dataset/MED_ABD_LYMPH_CANDIDATES"
    datalabletxt = '../lymph_dataset/lable.txt'
    mylable = '../lymph_dataset/mylable.txt'
    trainpath = r'../lymph_dataset/trainpath.txt'
    testpath = r'../lymph_dataset/testpath.txt'

    patientDirlist = os.listdir(dataPath)
    lableDirlist = os.listdir(lablePath)
    candidateDirlist = os.listdir(candidatePath)
    # print(patientDirlist)
    # 获取指定病人DICOM数据集
    allPPathList = [dataPath + "/" + patientNum for patientNum in patientDirlist]
    candidatePathlist = [candidatePath + "/" + lableNum for lableNum in candidateDirlist]
    maketxtfile(allPPathList, candidatePathlist)
    makelable(datalabletxt, mylable)
