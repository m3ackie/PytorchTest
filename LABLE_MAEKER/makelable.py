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


def maketxtfile(imagepath, labelpath):
    if os.path.exists('../lymph_dataset/label.txt'):
        os.remove('../lymph_dataset/label.txt')
    if os.path.exists('../lymph_dataset/mylable.txt'):
        os.remove('../lymph_dataset/mylable.txt')
    if os.path.exists('../lymph_dataset/myABDneglabel.txt'):
        os.remove('../lymph_dataset/myABDneglabel.txt')
    if os.path.exists('../lymph_dataset/myABDposlabel.txt'):
        os.remove('../lymph_dataset/myABDposlabel.txt')
    if os.path.exists('../lymph_dataset/myMEDneglabel.txt'):
        os.remove('../lymph_dataset/myMEDneglabel.txt')
    if os.path.exists('../lymph_dataset/myMEDposlabel.txt'):
        os.remove('../lymph_dataset/myMEDposlabel.txt')
    if os.path.exists('../lymph_dataset/trainpath.txt'):
        os.remove('../lymph_dataset/trainpath.txt')
    if os.path.exists('../lymph_dataset/testpath.txt'):
        os.remove('../lymph_dataset/testpath.txt')
    for i in np.arange(len(imagepath)):
        row = imagepath[i] + '  ' + labelpath[i] + '\n'
        with open('../lymph_dataset/label.txt', 'a') as f:
            f.write(row)


def make_traintest_label(neglabel, poslabel, trainpath, testpath):
    if os.path.exists(trainpath):
        os.remove(trainpath)
    if os.path.exists(testpath):
        os.remove(testpath)
    negcount = len(open(neglabel, 'rU').readlines())
    poscount = len(open(poslabel, 'rU').readlines())
    i = 0
    j = 0
    with open(neglabel, 'r') as fopen:
        for line in fopen:
            if i % 9 > 0:
                with open(trainpath, 'a') as topen:
                    topen.write(line)
            else:
                with open(testpath, 'a') as teopen:
                    teopen.write(line)
            i = i + 1
    with open(poslabel, 'r') as fopen:
        for line in fopen:
            if j % 9 > 0:
                with open(trainpath, 'a') as topen:
                    topen.write(line)
            else:
                with open(testpath, 'a') as teopen:
                    teopen.write(line)
            j = j + 1
    print(negcount, poscount)


def makelabel(txt, myABDneglabel, myABDposlabel, myMEDneglabel, myMEDposlabel):
    lstFilesDCM = []
    lablefile = []
    dcm_txt = []
    m = 0
    with open(txt, 'r') as fopen:
        for line in fopen:
            line = line.strip('\n')
            line = line.split('  ')
            datapath, lablepath = line[0], line[1]
            if "ABD" in datapath:
                myneglabel = myABDneglabel
                myposlabel = myABDposlabel
            elif "MED" in datapath:
                myneglabel = myMEDneglabel
                myposlabel = myMEDposlabel
                # print(datapath, lablepath)
            allnegLymphPos = []
            allposLymphPos = []
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
                                if i % 4 == 0:
                                    allnegLymphPos.append(anegLymphPos)
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
                                allposLymphPos.append(aposLymphPos)

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
                with open(myneglabel, 'a') as mfopen:
                    for _ in allnegLymphPos:
                        mfopen.write(datapath + '@')
                        for i in _:
                            if _.index(i) == 2:
                                mfopen.write(i + '@')
                            else:
                                mfopen.write(i + '  ')
                        mfopen.write('\n')
                with open(myposlabel, 'a') as mfopen:
                    for _ in allposLymphPos:
                        mfopen.write(datapath + '@')
                        for i in _:
                            if _.index(i) == 2:
                                mfopen.write(i + '@')
                            else:
                                mfopen.write(i + '  ')
                        mfopen.write('\n')


if __name__ == "__main__":

    dataPath = r"../lymph_dataset/CT_Lymph_Nodes"
    lablePath = r"../lymph_dataset/MED_ABD_LYMPH_ANNOTATIONS"
    candidatePath = r"../lymph_dataset/MED_ABD_LYMPH_CANDIDATES"
    datalabletxt = '../lymph_dataset/label.txt'
    mylable = '../lymph_dataset/mylabel.txt'
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
    makelabel(datalabletxt, mylabel)