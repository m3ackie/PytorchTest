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


def load_scan(path):

    lstFilesDCM = []
    for dirName, subdirList, fileList in os.walk(path):
        for filename in fileList:
            if ".dcm" in filename.lower():
                lstFilesDCM.append(os.path.join(dirName, filename))
                # print(filename)
    # print(lstFilesDCM)
    slices = [dicom.read_file(s) for s in lstFilesDCM]
    slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))


    return slices


# Load the scans in given folder path
def set_slice_thickness(slices):

    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
    for s in slices:
        s.SliceThickness = slice_thickness
        # print(s.SliceThickness)
    return slices


def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)
    image[image == -2000] = 0

    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)


def get_window_size(organ_name):
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


def setDicomCenWid(slices, organ_name):
    img = slices
    center, width = get_window_size(organ_name)
    min = (2 * center - width)/2.0 + 0.5
    max = (2 * center + width)/2.0 + 0.5

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


def resample(image, slice, new_spacing=[1, 1, 1]):
    spacing = map(float, ([slice.SliceThickness]+[slice.PixelSpacing[0], slice.PixelSpacing[1]]))
    spacing = np.array(list(spacing))
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    print(new_spacing)

    return image, new_spacing


def getpixelposition(firstslice, position):
    x, y, z = firstslice.ImagePositionPatient[0], firstslice.ImagePositionPatient[1], firstslice.ImagePositionPatient[2]
    spacing = firstslice.PixelSpacing[0]
    thickness = firstslice.SliceThickness
    imageposition = [round((position[0] - x)/spacing), round((position[1] - y)/spacing), round((position[2] - z)/thickness)]
    print(imageposition)

    return imageposition


if __name__ == "__main__":

    dataPath = r"F:\lymph_dataset\CT Lymph Nodes"

    patientDirlist = os.listdir(dataPath)
    # print(patientDirlist)
    # 获取指定病人DICOM数据集
    allPPathList = [dataPath + "\\" + patientNum for patientNum in patientDirlist]

    # 对病人切片进行处理
    # for patientPath in allPPathList:
    #     allPSlices = set_slice_thickness(load_scan(patientPath))
    #     allPHU = get_pixels_hu(allPSlices)

    firstPList = allPPathList[0]
    print(allPPathList)
    firstPSlices = load_scan(firstPList)
    # Get Dicom Picture
    # RefDs = dicom.read_file(first_patient[0])
    # res is the .dcm file list that dicom.read_file method returned
    # 该病人的所有切片列表
    # 该数据集中SliceThickness属性缺失，通过以下方法计算
    firstPSlices = set_slice_thickness(firstPSlices)
    # 该病人的第一个切片
    RefDs = firstPSlices[0]
    print(RefDs.ImagePositionPatient)
    getpixelposition(RefDs, [-64.59, 126.48, -623.60])
    # sip = RefDs[0x00200032].value
    HU = get_pixels_hu(firstPSlices)
    # res = resample(HU, RefDs)
    print(HU.shape)
    # print(firstPSlices[0].pixel_array.shape, firstPSlices[0].pixel_array[0])
    sip = RefDs.ImagePositionPatient
    # print(sip, RefDs.PixelSpacing[0], RefDs.PixelSpacing[1])
    # 图表显示病人HU分布
    # plt.hist(HU.flatten(), bins=80, color='c')
    # plt.xlabel("Hounsfield Units (HU)")
    # plt.ylabel("Frequency")
    # plt.show()
    # 窗口操作
    # img = setDicomCenWid(HU, "abdomen")
    # plt.imshow(img[0], cmap=plt.cm.gray)
    # plt.show()
    # print(RefDs.SliceThickness)
    # print(RefDs.pixel_array[0:100][0:100])
    # RefDs = dicom.dcmread(first_patient[0])
    # print(RefDs, sip)
    # print(RefDs.dir("pat"))
    # print(RefDs.pixel_array.shape)
    # plt.imshow(RefDs.pixel_array, cmap=plt.cm.bone)
    # plt.show()
    # cv2.imshow("patient_1", RefDs.pixel_array)
    # cv2.waitKey(0)
    # # Load dimensions based on the number of row, columns, and slices (along the two axis)
    ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(firstPSlices))
    # print(ConstPixelDims)

    # PixelSpacing - 每个像素点实际的长度与宽度,单位(mm)
    # SliceThickness - 每层切片的厚度,单位(mm)


    ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))

    # 三维数据
    x = np.arange(0.0, (ConstPixelDims[0] + 1) * ConstPixelSpacing[0], ConstPixelSpacing[0])
    y = np.arange(0.0, (ConstPixelDims[1] + 1) * ConstPixelSpacing[1], ConstPixelSpacing[1])
    z = np.arange(0.0, (ConstPixelDims[2] + 1) * ConstPixelSpacing[2], ConstPixelSpacing[2])
    # The array is sized based on 'ConstpixelDims'
    ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)
    # print(ArrayDicom.shape)
    # 遍历所有的dicom文件，读取图像数据，存放在numpy数组中
    for posDCM in firstPSlices:
        ArrayDicom[:, :, firstPSlices.index(posDCM)] = posDCM.pixel_array


    # # 轴状面显示
    # # dpi是指每英寸的像素数,dpi越大,表示打印出来的图片越清晰。不是指图片的大小.
    # # 像素用在显示领域 分辨率用在打印领域 也就是你的图像是用来打印的时候才去考虑分辨率的问题
    # plt.figure(dpi=1000)
    # # 将坐标轴都变为同等长度
    # # pyplot.axes().set_aspect('equal', 'datalim')
    # plt.axes().set_aspect('equal')
    # # 将图片变为gray颜色
    # plt.set_cmap(plt.gray())
    #
    # plt.imshow(ArrayDicom[:, :, 330], cmap=plt.cm.bone)  # 第三个维度表示现在展示的是第几层
    # plt.show()
    # # 矢状面显示
    # plt.figure(dpi=1000)
    # plt.axes().set_aspect('equal', 'datalim')
    # plt.set_cmap(plt.gray())
    # plt.imshow(ArrayDicom[:, 255, :], cmap=plt.cm.bone)
    # plt.show()
    # # 冠状面显示
    # plt.figure(dpi=1000)
    # plt.axes().set_aspect('equal', 'datalim')
    # plt.set_cmap(plt.gray())
    # plt.imshow(ArrayDicom[255, :, :], cmap=plt.cm.bone)
    # plt.show()




