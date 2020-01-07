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
import torchvision.transforms as transforms

import LABLE_MAEKER.makelable as ml
import Capsule3D.modle as md

BATCH_SIZE = 2
learning_rate = 1e-5
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
        self.lymphlable = []

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
        itlable = np.asarray(int(self.lymphlable[index]))
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
            # print(pixelposition)
            if pixelposition[2] >= 15:
                lymphslices = slices[pixelposition[2] - 15:pixelposition[2] + 15]
                if len(slices) - pixelposition[2] < 15:
                    lymphslices = slices[len(slices) - 30:len(slices)]
            else:
                lymphslices = slices[0:30]
            # print(lymphslices)
            slices = self.get_pixels_hu(lymphslices, pixelposition)
            image = self.setDicomCenWid(slices, "lymph")
            # image = self.normalize(image)
            image = image[np.newaxis, :, :, :]
            # print(image.shape)
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
                          positioninslice[0]:positioninslice[0] + 30,
                          positioninslice[1]: positioninslice[1] + 30] for s in slices])
        image = image.astype(np.int16)
        image[image == -2000] = 0

        for slice_number in range(len(slices)):
            intercept = slices[slice_number].RescaleIntercept
            slope = slices[slice_number].RescaleSlope
            if slope != 1:
                image[slice_number] = slope * image[slice_number].astype(np.float64)
                image[slice_number] = image[slice_number].astype(np.int16)
            image[slice_number] += np.int16(intercept)

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
            center = 50
            width = 300
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

    def normalize(self, slices):
        _range = np.max(slices) - np.min(slices)
        return (slices - np.min(slices)) / _range

    def resample(self, image, slice, new_spacing=[1, 1, 1]):
        spacing = map(float, ([slice.SliceThickness] + [slice.PixelSpacing[0], slice.PixelSpacing[1]]))
        spacing = np.array(list(spacing))
        resize_factor = spacing / new_spacing
        new_real_shape = image.shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / image.shape
        new_spacing = spacing / real_resize_factor
        image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

        return image, new_spacing


def train(model, train_loader, test_loader, args):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param train_loader: torch.utils.data.DataLoader for training data
    :param test_loader: torch.utils.data.DataLoader for test data
    :param args: arguments
    :return: The trained model
    """
    print('Begin Training' + '-' * 70)
    from time import time
    import csv
    logfile = open(args.save_dir + '/log.csv', 'a')
    logwriter = csv.DictWriter(logfile, fieldnames=['epoch', 'loss', 'val_loss', 'val_acc'])
    logwriter.writeheader()

    print("Model's state_dict:")
    # Print model's state_dict
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    t0 = time()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = md.CapsuleLoss()
    lr_decay = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)
    best_val_acc = 0.
    for epoch in range(args.epochs):
        print('*' * 25, 'epoch{}'.format(epoch + 1), '*' * 25)
        model.train()  # set to training mode

        ti = time()
        training_loss = 0.0
        for i, data in enumerate(train_loader, start=1):
            image, label, pos = data[0], data[1], data[2]
            if pd.isna(image).any():
                continue
            else:
                # print(image)
                # label = label.long()
                label = torch.zeros(label.size(0), 2).scatter_(1, label.view(-1, 1), 1.)  # change to one-hot coding
                # print(label)
                if use_gpu:
                    image = Variable(image).cuda()
                    label = Variable(label).cuda()
                else:
                    image = Variable(image)
                    label = Variable(label)
                optimizer.zero_grad()  # set gradients of optimizer to zero
                classes = model(image, label)  # forward
                # print(classes, reconstructions.shape)
                loss = criterion(image, label, classes)  # compute loss
                print(loss.data.item())
                loss.backward()  # backward, compute all gradients of loss w.r.t all Variables
                training_loss += loss.data.item() * image.size(0)  # record the batch loss
                optimizer.step()  # update the trainable parameters with computed gradients
        lr_decay.step()  # decrease the learning rate by multiplying a factor `gamma`
        # compute validation loss and acc
        val_loss, val_acc = test(model, test_loader, args)
        logwriter.writerow(dict(epoch=epoch, loss=training_loss / len(train_loader.dataset),
                                val_loss=val_loss, val_acc=val_acc))
        print("==> Epoch %02d: loss=%.5f, val_loss=%.5f, val_acc=%.4f, time=%ds"
              % (epoch, training_loss / len(train_loader.dataset),
                 val_loss, val_acc, time() - ti))
        if val_acc > best_val_acc:  # update best validation acc and save model
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.save_dir + '/epoch%d.pkl' % epoch)
            print("best val_acc increased to %.4f" % best_val_acc)
    logfile.close()
    torch.save(model.state_dict(), args.save_dir + '/trained_model_3.pkl')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)
    print("Total time = %ds" % (time() - t0))
    print('End Training' + '-' * 70)
    return model


def test(model, test_loader, args):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = md.CapsuleLoss()
    for i, data in enumerate(test_loader, start=1):
        image, label, pos = data[0], data[1], data[2]
        # label = label.long()
        label = torch.zeros(label.size(0), 2).scatter_(1, label.view(-1, 1), 1.)  # change to one-hot coding
        if use_gpu:
            image = Variable(image).cuda()
            label = Variable(label).cuda()
        else:
            image = Variable(image)
            label = Variable(label)

        y_pred = model(image)
        test_loss += criterion(image, label, y_pred).data.item() * image.size(0)  # sum up batch loss
        y_pred = y_pred.data.max(1)[1]
        y_true = label.data.max(1)[1]
        print(y_pred, y_true)
        correct += y_pred.eq(y_true).cpu().sum()
        correct = correct.item()

    test_loss /= len(test_loader.dataset)
    print(correct, len(test_loader.dataset))
    return test_loss, correct / len(test_loader.dataset)


def load_dataset(download=False, batch_size=4, shift_pixels=2):
    """
    Construct dataloaders for training and test data. Data augmentation is also done here.
    :param path: file path of the dataset
    :param download: whether to download the original data
    :param batch_size: batch size
    :param shift_pixels: maximum number of pixels to shift in each direction
    :return: train_loader, test_loader
    """
    trainpath = r'../lymph_dataset/trainpath.txt'
    testpath = r'../lymph_dataset/testpath.txt'

    traindata = MyDataset(trainpath, augment=True)
    testdata = MyDataset(testpath, augment=True)
    train_loader = DataLoader(traindata, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(testdata, batch_size=batch_size, shuffle=True, num_workers=4)

    return train_loader, test_loader


# def show_reconstruction(model, test_loader, n_images, args):
#     import matplotlib.pyplot as plt
#     from utils import combine_images
#     from PIL import Image
#     import numpy as np
#
#     model.eval()
#     for x, _ in test_loader:
#         x = Variable(x[:min(n_images, x.size(0))].cuda(), volatile=True)
#         _, x_recon = model(x)
#         data = np.concatenate([x.data, x_recon.data])
#         img = combine_images(np.transpose(data, [0, 2, 3, 1]))
#         image = img * 255
#         Image.fromarray(image.astype(np.uint8)).save(args.save_dir + "/real_and_recon.png")
#         print()
#         print('Reconstructed images are saved to %s/real_and_recon.png' % args.save_dir)
#         print('-' * 70)
#         plt.imshow(plt.imread(args.save_dir + "/real_and_recon.png", ))
#         plt.show()
#         break


if __name__ == "__main__":
    import argparse

    print('start')
    dataPath = r"../lymph_dataset/CT_Lymph_Nodes"
    lablePath = r"../lymph_dataset/MED_ABD_LYMPH_ANNOTATIONS"
    candidatePath = r"../lymph_dataset/MED_ABD_LYMPH_CANDIDATES"
    labeltxt = r'../lymph_dataset/label.txt'
    myABDneglabel = '../lymph_dataset/myABDneglabel.txt'
    myABDposlabel = '../lymph_dataset/myABDposlabel.txt'
    myMEDneglabel = '../lymph_dataset/myMEDneglabel.txt'
    myMEDposlabel = '../lymph_dataset/myMEDposlabel.txt'
    trainpath = r'../lymph_dataset/trainpath.txt'
    testpath = r'../lymph_dataset/testpath.txt'

    patientDirlist = os.listdir(dataPath)
    lableDirlist = os.listdir(lablePath)
    candidateDirlist = os.listdir(candidatePath)
    # print(patientDirlist)
    # 获取指定病人DICOM数据集
    allPPathList = [dataPath + "/" + patientNum for patientNum in patientDirlist]
    candidatePathlist = [candidatePath + "/" + lableNum for lableNum in candidateDirlist]
    ml.maketxtfile(allPPathList, candidatePathlist)
    print("successfully maketxt")
    ml.makelabel(labeltxt, myABDneglabel, myABDposlabel, myMEDneglabel, myMEDposlabel)
    print("successfully mylable")
    ml.make_traintest_label(myABDneglabel, myABDposlabel, trainpath, testpath)
    print("successfully traintesttxt")

    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="3D capsule networck on lymph dataset")
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.9, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('--lam_recon', default=0.0005 * 784, type=float,
                        help="The coefficient for the loss of decoder")
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")  # num_routing should > 0
    parser.add_argument('--shift_pixels', default=2, type=int,
                        help="Number of pixels to shift at most in each direction.")
    parser.add_argument('--download', action='store_true',
                        help="Download the required data.")
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    args = parser.parse_args()
    print(args)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load data
    train_loader, test_loader = load_dataset(download=False, batch_size=args.batch_size)

    model = md.CapsuleNet_3D()
    use_gpu = torch.cuda.is_available()

    if use_gpu:
        model = model.cuda()

    if args.weights is not None:  # init the model weights with provided one
        model.load_state_dict(torch.load(args.weights))

    if not args.testing:
        train(model, train_loader, test_loader, args)
    else:  # testing
        if args.weights is None:
            print('No weights are provided. Will test using random initialized weights.')
        test_loss, test_acc = test(model=model, test_loader=test_loader, args=args)
        print('test acc = %.4f, test loss = %.5f' % (test_acc, test_loss))

        # show_reconstruction(model, test_loader, 50, args)
