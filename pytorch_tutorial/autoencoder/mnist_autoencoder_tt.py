'''
Created on 2018. 4. 30.

@author: DMSL-CDY


MNIST_autoencoder Example
'''

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np


# Hyper Parameters
EPOCH = 10
BATCH_SIZE = 64
LR = 0.005   # learning rate
DOWNLOAD_MNIST = False
N_TEST_IMG = 5;

# Mnist digits dataset
train_data = torchvision.datasets.MNIST(
    root = 'E:/git/eclipse/eclipse/pytorch_tutorial/MNIST/',
    train = True,
    transform = torchvision.transforms.ToTensor(),
    download = DOWNLOAD_MNIST
    )
