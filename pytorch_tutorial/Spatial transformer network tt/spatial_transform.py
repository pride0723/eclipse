'''
Created on 2018. 4. 16.

@author: cdy

spatial transformer example
'''

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

plt.ion()

'''
# Loding the data

In this post we experiment with the classic MNIST dataset. Using a standard
convolutional ntwork augmented with a spatial tansformer network.
'''

use_cuda = torch.cuda.is_available()

# Traiining dataset
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='.', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                       ])), batch_size = 64, shuffle=True, num_workers=4)
# Test dataset
test_loader= torch.utils.data.DataLoader(
    dataset.MNIST(root='.', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307, ), (0.3081, ))
        ])), batch_size=64, shuffle=True, num_workers=4)



'''
Depicting spatial transformer networks
'''

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size = 5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size = 5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MxaxPool2d(2, stride=2),
            nn.ReLu(True),
            nn.Conv2d(8,10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        
        # Regressor for the 3 * 2 affine maxtrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3*2)
        )
        
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.fill_(0)
        self.fc_loc[2],bias.data = torch.FloatTensor([1, 0, 0, 0, 1, 0])
        
    # Spatial transformer network forward function
    def stn(self,x):
        xs= self.localization(x)





