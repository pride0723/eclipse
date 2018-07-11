'''
Created on 2018. 4. 30.

@author: DMSL-CDY


MNIST_autoencoder Example
'''

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np



# Hyper Parameters
EPOCH = 20
BATCH_SIZE = 64
LR = 0.005   # learning rate
DOWNLOAD_MNIST = False
N_TEST_IMG = 5;

# Mnist digits dataset
train_data = torchvision.datasets.MNIST(
    root = 'E:/git/eclipse/eclipse/pytorch_tutorial/MNIST/',
    train = True,
    transform = torchvision.transforms.ToTensor(),
    download = DOWNLOAD_MNIST,    
    )


# plot on e example
print(train_data.train_data.size())   # (60000, 28, 28)
print(train_data.train_labels.size()) # (60000)
#plt.imshow(train_data.train_data[2].numpy(), cmap='gray')
#plt.title('%i' % train_data.train_labels[2])
#plt.show()

# Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
train_loader = Data.DataLoader(dataset = train_data, batch_size = BATCH_SIZE, shuffle=True)

class AutoEncoder_simple(nn.Module):
    def __init__(self):
        super(AutoEncoder_simple, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64,12),
            nn.Tanh(),
            nn.Linear(12,3)  # compress to 3 features which can be visualized in plt
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.Tanh(),
            nn.Linear(12, 64),
            nn.Tanh(),
            nn.Linear(64,128),
            nn.Tanh(),
            nn.Linear(128, 28*28),
            nn.Sigmoid(), # comptress to a range (0, 1)
        )
    
    def forward(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    
autoencoder = AutoEncoder_simple()
optimizer = optim.Adam(autoencoder.parameters(), lr=LR)
loss_func = nn.MSELoss()


# initialized figure
f, a = plt.subplots(2, N_TEST_IMG, figsize = (5,2))
plt.ion()  # continously plot

# Original data(first row) for viewing
view_data = Variable(train_data.train_data[:N_TEST_IMG].view(-1, 28*28).type(torch.FloatTensor)/255.)
for i in range(N_TEST_IMG):
    a[0][i].imshow(np.reshape(view_data.data.numpy()[i], (28, 28)), cmap='gray'); a[0][i].set_xticks(()); a[0][i].set_yticks(())

for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        b_x = Variable(x.view(-1, 28*28))  # batch x, shape (batch, 28*28)
        b_y = Variable(x.view(-1, 28*28))  # batch y, shape (batch, 28*28)
        b_label = Variable(y)              # batch label 
        
        encoded, decoded = autoencoder(b_x)
        
        loss = loss_func(decoded, b_y)     # mean square error
        optimizer.zero_grad()              # clear gradients for this training setp     
        loss.backward()                    # backpropagation, compute gradients
        optimizer.step()                   # apply gradients
        
        if step % 100 == 0:
            print('Epoch: ', epoch, '| train loss : {}'.format(loss.data[0]))
            
            
            # plotting decoded image (second row)
            _, decoded_data = autoencoder(view_data)
            for i in range(N_TEST_IMG):
                a[1][i].clear()
                a[1][i].imshow(np.reshape(decoded_data.data.numpy()[i], (28, 28)), cmap='gray')
                a[1][i].set_xticks(()); a[1][i].set_yticks(())
            plt.draw(); plt.pause(0.05)
        
plt.ioff()
plt.show()

# visualize in 3D plot
view_data = Variable(train_data.train_data[:200].view(-1, 28*28).type(torch.FloatTensor)/255.)
encoded_data, _ = autoencoder(view_data)
fig = plt.figure(2); ax = Axes3D(fig)
X, Y, Z = encoded_data.data[:, 0].numpy(), encoded_data.data[:, 1].numpy(), encoded_data.data[:, 2].numpy()
values = train_data.train_labels[:200].numpy()
for x, y, z, s in zip(X, Y, Z, values):
    c = cm.rainbow(int(255*s/9));    ax.text(x, y, z, s, backgroundcolor=c)  #@UndefinedVariable
ax.set_xlim(X.min(), X.max()); ax.set_ylim(Y.min(), Y.max()); ax.set_zlim(Z.min(0), Z.max())
plt.show()


            
            
            
        

    





