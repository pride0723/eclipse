'''
Created on 2018. 5. 22.

@author: DMSL-CDY

GAN tutorial
'''

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd import Variable

def to_var(x):
    if torch.cuda.is_available(): #@UndefinedVariable
        x = x.cuda()
    return Variable(x)

def denorm(x):
    out = (x+1)/2
    return out.clamp(0,1)

Train_only = False




# Image processing 
transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                         std =(0.5, 0.5, 0.5))])

# MNIST dataset
mnist = datasets.MNIST(root='E:\git\eclipse\eclipse\pytorch_tutorial\MNIST',
                       train = True, 
                       transform=transform,
                       download=False)

data_loader = torch.utils.data.DataLoader(dataset=mnist, batch_size = 100, shuffle=True) #@UndefinedVariable


# Discriminator

D = nn.Sequential(
    nn.Linear(784,256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256,1),
    nn.Sigmoid())

# Generator
G = nn.Sequential(
    nn.Linear(64, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 784),
    
    nn.Tanh())

if torch.cuda.is_available():#@UndefinedVariable
    D.cuda()
    G.cuda()
    
# Binary cross entropy loss ans optimizer
criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0003) #@UndefinedVariable
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0003) #@UndefinedVariable



if Train_only == True:

    # Start training
    for epoch in range(200):
        for i, (images, _) in enumerate(data_loader):
            # Build mini-bacth dataset
            batch_size = images.size(0)
            images = to_var(images.view(batch_size, -1))
            
            # Create the labes which are later used as imput for the BCE loss
            real_labels = to_var(torch.ones(batch_size))
            fake_labels = to_var(torch.zeros(batch_size))
                
            #============== Train the discriminator =========#
            # Compute BCE_Loss using real images BCE_Loss(x, y) - y * log(D(x) - (1-y) * log(1-D(x))
            # Second term of the loss is always zero since real_labels == 1
            outputs = D(images)
            d_loss_real = criterion(outputs, real_labels)
            real_score = outputs
        
            # Compute BCELoss using fake images
            # First term of the loss is always zeros since fake_labels == 0
            z = to_var(torch.randn(batch_size, 64))
            fake_images = G(z)
            outputs =  D(fake_images)
            d_loss_fake = criterion(outputs, fake_labels)
            fake_score = outputs
            
            # Backprop + Optimize
            d_loss = d_loss_real + d_loss_fake
            D.zero_grad()
            d_loss.backward()
            d_optimizer.step()
            
            #=============== Train the generator =========#
            # compute loss with fake images 
            z = to_var(torch.randn(batch_size, 64))
            fake_images = G(z)
            outputs = D(fake_images)
            
            # We train G to maximize log(D(G(z)) instead of minimzing log(1-D(G(z)))
            # For the reason, see the last paragraph of section 3. http://arxiv.org/pdf/1406.2661.pdf
            g_loss = criterion(outputs, real_labels)
            
            # Backprop + Optimize
            D.zero_grad()
            G.zero_grad()
            g_loss.backward()
            g_optimizer.step()
            
            if (i+1) % 300 == 0:
                print('Epoch[%d/%d], Step[%d/%d], d_loss: %.4f, '
                      'g_loss: %.4f, D(x): %.2f, D(G(z)): %.2f'
                      %(epoch, 200, i+1, 600, d_loss.data[0], g_loss.data[0],
                        real_score.data.mean(), fake_score.mean()))
                
        # Save real images
        if (epoch+1) == 1:
            images = images.view(images.size(0), 1, 28, 28)
            save_image(denorm(images.data), 'real_images.png')
                    
        # Save sampled images
                    
        
        fake_images = fake_images.view(images.size(0), 1, 28, 28)
        save_image(denorm(fake_images.data), 'fake_images-%d.png' %(epoch+1))
        





    # Save the trained parameters
    torch.save(G.state_dict(), './generator.pkl')
    torch.save(D.state_dict(), './discriminator.pkl')



# inference only

G_model_path = './generator.pkl'
        
G.load_state_dict(torch.load(G_model_path))

batch_size = 100  
z = to_var(torch.randn(batch_size, 64))
fake_images = G(z)       
    
fake_images = fake_images.view(batch_size, 1, 28, 28)
save_image(denorm(fake_images.data), 'fake_images-infer.png')
                
        
        
        
        
        
        
    




