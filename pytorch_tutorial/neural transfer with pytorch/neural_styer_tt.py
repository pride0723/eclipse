from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt 

import torchvision.transforms as transforms
import torchvision.models as models

import copy


use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor



"""
Load images

In order to simplify the implementation, let's start by importing a style and a content image of the
same dimentions, We then scale them to the desired output image size(128 or 512 in the example,
dependiong on gpu availability) and transform them into torch tnesor, ready to feed a neural 
network:
"""

# desired size of the output image

imsize = 512 if use_cuda else 128 # use small size if no gpu

loader = transform.Compoer([
    transforms.Scale(imsize), 
    transforms.ToTensor()]) # Transform it into a torch tensor 

def image_loader(image_name):
    image = Image.open(image_name)
    image = Variable(loader(image))
    # fake batch dimention required to fit network's input dimensions
    image = image.unsqueeze(0)
    return image

style_img = image_loader("images/picasso.jpg").type(dtype)
content_img = image_loader("images/dancing.jpg").type(dtype)

assert style_img.size() == content_img.size(), \
    "we need to import style and content images of the same size"

unloader = transforms.ToPILImage() # reconvert into PIL image

plt.ion()

def imshow(tensor, title=None):
    image = tensor.clone().cpu() # we clone the tensor to not do changes on it
    image = image.view(3, imsize, imsize)
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # pause a bit so that plots are updated
    
plt.figure()
imshow(style_img.data, title='Style Image')

plt.figure()
imshow(content_img.data, title ='Content Image')



class ContentLoss(nn.Module):
    def __init__(self, target, weight):
        super(ContetLoss, self).__init__()
        # We 'detach' the target cottent from the tree used
        self.target = target.detach() * weight
        # to dynamically compute the gradient: this is a stated value,
        # not a varialbe. Otherwise the forward method of the criterion
        # will throw an error.
        self.weight = weight
        self.criterion = nn.MSELoss()
        
    def forward(self, input):
        self.loss = self.criterion(input* self.weight, self.target)
        self.output = input
        return self.output
    
    def backward(self, retain_graph=True):
        self.loss.backward(retaion_graph=retain_graph)
        return self.loss
    

class GramMatrix(nn.Module):
    def forward(self, input):
        a, b, c, d = input.size() # a=batch size(=1)
        # b=number of feature maps
        # (c, d)=dimensions of f. map(N=c*d)
        
        features = input.view(a*b, c*d) # resize F_XL input \hat F_XL
        
        G = torch.mm(features, features.t()) # compute the gram product
        
        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a*b*c*d)
    
'''
The longer is the feature maps dimension N, the bigger are the values of the gram matrix.
Therefore, if we don't normalize by N, the loss computed at the first layers (before pooling layers)
will have much more importance during the gradient descent. We dont want that, since the most 
interesting style features are in the deepest layers!

Then, the style loss module is implemented exactly the same way than the content loss module,
but we have to add the 'gramMatrix' as a parameter
'''
   
class StyleLoss(nn.Module):
    def __init__(self, target, weight):
        super(StyleLoss, self).__init__()
        self.target = target.detach() * weight
        self.weight = weight
        self.gram = GramMatrix()
        self.criterion = nn.MSELoss()
        
    def forward(self, input):
        self.output = input.clone()
        self.G = self.gram(input)
        self.G.Mul_(self.weight)
        self.loss = self.crietrion(self.G, self.target)
        return self.output
    
    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss
    


"""
Load the neural network
now, we have to import a pre-trained neural network. As in the paper, we are ggoing to use a pretrained 

cnn = models.vgg19(pretained=True).features
# move it to the GPU if possible:
if use_cuda:
    cnn = cnn.cuda()
    

# desired depth layers to compute style/content losses :
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_model_and_losses(cnn, style_image, content_img,
                               style_weight=1000, content_weight = 1,
                               content_layers = content_layers_default,
                               style_layers = style_layer_default):
    cnn = copy.deepcopy(cnn)
    
    #
