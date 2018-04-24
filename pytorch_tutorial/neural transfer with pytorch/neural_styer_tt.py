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


#use_cuda = torch.cuda.is_available()
use_cuda = False
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor



"""
Load images

In order to simplify the implementation, let's start by importing a style and a content image of the
same dimentions, We then scale them to the desired output image size(128 or 512 in the example,
dependiong on gpu availability) and transform them into torch tnesor, ready to feed a neural 
network:
"""

# desired size of the output image

imsize = 512 if use_cuda else 512 # use small size if no gpu

loader = transforms.Compose([
    transforms.Resize(imsize), 
    transforms.ToTensor()]) # Transform it into a torch tensor 

def image_loader(image_name):
    image = Image.open(image_name)
    image = Variable(loader(image))
    # fake batch dimention required to fit network's input dimensions
    image = image.unsqueeze(0)
    return image

style_img = image_loader("picasso.jpg").type(dtype)
content_img = image_loader("dancing.jpg").type(dtype)

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
        super(ContentLoss, self).__init__()
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
        self.loss.backward(retain_graph=retain_graph)
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
        self.G.mul_(self.weight)
        self.loss = self.criterion(self.G, self.target)
        return self.output

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss
    




"""
Load the neural network
now, we have to import a pre-trained neural network. As in the paper, we are going to use a 
prerained VGG network with 19 layers(VGG19)

Pythoech'implementation of VGG is a module divided in two child "Sequential" module: "features"
(containing convolution and pooling layers) and "classifier" (containing fully connected layers). We
are just intetrdyrf by "featyres":
""" 

cnn = models.vgg19(pretrained=True).features
# move it to the GPU if possible:
if use_cuda:
    cnn = cnn.cuda()

"""
A sequential module contatins an ordered list of child modules. For instance, "vgg19.features"
contains a sequence(Conv2d, ReLU, Maxpoo2d, Conv2d, ReLU...) aligned in the right order of 
depth. As we said in Content loss section, we wand to add our style and content loss modules as
additive 'transparent' layers in our networks, at desired depths. For that, we construct a new
"Sequential moduls, in which we are going to add modules from "vgg19" and loss modules in the
right order:
"""

# desired depth layers to compute style/content losses :
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_model_and_losses(cnn, style_image, content_img,
                               style_weight=1000, content_weight = 1,
                               content_layers = content_layers_default,
                               style_layers = style_layers_default):
    cnn = copy.deepcopy(cnn)
    
    # just in order to have an iterable access to or list of content/style
    # losses
    content_losses = []
    style_losses = []
    
    model = nn.Sequential() # the new Sequential module network
    gram = GramMatrix() # we need a gram module in order to compute style targets
    
    # move these modules to the GPU if possible:
    if use_cuda:
        model=model.cuda()
        gram = gram.cuda()
    
    i = 1
    for layer in list(cnn):
        if isinstance(layer, nn.Conv2d):
            name = "conv_" + str(i)
            model.add_module(name, layer)
            
            if name in content_layers:
                # add content loss:
                target = model(content_img).clone()
                content_loss = ContentLoss(target, content_weight)
                model.add_module("Content_loss_"+str(i), content_loss)
                content_losses.append(style_loss)
            
            if name in style_layers:
                # add style loss:
                target_feature = model(style_img).clone()
                target_feature_gram = gram(target_feature)
                style_loss = StyleLoss(target_feature_gram, style_weight)
                model.add_module("Style_loss_" + str(i), style_loss)
                style_losses.append(style_loss)
            
        if isinstance(layer, nn.ReLU):
            name = "relu_" + str(i)
            model.add_module(name, layer)
            
            if name in content_layers:
                # add content loss:
                target = model(content_img).clone()
                content_loss = ContentLoss(target, content_weight)
                model.add_module("content_loss_" + str(i), content_loss)
                content_losses.append(content_loss)
                
            if name in style_layers:
                # add style loss:
                target = model(content_img).clone()
                content_loss = ContentLoss(target, content_weight)
                model.add_module("Style_loss_" + str(i), style_loss)
                style_losses.append(style_loss)
            
            i += 1
        
        if isinstance(layer, nn.MaxPool2d):
            name = "pool_" + str(i)
            model.add_module(name, layer) # ***
    
    return model, style_losses, content_losses

"""
Input image
Agin, in order to simplify the code, we take an image of the same dimensions than content and style
images. This image can be a white noise, or it can also be a copy of the content-image.

"""

input_img = content_img.clone()
# if you want to use a white noise instead uncomment the below line:
# input_img = Varibale(torch.randn(content_img.data.size()).type(dtype) 

# add the original input image to the figure:
plt.figure()
imshow(input_img.data, title='Input Image')


                
                
            
""" 
Gradient descent
As Leon Gatys, the author of the algorithm, suggetsted here, we will use L-BFGS algorithm to run our
gradient decent. Unlike training a network, we want to train the input image in order to minimize
the content/style losses. We would like to simply create a PyTorch L-BFGS optimizer, passing our
image as the variable to optimizer. But "opti.LBFGS" takes as first argument a list of PyTorch "variable"
that require gradients. In order to show that this variable requires a gradient, a possibility is to
construct a "Parameter" Object from the input image. Then we just give a list containing this "Parameter"
to the optimizer's" constructor:
"""

def get_input_param_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    input_param = nn.Parameter(input_img.data)
    optimizer = optim.LBFGS([input_param])
    return input_param, optimizer

"""
Last step: the loop of gradient descent. At each step, we must feed the network with the updated
input in order ot compute the new losses, we must run the "backward" methods of each loss to 
dynamically compute their gradients and perform the step of gradient descent. The optimizer
requires as argument a "closure": a function that reevaluates the model and returns the loss.

However, there's small catch. The optimized image may take its values between -inf, and +inf
instead of staying between 0 and 1. In other words, the image might be well optimized and have
absurd values. In fact we must perform an optimization under constraints in order to keep having
right values into our input image. There is a simple solution: at each step, to correct the image to
maintain its values into the 0-1 interval.
"""

def run_style_transfer(cnn, content_img, style_img, input_img, num_steps=300,
                       style_weight=1000, content_weight=1):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn, style_img, content_img, style_weight, content_weight)
    input_param, optimizer = get_input_param_optimizer(input_img)
    
    print('Optimizing...')
    run = [0]
    while run[0] <= num_steps:
        
        def closure():
            # correct the values of updated input image
            input_param.data.clamp_(0,1)
            
            optimizer.zero_grad()
            model(input_param)
            style_score = 0
            content_score = 0
            
            for sl in style_losses:
                style_score += sl.backward()
            for cl in content_losses:
                content_score += cl.backward()
            
            run[0] += 1
            if run[0] % 2 == 0:
                print("run {}:".format(run))

                print('Style Loss : {:4f} Content Loss {:4f}: '.format(
                    style_score.data[0], content_score.data[0]))
                
            return style_score + content_score
        optimizer.step(closure)
    
    # a last correction...
    input_param.data.clamp_(0,1)
    
    return input_param.data


output = run_style_transfer(cnn, content_img, style_img, input_img)        

plt.figure()
imshow(output, title='Output Image')

# sphinx_gallery_thumnail_number = 4
plt.ioff()
plt.show()

    