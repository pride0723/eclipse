'''
Created on 2018. 7. 16.

@author: DMSL-CDY

cycle GAN implementation
'''

import os
import torch
from collections import OrderedDict
from . import networks
from Cython.Plex.Regexps import Opt

class BaseModel():
    
    # modifiy parser to add command line options,
    # and also charge the defulat values if needed
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser
    
    def name(self):
        return 'BaseModel'
    
    def initializer(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain=opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.save_dir=os.path.join(opt.checkpoints_dir, opt.name)
        if opt.resize_or_crop != 'scale_width':
            torch.backends.cudnn.benchmakr=True
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.image_paths = []
        
    def set_input(self, input):
        self.input=input
    
    def forward(self):
        pass
    
    # load and print networks; create schedlers
    def setup(self, opt, parser=None):
        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
            
        if not self.isTrain of opt.continue_train:
            self.load_networks(opt.which_epoch)
        self.print_networks(opt.verbose)
        
    # make models eval mode during test time
    def eval(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()
            
    # used in test time, wrapping 'forward' in no_grad() so we don't save
    # intermediate steps for backprop
    def test(self):
        with torch.no_grad():
            self.forward()
            
    # get image paths
    def get_image_paths(self):
        return self.image_paths
    
    def optimize_paramters(self):
        pass
    
    # update learning rate (called once every epoch)
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %7f' % lr)
        
    # return visualization images. train.py will display these images, and save the images to a html
    def get_current_visuals(self):
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret
    
    # return training losses/errors. train.py will print out these errors as debugging information
    def get_current_losses(self):
        errors_net = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                # float(...) works for both scalar tensor and float number
            
        
    
            
        
        
    
    
