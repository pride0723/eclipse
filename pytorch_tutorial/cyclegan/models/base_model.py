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
        self.gpu_ids = opt.gpu_dioe
    
    
