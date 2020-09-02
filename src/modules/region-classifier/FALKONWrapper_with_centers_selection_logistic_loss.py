import sys
import os
basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir)))
from falkon import Falkon, kernels, LogisticFalkon, gsc_losses
import ClassifierAbstract as ca
import torch
import yaml

import copy

from MyCenterSelector import MyCenterSelector
from falkon.options import *

import time


class FALKONWrapper(ca.ClassifierAbstract):
    def __init__(self, cfg_path=None, is_rpn=False):
        if cfg_path is not None:
            self.cfg = yaml.load(open(cfg_path), Loader=yaml.FullLoader)
            if is_rpn:
                self.cfg = self.cfg['RPN']
            opts = self.cfg['ONLINE_REGION_CLASSIFIER']['CLASSIFIER']

            if 'sigma' in opts:
                self.sigma = opts['sigma']
            else:
                print('Sigma not given for creating Falkon, default value is used.')
                self.sigma = 25

            if 'lambda' in opts:
                self.lam = opts['lambda']
            else:
                print('Lambda not given for creating Falkon, default value is used.')
                self.lam = 0.001

            self.kernel = None
            #if opts['kernel_type'] == 'gauss':
            #    self.kernel = kernels.GaussianKernel(sigma=self.sigma)
            #else:
            #    print('Kernel type: %s unknown'.format(opts['kernel_type']))
            self.nyst_centers = opts['M']

    def train(self, X, y, sigma=None, lam=None):
        if sigma is None:
            sigma = self.sigma
        if lam is None:
            lam = self.lam
        self.kernel = kernels.GaussianKernel(sigma=sigma)
        indices = self.compute_indices_selection(y)
        #X = X[indices,:]
        #y = y[indices]
        center_selector = MyCenterSelector(indices)
        opt = FalkonOptions(use_cpu=True)
        if self.kernel is not None:
            #self.nyst_centers = len(indices)#opts['M']
            self.model = LogisticFalkon(
                kernel=self.kernel,
                penalty_list=[lam],
                iter_list = [20],
                loss = gsc_losses.LogisticLoss(self.kernel),      #TODO change this if sigma=100
                M=len(indices),#self.nyst_centers,
#                debug=False,
#                use_cpu=True
                center_selection = center_selector,
                error_fn=binary_loss,
            )
        else:
            print('Kernel is None in trainRegionClassifier function')
            sys.exit(0)

        if self.model is not None:
            #if sigma is not None:
            #    self.model.kernel = self.kernel
            #if lam is not None:
            #    self.model.penalty = lam                
            #self.model.M = len(indices)
            #start = time.time()
            self.model.fit(X, y)
            #print('Fit time:', time.time()-start)
        else:
            print('Model is None in trainRegionClassifier function')
            sys.exit(0)
        #start = time.time()
        #a = copy.deepcopy(self.model)
        #print('Deepcopy time:', time.time()-start)
        return copy.deepcopy(self.model)

    def predict(self, model, X_np, y=None):
#        X = torch.from_numpy(X_np)
        if y is not None:
            predictions = model.predict(X_np, y)
        else:
            predictions = model.predict(X_np)

        return predictions

    def test(self):
        pass

    def compute_indices_selection(self, y):
        #print(type(y))
        positive_indices = (y == 1).nonzero()
        #print(type(positive_indices))
        if positive_indices.size()[0] > int(self.nyst_centers/2):
            positive_indices = positive_indices[torch.randint(positive_indices.size()[0], (int(self.nyst_centers/2),))]
        #print(positive_indices)
        negative_indices = (y == -1).nonzero()
        if negative_indices.size()[0] > self.nyst_centers - positive_indices.size()[0]:
            negative_indices = negative_indices[torch.randint(negative_indices.size()[0], (self.nyst_centers - positive_indices.size()[0],))]
        #print(negative_indices)
        
        indices = torch.cat((positive_indices, negative_indices), dim=0)
        #print(indices.size())
        #print(len(indices.tolist()))
        #quit()
        
        return indices.squeeze().tolist()

def binary_loss(true, pred):
    return torch.mean((true != torch.sign(pred)).to(torch.float32))
