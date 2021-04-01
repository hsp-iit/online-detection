import sys
import os
basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir)))
from falkon import Falkon, kernels
import ClassifierAbstract as ca
import torch
import yaml

import copy

from MyCenterSelector import MyCenterSelector
from falkon.options import *


class FALKONWrapper(ca.ClassifierAbstract):
    def __init__(self, cfg_path=None, is_rpn=False, is_segmentation=False):
        if cfg_path is not None:
            self.cfg = yaml.load(open(cfg_path), Loader=yaml.FullLoader)
            if is_rpn:
                self.cfg = self.cfg['RPN']
        if not is_segmentation:
            opts = self.cfg['ONLINE_REGION_CLASSIFIER']['CLASSIFIER']
        else:
            opts = self.cfg['ONLINE_SEGMENTATION']['CLASSIFIER']

        if 'sigma' in opts:
            self.sigma = opts['sigma']
        else:
            print('Sigma not given for creating Falkon, default value is used.')
            self.sigma = 5

        if 'lambda' in opts:
            self.lam = opts['lambda']
        else:
            print('Lambda not given for creating Falkon, default value is used.')
            self.lam = 0.001

        self.kernel = None
        self.nyst_centers = opts['M']

    def train(self, X, y, sigma=None, lam=None):
        # Set sigma and lambda
        if sigma is None:
            sigma = self.sigma
        if lam is None:
            lam = self.lam
        # Initialize kernel
        self.kernel = kernels.GaussianKernel(sigma=sigma)
        # Compute indices of nystrom centers
        indices = self.compute_indices_selection(y)
        center_selector = MyCenterSelector(indices)
        opt = FalkonOptions(min_cuda_iter_size_32=0, min_cuda_iter_size_64=0,  keops_active="no")
        # Initialize FALKON model
        self.model = Falkon(
            kernel=self.kernel,
            penalty=lam,
            M=len(indices),
            center_selection = center_selector,
            options=opt
        )
        # Train FALKON model
        if self.model is not None:
            self.model.fit(X, y)
        else:
            print('Model is None in trainRegionClassifier function')
            sys.exit(0)

        return copy.deepcopy(self.model)

    def predict(self, model, X_np, y=None):
        # Predict values
        if y is not None:
            predictions = model.predict(X_np, y)
        else:
            predictions = model.predict(X_np)

        return predictions

    def test(self):
        pass

    def compute_indices_selection(self, y):
        # Choose at most M/2 nystrom centers from positive training examples
        positive_indices = (y == 1).nonzero()
        if positive_indices.size()[0] > int(self.nyst_centers/2):
            positive_indices = positive_indices[torch.randint(positive_indices.size()[0], (int(self.nyst_centers/2),))]
        # Fill the centers with negative examples training examples
        negative_indices = (y == -1).nonzero()
        if negative_indices.size()[0] > self.nyst_centers - positive_indices.size()[0]:
            negative_indices = negative_indices[torch.randint(negative_indices.size()[0], (self.nyst_centers - positive_indices.size()[0],))]
        
        indices = torch.cat((positive_indices, negative_indices), dim=0)
        
        return indices.squeeze().tolist()

