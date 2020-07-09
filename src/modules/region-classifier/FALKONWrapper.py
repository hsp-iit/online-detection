import sys
import os
basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir)))
from falkon import Falkon, kernels
import ClassifierAbstract as ca
import torch
import yaml

import copy
from falkon.options import *


class FALKONWrapper(ca.ClassifierAbstract):
    def __init__(self, cfg_path=None):
        if cfg_path is not None:
            self.cfg = yaml.load(open(cfg_path), Loader=yaml.FullLoader)
            opts = self.cfg['ONLINE_REGION_CLASSIFIER']['CLASSIFIER']

            if 'sigma' in opts:
                sigma = opts['sigma']
            else:
                print('Sigma not given for creating Falkon, default value is used.')
                sigma = 25

            if 'lam' in opts:
                lam = opts['lambda']
            else:
                print('Lambda not given for creating Falkon, default value is used.')
                lam = 0.001

            kernel = None
            if opts['kernel_type'] == 'gauss':
                kernel = kernels.GaussianKernel(sigma=sigma)
            else:
                print('Kernel type: %s unknown'.format(opts['kernel_type']))

            options = FalkonOptions(use_cpu=True) 

            if kernel is not None:
                self.nyst_centers = opts['M']
                self.model = Falkon(
                    kernel=kernel,
                    penalty=lam,
                    M=self.nyst_centers,
#                    debug=False,
#                    use_cpu=True
                    # use_display_gpu=True,
                    # gpu_use_processes=False,
                    # inter_type=torch.float32,
                    # final_type=torch.float32
                    options = options
                )
            else:
                print('Kernel is None in trainRegionClassifier function')
                sys.exit(0)

    def train(self, X, y, sigma=None, lam=None):

        if self.model is not None:
            if sigma is not None:
                self.model.kernel = kernels.GaussianKernel(sigma=sigma)
            if lam is not None:
                self.model.penalty = lam                
            self.model.M = min(self.nyst_centers, len(X))
            self.model.fit(X, y)
        else:
            print('Model is None in trainRegionClassifier function')
            sys.exit(0)

        return copy.deepcopy(self.model) #self.model

    def predict(self, model, X_np, y=None):
#        X = torch.from_numpy(X_np)
        if y is not None:
            predictions = model.predict(X_np, y)
        else:
            predictions = model.predict(X_np)

        return predictions

    def test(self):
        pass

