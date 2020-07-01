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


class FALKONWrapper(ca.ClassifierAbstract):
    def __init__(self, cfg_path=None):
        if cfg_path is not None:
            self.cfg = yaml.load(open(cfg_path), Loader=yaml.FullLoader)
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
            if opts['kernel_type'] == 'gauss':
                self.kernel = kernels.GaussianKernel(sigma=self.sigma)
            else:
                print('Kernel type: %s unknown'.format(opts['kernel_type']))
            self.nyst_centers = opts['M']
            """
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
                )
            else:
                print('Kernel is None in trainRegionClassifier function')
                sys.exit(0)
            """

    def train(self, X, y, sigma=None, lam=None):
        """
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
        """
        if sigma is None:
            sigma = self.sigma
        if lam is None:
            lam = self.lam

        indices = self.compute_indices_selection(y)
        X = X[indices,:]
        y = y[indices]
        #indices = torch.randint(X.size()[0], (min(X.size()[0], self.nyst_centers),)).squeeze().tolist()
        center_selector = MyCenterSelector(indices)
        if self.kernel is not None:
            #self.nyst_centers = len(indices)#opts['M']
            self.model = Falkon(
                kernel=self.kernel,
                penalty=lam,
                M=len(indices),#self.nyst_centers,
#                debug=False,
#                use_cpu=True
                # use_display_gpu=True,
                # gpu_use_processes=False,
                # inter_type=torch.float32,
                # final_type=torch.float32
                #center_selection = center_selector
            )
        else:
            print('Kernel is None in trainRegionClassifier function')
            sys.exit(0)

        if self.model is not None:
            if sigma is not None:
                self.model.kernel = kernels.GaussianKernel(sigma=sigma)
            if lam is not None:
                self.model.penalty = lam                
            self.model.M = len(indices)
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

