#import h5py
#import numpy as np
import os
import time
#import warnings
from scipy import linalg
import torch
from utils import list_features, features_to_COXY

class RegionRefinerTrainer():
    def __init__(self, cfg):
        self.cfg = cfg
        self.features_format = self.cfg['FEATURES_FORMAT']
        self.path_to_features = self.cfg['PATHS']['FEATURES_PATH']+'/%s'+self.features_format
        self.path_to_imgset_train = self.cfg['PATHS']['IMAGESET_TRAIN']
        self.path_to_imgset_val = self.cfg['PATHS']['IMAGESET_VAL']
        self.features_dictionary_train = list_features(self.path_to_imgset_train)
        return

    def __call__(self):
        self.train()
        return

    def train(self):
        COXY = features_to_COXY(self.path_to_features, self.features_dictionary_train, self.cfg['NUM_CLASSES'])

        chosen_classes = ("__background__",
            "flower2", "flower5", "flower7",
            "mug1", "mug3", "mug4",
            "wallet6", "wallet7", "wallet10",
            "sodabottle2", "sodabottle3", "sodabottle4",
            "book4", "book6", "book9",
            "ringbinder4", "ringbinder5", "ringbinder6",
            "bodylotion2", "bodylotion5", "bodylotion8",
            "sprayer6", "sprayer8", "sprayer9",
            "pencilcase3", "pencilcase5", "pencilcase6",
            "hairclip2", "hairclip6", "hairclip8"
            )
        imdb = {
            'classes': chosen_classes
        }

        opts = self.cfg['opts']

        # cache_dir = 'bbox_reg/'
        # if not os.path.exists(cache_dir):
        #    os.mkdir(cache_dir)
        clss = imdb['classes']
        num_clss = len(clss)
        bbox_model_suffix = '_first_test'

        models = np.empty((0))
        # print(models)

        start_time = time.time()
        for i in range(1, num_clss):
            print('Training regressor for class %s (%d/%d)' % (imdb['classes'][i], i, num_clss - 1))
            # Compute indices where bboxes of class i overlap with the ground truth
            #I = np.logical_and(COXY['O'] > opts['min_overlap'], COXY['C'] == i)
            I = (COXY['O'] > opts['min_overlap']) & (COXY['C'] == i)
            #I = np.squeeze(np.where(I == True))
            I = torch.where(I == True)[0]
            print(I)
            if len(I) == 0:
                models = np.append(models, {'mu': None,
                                            'T': None,
                                            'T_inv': None,
                                            'Beta': None
                                            })
                print('No indices for class %s' % (imdb['classes'][i]))
                continue
            # Extract the corresponding values in the X matrix
            # Transpose is used to set the number of features as the number of columns (instead of the number of rows)
            Xi = COXY['X'][I]
            Yi = COXY['Y'][I]
            # TODO check if Oi and Ci computations are required
            # Add bias values to Xi
            #bias = np.ones((Xi.shape[0], 1), dtype=np.float32)
            bias = torch.ones((Xi.size()[0], 1), dtype=torch.float32).to("cuda")
            #Xi = np.append(Xi, bias, axis=1)
            Xi = torch.cat((Xi, bias), dim=1)

            # Center and decorrelate targets
            #mu = np.mean(Yi, axis=0)
            mu = torch.mean(Yi, dim=0)
            #Yi -= np.broadcast_arrays(Yi, mu)[1]  # Resize mu as Yi with broadcast_arrays in [1], subtract mu to Yi
            Yi -= mu
            # TODO understand where this formula comes from
            #S = np.matmul(Yi.T, Yi) / Yi.shape[0]
            S = torch.matmul(torch.t(Yi), Yi) / Yi.size()[0]
            #D, W = np.linalg.eig(S)  # D in python is a vector, in matlab is a diagonal matrix
            D, W = torch.eig(S, eigenvectors=True)
            D = D[:, 0]
            #T = np.matmul(np.matmul(W, np.diag(1 / np.sqrt(D + 0.001))), W.T)
            T = torch.matmul(torch.matmul(W, torch.diag(torch.sqrt(D + 0.001).pow_(-1))), torch.t(W))
            #T_inv = np.matmul(np.matmul(W, np.diag(np.sqrt(D + 0.001))), W.T)
            T_inv = torch.matmul(torch.matmul(W, torch.diag(torch.sqrt(D + 0.001))), torch.t(W))
            #Yi = np.matmul(Yi, T)
            Yi = torch.matmul(Yi, T)
            #print(Yi.shape)

            Beta = self.solve(Xi, Yi, opts['lambda'])

            models = np.append(models, {
                'mu': mu,
                'T': T,
                'T_inv': T_inv,
                'Beta': Beta
            })

            mean_losses = torch.empty(0).to("cuda")
            for elem in models[i - 1]['Beta']:
                mean_losses = torch.cat((mean_losses, torch.mean(models[i - 1]['Beta'][elem]['losses'], dim=0, keepdim=True)))
            print('Mean losses:', mean_losses)

            # break
        end_time = time.time()
        print('Time required to train %d regressors: %f seconds.' % (num_clss - 1, end_time - start_time))

        return

    def solve(self, X, y, lmbd):
        #X_torch = torch.from_numpy(X).to("cuda")
        #y_torch = torch.from_numpy(y).to("cuda")
        #start_mult = time.time()
        X_transposed_X = torch.matmul(torch.t(X), X) + lmbd * torch.eye(2049).to("cuda")
        #start_cho = time.time()
        #print('Cho: %f seconds.' % (start_cho - start_mult))
        R = torch.cholesky(X_transposed_X)
        #end_cho = time.time()
        #print('Cho: %f seconds.' % (end_cho - start_cho))
        to_return = {}
        for i in range(0, 4):
            y_torch_i = y[:, i]
            #torch.matmul(torch.t(X_torch), y_torch_i)
            z = torch.triangular_solve(torch.matmul(torch.t(X), y_torch_i).view(2049, 1), R, upper=False).solution
            #end_ls1 = time.time()
            #print('LS1: %f seconds.' % (end_ls1 - end_cho))
            w = torch.triangular_solve(z, torch.t(R)).solution.view(2049)
            #end_ls2 = time.time()
            #print('LS2: %f seconds.' % (end_ls2 - end_ls1))
            losses = 0.5 * torch.pow((torch.matmul(X, w) - y_torch_i), 2)
            #losses = losses.cpu().numpy()
            #w = w.cpu().numpy()
            to_return[str(i)] = {'weights': w,
                                 'losses': losses}
        return to_return
