import h5py
import numpy as np
import os
import time
import sys
#import warnings
from scipy import linalg
import torch
from utils import list_features, features_to_COXY, features_to_COXY_boxlist

basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir)))
sys.path.append(os.path.abspath(os.path.join(basedir, '..', '..')))
sys.path.append(os.path.abspath(os.path.join(basedir, '..', '..', '..')))
from py_od_utils import getFeatPath

sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir, 'src', 'modules', 'region-classifier')))
import FALKONWrapper as falkon
#import FALKONWrapper_with_centers_selection as falkon

from sklearn import linear_model

class RegionRefinerTrainer():
    def __init__(self, cfg):
        self.cfg = cfg
        #self.features_format = self.cfg['FEATURE_INFO']['FORMAT']
        #self.feature_folder = getFeatPath(self.cfg)
        #feat_path = os.path.join(basedir, '..', '..', '..', '..', 'Data', 'feat_cache', self.feature_folder, 'train_val')
        #if 'UPDATE_RPN' in self.cfg:
        #    if self.cfg['UPDATE_RPN']:
        #        feat_path = os.path.join(basedir, '..', '..', '..', '..', 'Data', 'feat_cache_RPN', self.feature_folder, 'train_val')
        #self.path_to_features = feat_path + '/%s' + self.features_format
        self.path_to_imgset_train = self.cfg['DATASET']['TARGET_TASK']['TRAIN_IMSET']
        self.path_to_imgset_val = self.cfg['DATASET']['TARGET_TASK']['VAL_IMSET']
        self.features_dictionary_train = list_features(self.path_to_imgset_train)
        """
        self.stats_rpn = None
        if 'UPDATE_RPN' in self.cfg:
            if self.cfg['UPDATE_RPN']:
                try:
                    self.stats_rpn = torch.load(os.path.join(basedir, '..', '..', '..', '..', 'Data', 'feat_cache_RPN', self.feature_folder) + '/rpn_stats')
                    for key in self.stats_rpn.keys():
                        self.stats_rpn[key] = self.stats_rpn[key].to('cuda')
                except:
                    self.stats_rpn = None

        try:
            self.stats = torch.load(os.path.join(basedir, '..', '..', '..', '..', 'Data', 'feat_cache', self.feature_folder) + '/stats')
        except:
            self.stats = None
        """
        self.stats = None
        #TODO re implement this cross validation
        self.lambd = self.cfg['REGION_REFINER']['opts']['lambda']

        #feat_path_test = os.path.join(basedir, '..', '..', '..', '..', 'Data', 'feat_cache_RPN', self.feature_folder+'_debug', 'test')
        #self.path_to_features_test = feat_path_test + '/%s' + self.features_format
        self.path_to_imgset_test = self.cfg['DATASET']['TARGET_TASK']['TEST_IMSET']
        self.features_dictionary_test = list_features(self.path_to_imgset_test)
        self.test_losses = False
        self.percentile = 0#5
        self.normalize_features = True
        self.normalize_features_rpn = True

        self.COXY = None
        self.is_rpn = False

    def __call__(self):
        models = self.train()
        return models

    def train(self):
        chosen_classes = self.cfg['CHOSEN_CLASSES']
        start_index = 1

        if self.is_rpn:
            start_index = 0
        opts = self.cfg['REGION_REFINER']['opts']

        #feat_path = self.path_to_features
        """
        if self.COXY is None:
            print('here')
            quit()
            if 'UPDATE_RPN' in self.cfg:
                if self.cfg['UPDATE_RPN']:
                    self.COXY = features_to_COXY_boxlist(self.path_to_features, self.features_dictionary_train, min_overlap=opts['min_overlap'], feat_dim=1024, normalize_features = self.normalize_features_rpn, stats = self.stats_rpn)
                    if self.test_losses:
                        COXY_test = features_to_COXY_boxlist(self.path_to_features_test, self.features_dictionary_test, min_overlap=opts['min_overlap'], feat_dim=1024, normalize_features = self.normalize_features_rpn, stats = self.stats_rpn)
            else:
                try:
                    self.COXY = torch.load(os.path.join(basedir, '..', '..', '..', '..', 'Data', 'feat_cache', self.feature_folder, 'coxy_detector'))
                    print('Loading COXY from file')
                except:
                    self.COXY = features_to_COXY(self.path_to_features, self.features_dictionary_train, min_overlap=opts['min_overlap'], normalize_features = self.normalize_features, stats = self.stats)
                    torch.save(self.COXY, os.path.join(basedir, '..', '..', '..', '..', 'Data', 'feat_cache', self.feature_folder, 'coxy_detector'))
        """
        num_clss = len(chosen_classes)

        models = np.empty((0))

        start_time = time.time()
        for i in range(start_index, num_clss):#TODO was for i in range(1, num_clss)
            print('Training regressor for class %s (%d/%d)' % (chosen_classes[i], i, num_clss - 1)) # TODO was print('Training regressor for class %s (%d/%d)' % (chosen_classes[i], i, num_clss - 1))
            # Compute indices where bboxes of class i overlap with the ground truth
            #I = np.logical_and(COXY['O'] > opts['min_overlap'], COXY['C'] == i)
            if self.COXY['O'] is not None:
                I = (self.COXY['O'] >= opts['min_overlap']) & (self.COXY['C'] == i)
            else:
                I = self.COXY['C'] == i
            #print(torch.sum(I))
            #I = np.squeeze(np.where(I == True))
            I = torch.where(I == True)[0]
            #print(I)
            print('Training with %i examples' %len(I))
            if len(I) == 0:
                models = np.append(models, {'mu': None,
                                            'T': None,
                                            'T_inv': None,
                                            'Beta': None
                                            })
                #models_falkon.append(None)
                print('No indices for class %s' % (chosen_classes[i]))
                continue
            # Extract the corresponding values in the X matrix
            # Transpose is used to set the number of features as the number of columns (instead of the number of rows)
            Xi = self.COXY['X'][I]
            Yi = self.COXY['Y'][I]
            # Add bias values to Xi
            bias = torch.ones((Xi.size()[0], 1), dtype=torch.float32).to("cuda")
            Xi = torch.cat((Xi, bias), dim=1)

            # Center and decorrelate targets
            mu = torch.mean(Yi, dim=0)
            Yi -= mu
            S = torch.matmul(torch.t(Yi), Yi) / Yi.size()[0]
            D, W = torch.eig(S, eigenvectors=True)
            D = D[:, 0]
            T = torch.matmul(torch.matmul(W, torch.diag(torch.sqrt(D + 0.001).pow_(-1))), torch.t(W))
            T_inv = torch.matmul(torch.matmul(W, torch.diag(torch.sqrt(D + 0.001))), torch.t(W))
            Yi = torch.matmul(Yi, T)

            print(self.lambd)

            if self.test_losses:
                # Compute indices where bboxes of class i overlap with the ground truth
                I_test = (COXY_test['O'] >= opts['min_overlap']) & (COXY_test['C'] == i)
                I_test = torch.where(I_test == True)[0]
                print('Testing with %i positives' %len(I_test))
                # Extract the corresponding values in the X matrix
                # Transpose is used to set the number of features as the number of columns (instead of the number of rows)
                Xi_test = COXY_test['X'][I_test]
                Yi_test = COXY_test['Y'][I_test]
                # Add bias values to Xi
                bias_test = torch.ones((Xi_test.size()[0], 1), dtype=torch.float32).to("cuda")
                Xi_test = torch.cat((Xi_test, bias_test), dim=1)
                Yi_test -= mu
                Yi_test = torch.matmul(Yi_test, T)

                Beta = self.solve(Xi, Yi, self.lambd, Xi_test, Yi_test)#opts['lambda'])
            else:
                Beta = self.solve(Xi, Yi, self.lambd)

            if self.percentile > 0:
                indices = []
                for elem in Beta:
                    losses_i = Beta[elem]['losses']
                    threshold_i = np.percentile(losses_i.cpu(), self.percentile)
                    indices_i = torch.where(losses_i > threshold_i)[0]
                    indices.append(indices_i)
                Beta = self.solve(Xi, Yi, self.lambd, Xi_test, Yi_test, indices)

            models = np.append(models, {
                'mu': mu,
                'T': T,
                'T_inv': T_inv,
                'Beta': Beta
            })

            mean_losses = torch.empty(0).to("cuda")
            if self.test_losses:
                mean_losses_test = torch.empty(0).to("cuda")

            for elem in models[i - start_index]['Beta']:
                mean_losses = torch.cat((mean_losses, torch.mean(models[i - start_index]['Beta'][elem]['losses'], dim=0, keepdim=True)))
                if self.test_losses:
                    mean_losses_test = torch.cat((mean_losses_test, torch.mean(models[i - start_index]['Beta'][elem]['losses_test'], dim=0, keepdim=True)))
            print('Mean losses:', mean_losses)
            if self.test_losses:
                print('Mean losses_test:', mean_losses_test)

        end_time = time.time()
        print('Time required to train %d regressors: %f seconds.' % (num_clss, end_time - start_time))
        return models


    def solve(self, X, y, lmbd, X_test=None, Y_test=None, indices=None):
        if indices is None:
            R = None        
            while R is None:
                try:        # TODO check whether this is a good solution
                    X_transposed_X = torch.matmul(torch.t(X), X) + lmbd * torch.eye(X.size()[1]).to("cuda")
                    R = torch.cholesky(X_transposed_X)
                except:
                    lmbd *= 10
            print('Lambda:', lmbd)
        to_return = {}
        X_original = X.clone()
        y_original = y.clone()
        for i in range(0, 4):
            if indices is not None:
                X = X_original[indices[i]]
                y = y_original[indices[i]]
                R = None        
                while R is None:
                    try:
                        X_transposed_X = torch.matmul(torch.t(X), X) + lmbd * torch.eye(X.size()[1]).to("cuda")
                        R = torch.cholesky(X_transposed_X)
                    except:
                        lmbd *= 10
                print('Lambda:', lmbd)
            y_torch_i = y[:, i]
            z = torch.triangular_solve(torch.matmul(torch.t(X), y_torch_i).view(X.size()[1], 1), R, upper=False).solution
            w = torch.triangular_solve(z, torch.t(R)).solution.view(X.size()[1])
            losses = 0.5 * torch.pow((torch.matmul(X, w) - y_torch_i), 2)
            to_return[str(i)] = {'weights': w,
                                 'losses': losses}
            if X_test is not None:
                losses_test = 0.5 * torch.pow((torch.matmul(X_test, w) - Y_test[:, i]), 2)
                to_return[str(i)]['losses_test'] = losses_test 
        return to_return


