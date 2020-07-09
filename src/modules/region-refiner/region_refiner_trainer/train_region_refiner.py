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
        self.features_format = self.cfg['FEATURE_INFO']['FORMAT']
        feature_folder = getFeatPath(self.cfg)
        feat_path = os.path.join(basedir, '..', '..', '..', '..', 'Data', 'feat_cache_RPN', feature_folder+'_debug', 'train_val')
        self.path_to_features = feat_path + '/%s' + self.features_format
        self.path_to_imgset_train = self.cfg['DATASET']['TARGET_TASK']['TRAIN_IMSET']
        self.path_to_imgset_val = self.cfg['DATASET']['TARGET_TASK']['VAL_IMSET']
        self.features_dictionary_train = list_features(self.path_to_imgset_train)
        self.stats = torch.load('/home/IIT.LOCAL/fceola/workspace/ws_mask/python-online-detection/Data/feat_cache/R50_ep4_FTicwt100_TTicwt30/rpn_stats')
        for key in self.stats.keys():
            self.stats[key] = self.stats[key].to('cuda')
        #TODO re implement this cross validation
        self.lambd = None
        self.sigma = None

        feat_path_test = os.path.join(basedir, '..', '..', '..', '..', 'Data', 'feat_cache_RPN', feature_folder+'_debug', 'test')
        self.path_to_features_test = feat_path_test + '/%s' + self.features_format
        self.path_to_imgset_test = self.cfg['DATASET']['TARGET_TASK']['TEST_IMSET']
        self.features_dictionary_test = list_features(self.path_to_imgset_test)
        self.test_losses = False
        self.percentile = 0#5
        self.normalize_features = True

        self.pretrained_reg_weights = None #torch.load('/home/IIT.LOCAL/fceola/workspace/ws_mask/python-online-detection/experiments/first_experiment/reg_wgts').squeeze().cpu()
        self.pretrained_reg_bias = None #torch.load('/home/IIT.LOCAL/fceola/workspace/ws_mask/python-online-detection/experiments/first_experiment/reg_bias').squeeze().cpu()

    def __call__(self):
        models = self.train()
        return models

    def train(self):
        chosen_classes = self.cfg['CHOSEN_CLASSES']

        if 'UPDATE_RPN' in self.cfg:
            if self.cfg['UPDATE_RPN']:
                chosen_classes = self.cfg['CHOSEN_CLASSES_RPN']
        opts = self.cfg['REGION_REFINER']['opts']

        feat_path = self.path_to_features
        #if 'UPDATE_RPN' in self.cfg:
        #    positives_file = os.path.join(feat_path[:-15], 'RPN_bbox_positives')
        #else:
        #    positives_file = os.path.join(feat_path[:-15], 'bbox_positives')
        #try:
        #    COXY = torch.load(positives_file)
        #except:
        if 'UPDATE_RPN' in self.cfg:
            if self.cfg['UPDATE_RPN']:
                COXY = features_to_COXY_boxlist(self.path_to_features, self.features_dictionary_train, min_overlap=opts['min_overlap'], feat_dim=1024, normalize_features = self.normalize_features, stats = self.stats)
                #COXY = features_to_COXY_boxlist(self.path_to_features_test, self.features_dictionary_test, min_overlap=opts['min_overlap'], feat_dim=1024, normalize_features = True, stats = self.stats) # TODO remove this
                if self.test_losses:
                    COXY_test = features_to_COXY_boxlist(self.path_to_features_test, self.features_dictionary_test, min_overlap=opts['min_overlap'], feat_dim=1024, normalize_features = self.normalize_features, stats = self.stats)
        else:
            COXY = features_to_COXY(self.path_to_features, self.features_dictionary_train, min_overlap=opts['min_overlap'])
        #torch.save(COXY, positives_file)

        # cache_dir = 'bbox_reg/'
        # if not os.path.exists(cache_dir):
        #    os.mkdir(cache_dir)
        num_clss = len(chosen_classes)
        #bbox_model_suffix = '_first_test'

        models = np.empty((0))
        # print(models)

        #models_falkon = []
        start_time = time.time()
        for i in range(num_clss):#TODO was for i in range(1, num_clss)
            print('Training regressor for class %s (%d/%d)' % (chosen_classes[i], i, num_clss - 1)) # TODO was print('Training regressor for class %s (%d/%d)' % (chosen_classes[i], i, num_clss - 1))
            # Compute indices where bboxes of class i overlap with the ground truth
            #I = np.logical_and(COXY['O'] > opts['min_overlap'], COXY['C'] == i)
            I = (COXY['O'] >= opts['min_overlap']) & (COXY['C'] == i)
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
            Xi = COXY['X'][I]
            Yi = COXY['Y'][I]
            # TODO check if Oi and Ci computations are required
            # Add bias values to Xi
            #bias = np.ones((Xi.shape[0], 1), dtype=np.float32)
            bias = torch.ones((Xi.size()[0], 1), dtype=torch.float32).to("cuda")
            #Xi = np.append(Xi, bias, axis=1)
            Xi = torch.cat((Xi, bias), dim=1)
            #Beta = self.solve_with_falkon(Xi[:,:-1], Yi)    #TODO remove this
            #Beta = self.solve_with_L1_smooth_sklearn(Xi[:,:-1], Yi, i)

            # Center and decorrelate targets
            #mu = np.mean(Yi, axis=0)
            mu = torch.mean(Yi, dim=0)
            #Yi -= np.broadcast_arrays(Yi, mu)[1]  # Resize mu as Yi with broadcast_arrays in [1], subtract mu to Yi
            Yi -= mu
            #print(mu)
            # TODO understand where this formula comes from
            #S = np.matmul(Yi.T, Yi) / Yi.shape[0]
            S = torch.matmul(torch.t(Yi), Yi) / Yi.size()[0]
            #D, W = np.linalg.eig(S)  # D in python is a vector, in matlab is a diagonal matrix
            D, W = torch.eig(S, eigenvectors=True)
            D = D[:, 0]
            #T = np.matmul(np.matmul(W, np.diag(1 / np.sqrt(D + 0.001))), W.T)
            T = torch.matmul(torch.matmul(W, torch.diag(torch.sqrt(D + 0.001).pow_(-1))), torch.t(W))
            #T = torch.matmul(torch.matmul(W, torch.diag(torch.sqrt(D + 1/self.lambd).pow_(-1))), torch.t(W))
            #T_inv = np.matmul(np.matmul(W, np.diag(np.sqrt(D + 0.001))), W.T)
            T_inv = torch.matmul(torch.matmul(W, torch.diag(torch.sqrt(D + 0.001))), torch.t(W))
            #T_inv = torch.matmul(torch.matmul(W, torch.diag(torch.sqrt(D + 1/self.lambd))), torch.t(W))
            #Yi = np.matmul(Yi, T)
            Yi = torch.matmul(Yi, T)           #TODO uncomment this
            #print(Yi.shape)



            if self.test_losses:
                # Compute indices where bboxes of class i overlap with the ground truth
                #I = np.logical_and(COXY['O'] > opts['min_overlap'], COXY['C'] == i)
                I_test = (COXY_test['O'] >= opts['min_overlap']) & (COXY_test['C'] == i)
                #print(torch.sum(I))
                #I = np.squeeze(np.where(I == True))
                I_test = torch.where(I_test == True)[0]
                print('Testing with %i positives' %len(I_test))
                # Extract the corresponding values in the X matrix
                # Transpose is used to set the number of features as the number of columns (instead of the number of rows)
                Xi_test = COXY_test['X'][I_test]
                Yi_test = COXY_test['Y'][I_test]
                # TODO check if Oi and Ci computations are required
                # Add bias values to Xi
                #bias = np.ones((Xi.shape[0], 1), dtype=np.float32)
                bias_test = torch.ones((Xi_test.size()[0], 1), dtype=torch.float32).to("cuda")
                #Xi = np.append(Xi, bias, axis=1)
                Xi_test = torch.cat((Xi_test, bias_test), dim=1)
                Yi_test -= mu
                Yi_test = torch.matmul(Yi_test, T)

                Beta = self.solve(Xi, Yi, self.lambd, Xi_test, Yi_test)#opts['lambda'])
            else:
                Beta = self.solve(Xi, Yi, self.lambd)

            #Beta = self.solve_with_falkon(Xi[:,:-1], Yi)
            # Remove outliers with the highest loss and retrain
            #Beta = self.solve_with_lasso_sklearn(Xi[:,:-1], Yi)

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
            """ TODO was
            for elem in models[i - 1]['Beta']:
                mean_losses = torch.cat((mean_losses, torch.mean(models[i - 1]['Beta'][elem]['losses'], dim=0, keepdim=True)))
            """
            # TODO uncomment later when training with linear regressors
            for elem in models[i]['Beta']:
                mean_losses = torch.cat((mean_losses, torch.mean(models[i]['Beta'][elem]['losses'], dim=0, keepdim=True)))
                if self.test_losses:
                    mean_losses_test = torch.cat((mean_losses_test, torch.mean(models[i]['Beta'][elem]['losses_test'], dim=0, keepdim=True)))
            print('Mean losses:', mean_losses)
            if self.test_losses:
                print('Mean losses_test:', mean_losses_test)


                

            # break
            #print(Yi.size())
            # --------------------------------------------------------------------
            #models_falkon.append(self.solve_with_falkon(Xi, Yi))
        #quit()
        end_time = time.time()
        print('Time required to train %d regressors: %f seconds.' % (num_clss, end_time - start_time))# TODO was print('Time required to train %d regressors: %f seconds.' % (num_clss - 1, end_time - start_time))
        #torch.save(models, 'models_regressor')
        #torch.save(models_falkon, 'regressors_falkon')
        #quit()
        return models


    def solve(self, X, y, lmbd, X_test=None, Y_test=None, indices=None):
        #lmbd *= y.size()[0]     # TODO Check if helps
        #print(lmbd)
        if indices is None:
            R = None        
            while R is None:
                try:        # TODO check whether this is a good solution
                    #X_torch = torch.from_numpy(X).to("cuda")
                    #y_torch = torch.from_numpy(y).to("cuda")
                    #start_mult = time.time()
                    X_transposed_X = torch.matmul(torch.t(X), X) + lmbd * torch.eye(X.size()[1]).to("cuda")
                    #print(torch.max(X_transposed_X))
                    #print(X.size(), X_transposed_X.size())
                    #start_cho = time.time()
                    #print('Cho: %f seconds.' % (start_cho - start_mult))        
                    R = torch.cholesky(X_transposed_X)
                except:
                    lmbd *= 10
            print('Lambda:', lmbd)
        #end_cho = time.time()
        #print('Cho: %f seconds.' % (end_cho - start_cho))
        to_return = {}
        X_original = X.clone()
        y_original = y.clone()
        for i in range(0, 4):
            if indices is not None:
                X = X_original[indices[i]]
                y = y_original[indices[i]]
                R = None        
                while R is None:
                    try:        # TODO check whether this is a good solution
                        #X_torch = torch.from_numpy(X).to("cuda")
                        #y_torch = torch.from_numpy(y).to("cuda")
                        #start_mult = time.time()
                        X_transposed_X = torch.matmul(torch.t(X), X) + lmbd * torch.eye(X.size()[1]).to("cuda")
                        #print(torch.max(X_transposed_X))
                        #print(X.size(), X_transposed_X.size())
                        #start_cho = time.time()
                        #print('Cho: %f seconds.' % (start_cho - start_mult))        
                        R = torch.cholesky(X_transposed_X)
                    except:
                        lmbd *= 10
                print('Lambda:', lmbd)
            y_torch_i = y[:, i]
            #torch.matmul(torch.t(X_torch), y_torch_i)
            z = torch.triangular_solve(torch.matmul(torch.t(X), y_torch_i).view(X.size()[1], 1), R, upper=False).solution
            #end_ls1 = time.time()
            #print('LS1: %f seconds.' % (end_ls1 - end_cho))
            w = torch.triangular_solve(z, torch.t(R)).solution.view(X.size()[1])
            #end_ls2 = time.time()
            #print('LS2: %f seconds.' % (end_ls2 - end_ls1))
            losses = 0.5 * torch.pow((torch.matmul(X, w) - y_torch_i), 2)
            #print(torch.matmul(X, w), y_torch_i.size(), 'here')
            #losses = losses.cpu().numpy()
            #w = w.cpu().numpy()
            to_return[str(i)] = {'weights': w,
                                 'losses': losses}
            if X_test is not None:
                losses_test = 0.5 * torch.pow((torch.matmul(X_test, w) - Y_test[:, i]), 2)
                to_return[str(i)]['losses_test'] = losses_test 
        return to_return

    def solve_with_falkon(self, X, y):#, lmdb, sigma):
        
        cfg_online_path = '/home/IIT.LOCAL/fceola/workspace/ws_mask/python-online-detection/experiments/Configs/config_federico_server_classifier.yaml'

        # Region Refiner initialization
        regressor = falkon.FALKONWrapper(cfg_path=cfg_online_path)
        #for i in range(5, 100, 5):
        #print(self.sigma, self.lambd)
        model = regressor.train(X.to('cpu'), y.to('cpu'), sigma=self.sigma, lam=self.lambd)
        losses = 0.5 * torch.pow((model.predict(X.to('cpu')).to('cuda') - y), 2)

        #model = regressor.train(X, y, sigma=self.sigma, lam=self.lambd)
        #losses = 0.5 * torch.pow((model.predict(X) - y), 2)
        mean_losses = torch.mean(losses, dim =0)
        print('Falkon losses:', mean_losses)


        return model

    def solve_with_lasso_sklearn(self, X, y):
        clf = linear_model.Lasso(alpha=self.lambd)
        clf.fit(X.cpu().numpy(), y.cpu().numpy())
        return clf


    def solve_with_L1_smooth_sklearn(self, X, y, cls):
        clfs = []
        for i in range(4):
            clf = linear_model.SGDRegressor(loss='huber', alpha=self.lambd, epsilon=1.0 / 9)
            clf.fit(X.cpu().numpy(), y[:,i].cpu().numpy(), coef_init=torch.t(self.pretrained_reg_weights[cls*4+i, :]).numpy(), intercept_init=np.array(self.pretrained_reg_bias[cls+i]))
            clfs.append(clf)
        return clfs


