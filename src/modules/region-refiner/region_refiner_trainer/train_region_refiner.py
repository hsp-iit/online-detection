import h5py
import numpy as np
import os
import time
import warnings
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

        opts = {
            'min_overlap': 0.6,
            'binarize': False,  # TODO not sure that this is required in python. Datatype is already float32
            'lambda': 1000,
            'robust': 0
        }

        # cache_dir = 'bbox_reg/'
        # if not os.path.exists(cache_dir):
        #    os.mkdir(cache_dir)
        clss = imdb['classes']
        num_clss = len(clss)
        bbox_model_suffix = '_first_test'

        # Set ridge regression method
        method = 'ridge_reg_chol'
        # method = 'ridge_reg_inv'
        # method = 'ls_mldivide'
        models = np.array([])
        # print(models)

        start_time = time.time()
        for i in range(1, num_clss):
            print('Training regressor for class %s (%d/%d)' % (imdb['classes'][i], i, num_clss - 1))
            # Compute indices where bboxes of class i overlap with the ground truth
            I = np.logical_and(COXY['O'] > opts['min_overlap'], COXY['C'] == i)
            I = np.squeeze(np.where(I == True))
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
            Xi = COXY['X'][I]#[:, I].T
            # TODO check binarize in line 36 of the matlab code
            Yi = COXY['Y'][I]#[:, I].T
            # TODO check if Oi and Ci computations are required
            # Add bias values to Xi
            bias = np.ones((Xi.shape[0], 1), dtype=np.float32)
            Xi = np.append(Xi, bias, axis=1)

            # Center and decorrelate targets
            mu = np.mean(Yi, axis=0)
            Yi -= np.broadcast_arrays(Yi, mu)[1]  # Resize mu as Yi with broadcast_arrays in [1], subtract mu to Yi
            # TODO understand where this formula comes from
            S = np.matmul(Yi.T, Yi) / Yi.shape[0]
            D, W = np.linalg.eig(S)  # D in python is a vector, in matlab is a diagonal matrix
            T = np.matmul(np.matmul(W, np.diag(1 / np.sqrt(D + 0.001))), W.T)
            T_inv = np.matmul(np.matmul(W, np.diag(np.sqrt(D + 0.001))), W.T)
            Yi = np.matmul(Yi, T)
            print(Yi.shape)

            Beta = np.array([
                self.solve_robust(Xi, Yi[:, 0], opts['lambda'], method, opts['robust']),
                self.solve_robust(Xi, Yi[:, 1], opts['lambda'], method, opts['robust']),
                self.solve_robust(Xi, Yi[:, 2], opts['lambda'], method, opts['robust']),
                self.solve_robust(Xi, Yi[:, 3], opts['lambda'], method, opts['robust'])
            ])
            models = np.append(models, {
                'mu': mu,
                'T': T,
                'T_inv': T_inv,
                'Beta': Beta
            })

            mean_losses = np.array([])
            mean_losses_robust = np.array([])
            for elem in models[i - 1]['Beta']:
                mean_losses = np.append(mean_losses, np.mean(elem['w_losses'][1]))
                if 'w_losses_robust' in elem:
                    mean_losses_robust = np.append(mean_losses_robust, np.mean(elem['w_losses_robust'][1]))
            print('Mean losses:', mean_losses)
            if len(mean_losses_robust) > 0:
                print('Mean losses (robust):', mean_losses_robust)

            # break
        end_time = time.time()
        print('Time required to train %d regressors: %f seconds.' % (num_clss - 1, end_time - start_time))

        return

    def solve_robust(self, X, y, lmbd, method, qtile):
        # w is x in Matlab, X is A in Matlab
        w, losses = self.solve(X, y, lmbd, method)
        w_losses_dict = {
            'w_losses': np.array([w, losses])
        }
        if qtile > 0:
            thresh = np.quantile(losses, 1 - qtile)
            I = np.where(losses < thresh)
            w_robust, losses_robust = self.solve(X[I], y[I], lmbd, method)
            w_losses_dict['w_losses_robust'] = np.array([w_robust, losses_robust])
        return w_losses_dict

    def solve(self, X, y, lmbd, method):
        # activate = torch.rand(1).to("cuda")
        X_torch = torch.from_numpy(X).to("cuda")
        print(X_torch.dtype)
        y_torch = torch.from_numpy(y).to("cuda")
        print(y_torch.dtype)

        #print(y_torch.shape)
        if method == 'ridge_reg_chol':
            # solve for x in min_w ||Xw - y||^ 2 + lambda * ||w|| ^ 2
            # solve (X_transposed*X + lambda*I)*w = X_transposed*y using cholesky factorization
            # R*R_transposed = (X_transposed*X + lambda*I)
            # R*z = X_transposed*y: solve for z => R*R_transposed*w = R*z  = > R_transposed*w = z
            # R_transposed*w = z                        : solve for w
            start_mult = time.time()
            # R = np.linalg.cholesky(np.matmul(X.T, X) + lmbd*np.eye(X.shape[1]))
            # X_transposed_X = np.matmul(X.T, X) + lmbd*np.eye(X.shape[1])
            # X_transposed_X = np.matmul(X.T, X)
            # np.fill_diagonal(X_transposed_X, X_transposed_X.diagonal()+lmbd)
            X_transposed_X = torch.matmul(torch.t(X_torch), X_torch) + lmbd * torch.eye(2049).to("cuda")
            # X_transposed_X.fill_diagonal_(torch.diagonal(X_transposed_X)+lmbd)
            start_cho = time.time()
            # tens = torch.from_numpy(X_transposed_X).to("cuda")
            print('Cho: %f seconds.' % (start_cho - start_mult))
            # R = linalg.cholesky(X_transposed_X, lower=True)
            R = torch.cholesky(X_transposed_X)
            # R = R.T
            # print(Rnp)
            # print(R)
            end_cho = time.time()
            print('Cho: %f seconds.' % (end_cho - start_cho))
            # z = linalg.solve_triangular(R, np.matmul(X.T, y), lower=True)
            # print(torch.matmul(torch.t(X_torch), y_torch).view(2049, 1))
            # print(R)
            # quit()
            print(R.size())
            torch.matmul(torch.t(X_torch), y_torch)
            z = torch.triangular_solve(torch.matmul(torch.t(X_torch), y_torch).view(2049, 1), R, upper=False).solution
            end_ls1 = time.time()
            print('LS1: %f seconds.' % (end_ls1 - end_cho))
            # w = linalg.solve_triangular(R.T, z)
            w = torch.triangular_solve(z, torch.t(R)).solution.view(2049)
            # inv_R_transposed_R = np.linalg.inv(np.matmul(R, R.T))
            # w = np.matmul(inv_R_transposed_R, np.matmul(R, z))
            end_ls2 = time.time()
            print('LS2: %f seconds.' % (end_ls2 - end_ls1))
            # Using scikit-learn, different results
            # reg = linear_model.Ridge(alpha=lmbd, solver='cholesky').fit(X, y)
            # w = reg.coef_
        elif method == 'ridge_reg_inv':
            # solve for x in min_w ||Xw - y||^ 2 + lambda * ||w|| ^ 2
            # w = (X_transposed*X + lambda*I)^-1 * X_transposed*y
            w = np.matmul(np.linalg.inv(np.matmul(X.T, X) + lmbd * np.eye(X.shape[1])), np.matmul(X.T, y))
        # elif method == 'ls_mldivide':
        #    # solve for x in min_w ||Xw - y||^ 2
        #    if lmbd > 0:
        #        warnings.warn('Ignoring lambda. No regularization used.')
        #    w = np.linalg.lstsq(X, y)[0]    #TODO check these results. From matlab seems that X doesn't have max rank
        #    #reg = LinearRegression().fit(X, y)
        #    #w = reg.coef
        else:
            raise ValueError('Invalid method name.')
        # losses = 0.5*(np.power((np.matmul(X, w.cpu().numpy()) - y), 2))

        losses = 0.5 * torch.pow((torch.matmul(X_torch, w) - y_torch), 2)
        losses = losses.cpu().numpy()
        w = w.cpu().numpy()
        return w, losses