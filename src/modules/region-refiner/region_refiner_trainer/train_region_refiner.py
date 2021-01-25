import numpy as np
import os
import time
import sys
import torch

basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir)))
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir, os.path.pardir)))
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir, os.path.pardir, os.path.pardir)))


class RegionRefinerTrainer():
    def __init__(self, cfg, lmbd, is_rpn):
        self.cfg = cfg
        self.lambd = lmbd
        self.percentile = 0
        self.COXY = None
        self.is_rpn = is_rpn

    def __call__(self, COXY, output_dir=None):
        self.COXY = COXY
        models = self.train(output_dir=output_dir)
        return models

    def train(self, output_dir=None):
        chosen_classes = self.cfg['CHOSEN_CLASSES']
        start_index = 1

        if self.is_rpn:
            start_index = 0
        opts = self.cfg['REGION_REFINER']['opts']

        num_clss = len(chosen_classes)

        models = np.empty((0))

        start_time = time.time()
        for i in range(start_index, num_clss):
            print('Training regressor for class %s (%d/%d)' % (chosen_classes[i], i, num_clss - 1))
            # Compute indices where bboxes of class i overlap with the ground truth
            I = self.COXY['C'] == i
            I = torch.where(I == True)[0]
            print('Training with %i examples' %len(I))
            if len(I) == 0:
                models = np.append(models, {'mu': None,
                                            'T': None,
                                            'T_inv': None,
                                            'Beta': None
                                            })
                print('No indices for class %s' % (chosen_classes[i]))
                continue
            # Extract the corresponding values in the X matrix
            # Transpose is used to set the number of features as the number of columns (instead of the number of rows)
            Xi = self.COXY['X'][I].type(torch.float64)
            Yi = self.COXY['Y'][I].type(torch.float64)
            # Add bias values to Xi
            bias = torch.ones((Xi.size()[0], 1), dtype=torch.float64, device=Xi.device)
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
                'mu': mu.to("cuda").type(torch.float32),
                'T': T.to("cuda").type(torch.float32),
                'T_inv': T_inv.to("cuda").type(torch.float32),
                'Beta': Beta
            })

            mean_losses = torch.empty(0, device=Xi.device, dtype=torch.float32)

            for elem in models[i - start_index]['Beta']:
                mean_losses = torch.cat((mean_losses, torch.mean(models[i - start_index]['Beta'][elem]['losses'], dim=0, keepdim=True)))
            print('Mean losses:', mean_losses)

        end_time = time.time()
        training_time = end_time - start_time
        print('Time required to train %d regressors: %f seconds.' % (num_clss-1, training_time))

        if output_dir and self.is_rpn:
            with open(os.path.join(output_dir, "result.txt"), "a") as fid:
                fid.write("RPN's Online Region Refiner training time: {}min:{}s \n".format(int(training_time/60), round(training_time%60)))
        elif output_dir and not self.is_rpn:
            with open(os.path.join(output_dir, "result.txt"), "a") as fid:
                fid.write("Detector's Online Region Refiner training time: {}min:{}s \n \n".format(int(training_time/60), round(training_time%60)))

        return models


    def solve(self, X, y, lmbd, X_test=None, Y_test=None, indices=None):
        if indices is None:
            X_transposed_X = torch.matmul(torch.t(X), X) + lmbd * torch.eye(X.size()[1], device=X.device, dtype=torch.float64)
            R = torch.cholesky(X_transposed_X)
            """
            X_transposed_X_cuda = torch.matmul(torch.t(X.type(torch.float64)), X.type(torch.float64)) + lmbd * torch.eye(X.size()[1], device=X.device).type(torch.float64)
            X_transposed_X = torch.matmul(torch.t(X.type(torch.float64).to('cpu')), X.type(torch.float64).to('cpu')) + lmbd * torch.eye(X.size()[1], device='cpu').type(torch.float64)
            X_transposed_X = X_transposed_X.to('cuda')
            #X_transposed_X = torch.mm(torch.t(X), X) + lmbd * torch.eye(X.size()[1], device=X.device)

            print(torch.where(torch.eig(X_transposed_X)[0][:,0]<0)[0])
            print(torch.eig(X_transposed_X)[0][torch.where(torch.eig(X_transposed_X)[0][:,0]<0)[0]])
            print(X_transposed_X-X_transposed_X_cuda)
            print(torch.max(torch.abs(X_transposed_X-X_transposed_X_cuda)))
            print(torch.max(torch.abs(X-torch.clone(X).to('cpu').to('cuda'))))
            R = torch.cholesky(X_transposed_X.type(torch.float32))
            """
            """
            try:
                X_transposed_X = torch.matmul(torch.t(X), X) + lmbd * torch.eye(X.size()[1], device=X.device)
                R = torch.cholesky(X_transposed_X)
            except:
                X_transposed_X = torch.matmul(torch.t(X).to('cpu'), X.to('cpu')) + lmbd * torch.eye(X.size()[1], device='cpu')
                R = torch.cholesky(X_transposed_X).to('cuda')
            """

        to_return = {}
        X_original = X.clone()
        y_original = y.clone()
        for i in range(0, 4):
            if indices is not None:
                X = X_original[indices[i]]
                y = y_original[indices[i]]
                X_transposed_X = torch.matmul(torch.t(X), X) + lmbd * torch.eye(X.size()[1], device=X.device, dtype=torch.float64)
                R = torch.cholesky(X_transposed_X)
            y_torch_i = y[:, i]
            z = torch.triangular_solve(torch.matmul(torch.t(X), y_torch_i).view(X.size()[1], 1), R, upper=False).solution
            w = torch.triangular_solve(z, torch.t(R)).solution.view(X.size()[1])
            losses = 0.5 * torch.pow((torch.matmul(X, w) - y_torch_i), 2)
            to_return[str(i)] = {'weights': w.to('cuda').type(torch.float32),
                                 'losses': losses.type(torch.float32)}
        return to_return


