import sys
import os
basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir)))
from falkon import Falkon, kernels
import ClassifierAbstract as ca
import torch


class FALKONWrapper(ca.ClassifierAbstract):
    def __init__(self):
        pass

    def train(self, X_np, y_np, opts):
        kernel = None
        if opts['kernel_type'] == 'gauss':
            kernel = kernels.GaussianKernel(sigma=opts['sigma'])
        else:
            print('Kernel type: %s unknown'.format(opts['kernel_type']))

        model = None
        if kernel is not None:
            model = Falkon(
                        kernel=kernel,
                        la=opts['lambda'],
                        M=opts['M'],
                        use_cpu=True
                        # use_display_gpu=True,
                        # gpu_use_processes=False,
                        # inter_type=torch.float32,
                        # final_type=torch.float32
                    )
        else:
            print('Kernel is None in trainRegionClassifier function')
            sys.exit(0)

        if model is not None:
            X = torch.from_numpy(X_np)
            y = torch.from_numpy(y_np).to(torch.float32)
            model.fit(X, y)
        else:
            print('Model is None in trainRegionClassifier function')
            sys.exit(0)

        return model

    def predict(self, model, X_np, y=None):
        X = torch.from_numpy(X_np)
        if y is not None:
            predictions = model.predict(X, y)
        else:
            predictions = model.predict(X)

        return predictions

    def test(self):
        pass

