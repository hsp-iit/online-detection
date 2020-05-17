import sys
import os
basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir)))
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir, '..', '..', '..', 'external', 'FALKON-master')))
from falkon import Falkon, kernels
import ClassifierAbstract as ca


class RegionClassifierAbstract(ca.ClassifierAbstract):
    def __init__(self):
        pass

    def train(self, dataset, opts):
        kernel = None
        if opts.kernel_type == 'gauss':
            kernel = kernels.GaussianKernel(sigma=opts.sigma)
        else:
            print('Kernel type: %s unknown'.format(opts.kernel_type))

        model = None
        if kernel is not None:
            model = Falkon(
                        kernel=kernel,
                        la=opts.la,
                        M=opts.M,
                    )
        else:
            print('Kernel is None in trainRegionClassifier function')
            sys.exit(0)

        if model is not None:
            model.fit(dataset['x_train'], dataset['y_train'])
        else:
            print('Model is None in trainRegionClassifier function')
            sys.exit(0)

        return model

    def predict(self, dataset, model, mode='test'):
        predictions = None
        if mode == 'test':
            predictions = model.predict(dataset['x_test'], dataset['y_test'])
        elif mode == 'val':
            predictions = model.predict(dataset['x_val'], dataset['y_val'])
        else:
            print('Unknown modality in predict function. Returning None predictions')

        return predictions

    def test(self):
        pass

