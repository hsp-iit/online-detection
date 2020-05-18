import sys
import os

basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir)))
import NegativeSelectorAbstract as nsA
import h5py


class MinibootstrapSelector(nsA.NegativeSelectorAbstract):
    def __init__(self, iterations, batch_size):
        self.iterations = iterations
        self.batch_size = batch_size

    def selectNegatives(self, dataset, experiment_name, opts):
        feat_path = os.path.join(basedir, '..', '..', '..', 'Data', 'feat_cache', experiment_name)
        negatives_file = os.path.join(feat_path, experiment_name + '_negatives{}x{}.mat'.format(self.iterations, self.batch_size))
        try:
            mat_negatives = h5py.File(negatives_file, 'r')
            X_neg = mat_negatives['X_neg']
            mat_negatives[mat_negatives[X_neg[0, 0]][0, 0]] # Shape: (2048, 1994)
            negatives = []
            for i in range(opts['num_classes'] ):
                tmp = []
                for j in range(self.iterations):
                    tmp.append(mat_negatives[mat_negatives[X_neg[0, i]][0, j]][()].transpose()) # Shape: (2048, 1994)
                negatives.append(tmp)

        except:
            print('To implement selectNegatives in MinibootstrapSelector')
            negatives = None

        return negatives

    def setIterations(self, iterations) -> None:
        self.iterations = iterations

    def setBatchSize(self, batch_size) -> None:
        self.batch_size = batch_size
