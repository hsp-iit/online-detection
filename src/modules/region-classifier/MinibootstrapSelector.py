import sys
import os

basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir)))
import NegativeSelectorAbstract as nsA
from scipy.io import loadmat


class MinibootstrapSelector(nsA.NegativeSelectorAbstract):
    def __init__(self, iterations, batch_size):
        self.iterations = iterations
        self.batch_size = batch_size

    def selectNegatives(self, dataset):
        negatives_file = self.experiment_name + '_negatives%fx%f'.format(self.iterations, self.batch_size)
        negatives_file = self.experiment_name + '_negatives%fx%f'.format(self.iterations, self.batch_size)
        try:
            negatives = loadmat(negatives_file)
        except:
            print('To implement selectNegatives in MinibootstrapSelector')
            negatives = None
        return negatives

    def setIterations(self, iterations) -> None:
        self.iterations = iterations

    def setBatchSize(self, batch_size) -> None:
        self.batch_size = batch_size
