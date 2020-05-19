import sys
import os

basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir)))
import NegativeSelectorAbstract as nsA
import h5py
import numpy as np


class MinibootstrapSelector(nsA.NegativeSelectorAbstract):
    def __init__(self, iterations, batch_size, neg_easy_thresh, neg_hard_thresh):
        self.iterations = iterations
        self.batch_size = batch_size
        self.neg_easy_thresh = neg_easy_thresh
        self.neg_hard_thresh = neg_hard_thresh

    def selectNegatives(self, imset_path, experiment_name, opts, neg_ovr_thresh=0.3):
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
            # Initialize variables and parameters like neg_ovr_thresh=0.3
            # Compute how many to pick from each image for each class
            # For each image in the train set
            # - Load the feature file
            # - For each class
            # -- Find regions that match neg_ovr_thresh condition and pick randomly the decided number
            # -- Set keep_per_batch
            # -- For each batch
            # --- Concatenate the chosen regions to the current batch accounting for all the special cases
            negatives = []
            with open(imset_path, 'r') as f:
                path_list = f.readlines()
            feat_path = os.path.join(basedir, '..', '..', '..', 'Data', 'feat_cache', experiment_name)
            for i in range(len(path_list)):
                l = self.loadFeature(feat_path, path_list[i])
                if l is not None:
                    for c in range(opts['num_classes']):
                        I = np.nonzero(l['overlap'][:, c] > neg_ovr_thresh)
                        # idx = np.random.randint() ------------- HERE

            negatives = None

        return negatives

    def setIterations(self, iterations) -> None:
        self.iterations = iterations

    def setBatchSize(self, batch_size) -> None:
        self.batch_size = batch_size
