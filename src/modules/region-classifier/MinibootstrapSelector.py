import sys
import os

basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir)))
sys.path.append(os.path.abspath(os.path.join(basedir, '..', '..')))

import NegativeSelectorAbstract as nsA
import h5py
import numpy as np
from utils import loadFeature
import torch


class MinibootstrapSelector(nsA.NegativeSelectorAbstract):
    def __init__(self, iterations, batch_size, neg_easy_thresh, neg_hard_thresh):
        self.iterations = iterations
        self.batch_size = batch_size
        self.neg_easy_thresh = neg_easy_thresh
        self.neg_hard_thresh = neg_hard_thresh

    def selectNegatives(self, imset_path, experiment_name, opts, neg_ovr_thresh=0.3, max_regions=300, feat_type='mat'):
        feat_path = os.path.join(basedir, '..', '..', '..', 'Data', 'feat_cache', experiment_name)
        negatives_file = os.path.join(feat_path, experiment_name + '_negatives{}x{}'.format(self.iterations,
                                                                                            self.batch_size))
        try:
            if feat_type == 'mat':
                negatives_file = negatives_file + '.mat'
                mat_negatives = h5py.File(negatives_file, 'r')
                X_neg = mat_negatives['X_neg']
                negatives = []
                for i in range(opts['num_classes'] -1):
                    tmp = []
                    for j in range(self.iterations):
                        tmp.append(mat_negatives[mat_negatives[X_neg[0, i]][0, j]][()].transpose())
                    negatives.append(tmp)
            else:
                print('Unrecognized type of feature file')
                negatives = None
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
            # --- Concatenate the chosen regions to the current batch, accounting for all the special cases
            with open(imset_path, 'r') as f:
                path_list = f.readlines()
            feat_path = os.path.join(basedir, '..', '..', '..', 'Data', 'feat_cache', experiment_name)

            # Number of regions to keep from image for each class
            keep_from_image = int(np.ceil((self.batch_size*self.iterations)/len(path_list)))
            keep_from_image = min(max(keep_from_image, 1), max_regions)

            # Vector to track done batches and classes
            keep_doing = np.ones((opts['num_classes']-1, self.iterations))

            negatives = []
            for i in range(len(path_list)):
                l = loadFeature(feat_path, path_list[i].rstrip(), feat_type)
                print('{}/{} : {}'.format(i, len(path_list), path_list[i].rstrip()))
                if l is not None:
                    for c in range(opts['num_classes']-1):
                        if sum(keep_doing[c, :]) > 0:
                            I = np.nonzero(l['overlap'][:, c] < neg_ovr_thresh)[0]
                            idx = np.random.choice(I, min(len(I), keep_from_image), replace=False)
                            keep_per_batch = np.ceil(len(idx)/self.iterations)
                            kept = 0
                            for b in range(self.iterations):
                                if kept >= len(idx):
                                    break
                                if len(negatives) < c + 1:
                                    negatives.append([])
                                if len(negatives[c]) < b + 1:
                                    negatives[c].append([])

                                if len(negatives[c][b]) < self.batch_size:
                                    end_interval = int(kept + min(keep_per_batch, self.batch_size - len(negatives[c][b]),
                                                                  len(idx) - kept))
                                    new_idx = idx[np.arange(kept, end_interval)]

                                    if len(negatives[c][b]) == 0:
                                        negatives[c][b] = l['feat'][new_idx, :]
                                    else:
                                        negatives[c][b] = np.vstack((negatives[c][b], l['feat'][new_idx, :]))
                                    kept = end_interval
                                else:
                                    keep_doing[c, b] = 0
            if feat_type == 'mat':
                print('Unimplemented negatives saving')
        torch.save(negatives, negatives_file)

        return negatives

    def setIterations(self, iterations) -> None:
        self.iterations = iterations

    def setBatchSize(self, batch_size) -> None:
        self.batch_size = batch_size
