import sys
import os

basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir)))
sys.path.append(os.path.abspath(os.path.join(basedir, '..', '..')))

import NegativeSelectorAbstract as nsA
import h5py
import numpy as np
from py_od_utils import loadFeature, getFeatPath
import yaml


class MinibootstrapSelector(nsA.NegativeSelectorAbstract):
    def __init__(self, cfg_path):
        cfg = yaml.load(open(cfg_path), Loader=yaml.FullLoader)
        self.iterations = cfg['ONLINE_REGION_CLASSIFIER']['MINIBOOTSTRAP']['ITERATIONS']
        self.batch_size = cfg['ONLINE_REGION_CLASSIFIER']['MINIBOOTSTRAP']['BATCH_SIZE']
        self.neg_easy_thresh = cfg['ONLINE_REGION_CLASSIFIER']['MINIBOOTSTRAP']['EASY_THRESH']
        self.neg_hard_thresh = cfg['ONLINE_REGION_CLASSIFIER']['MINIBOOTSTRAP']['HARD_THRESH']
        self.num_classes = cfg['NUM_CLASSES']
        self.experiment_name = cfg['EXPERIMENT_NAME']
        self.train_imset = cfg['DATASET']['TARGET_TASK']['TRAIN_IMSET']
        self.feature_folder = getFeatPath(cfg)

    def selectNegatives(self, neg_ovr_thresh=0.3, max_regions=300, feat_type='h5'):
        print('Selecting negatives from the {} dataset.'.format(self.train_imset))
        feat_path = os.path.join(basedir, '..', '..', '..', 'Data', 'feat_cache', self.feature_folder)
        negatives_file = os.path.join(feat_path, 'negatives{}x{}'.format(self.iterations,
                                                                         self.batch_size))
        negatives = []
        try:
            if feat_type == 'mat':
                negatives_file = negatives_file + '.mat'
                print('Trying to load negative samples from {}'.format(negatives_file))
                mat_negatives = h5py.File(negatives_file, 'r')
                X_neg = mat_negatives['X_neg']
                negatives = []
                for i in range(self.num_classes - 1):
                    tmp = []
                    for j in range(self.iterations):
                        tmp.append(mat_negatives[mat_negatives[X_neg[0, i]][0, j]][()].transpose())
                    negatives.append(tmp)
            elif feat_type == 'h5':
                negatives_dataset = h5py.File(negatives_file, 'r')['negatives']
                negatives = []
                for i in range(self.num_classes - 1):
                    # tmp = []
                    # for j in range(self.iterations):
                    #     tmp.append(negatives_dataset[i][j])
                    negatives.append(negatives_dataset[i])
            else:
                print('Unrecognized type of feature file')
                negatives = None
        except:
            print('Loading failed. Starting negatives computation')
            with open(self.train_imset, 'r') as f:
                path_list = f.readlines()

            feat_path = os.path.join(basedir, '..', '..', '..', 'Data', 'feat_cache', self.feature_folder, 'trainval')

            # Number of regions to keep from image for each class
            keep_from_image = int(np.ceil((self.batch_size*self.iterations)/len(path_list)))
            keep_from_image = min(max(keep_from_image, 1), max_regions)

            # Vector to track done batches and classes
            keep_doing = np.ones((self.num_classes-1, self.iterations))

            for i in range(len(path_list)):
                if sum(sum(keep_doing)) == 0:
                    break
                l = loadFeature(feat_path, path_list[i].rstrip(), 'mat')
                print('{}/{} : {}'.format(i, len(path_list), path_list[i].rstrip()))
                if l is not None:
                    for c in range(self.num_classes-1):
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

            hf = h5py.File(negatives_file, 'w')
            hf.create_dataset('negatives', data=negatives)
            hf.close()

        return negatives

    def setIterations(self, iterations) -> None:
        self.iterations = iterations

    def setBatchSize(self, batch_size) -> None:
        self.batch_size = batch_size
