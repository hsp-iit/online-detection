import sys
import os

basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir)))
sys.path.append(os.path.abspath(os.path.join(basedir, '..', '..')))

import PositivesSelectorAbstract as nsA
import h5py
import numpy as np
from py_od_utils import loadFeature, getFeatPath
import yaml
import torch


class PositivesGTSelector(nsA.NegativeSelectorAbstract):
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

    def selectPositives(self, feat_type='h5'):
        feat_path = os.path.join(basedir, '..', '..', '..', 'Data', 'feat_cache', self.feature_folder)
        positives_file = os.path.join(feat_path, 'positives')
        try:
            if feat_type == 'mat':
                mat_positives = h5py.File(positives_file, 'r')
                X_pos = mat_positives['X_pos']
                positives_torch = []
                for i in range(self.num_classes-1):
                    positives_torch.append(mat_positives[X_pos[0, i]][()].transpose())
            elif feat_type == 'h5':
                positives_dataset = h5py.File(positives_file, 'r')['list']
                positives_torch = []
                for i in range(self.num_classes-1):
                    positives_torch.append(torch.tensor(np.asfortranarray(np.array(positives_dataset[str(i)]))))
            else:
                print('Unrecognized type of feature file')
                positives_torch = None
        except:
            with open(self.train_imset, 'r') as f:
                path_list = f.readlines()
            feat_path = os.path.join(basedir, '..', '..', '..', 'Data', 'feat_cache', self.feature_folder, 'trainval')
            positives = []
            for i in range(len(path_list)):
                l = loadFeature(feat_path, path_list[i].rstrip())
                for c in range(self.num_classes - 1):
                    if len(positives) < c + 1:
                        positives.append([])  # Initialization for class c-th
                    sel = np.where(l['class'] == c + 1)[0]  # TO CHECK BECAUSE OF MATLAB 1
                                                            # INDEXING Moreover class 0 is bkg
                    if len(sel):
                        if len(positives[c]) == 0:
                            positives[c] = l['feat'][sel, :]
                        else:
                            positives[c] = np.vstack((positives[c], l['feat'][sel, :]))
            hf = h5py.File(positives_file, 'w')
            grp = hf.create_group('list')
            for i in range(self.num_classes - 1):
                grp.create_dataset(str(i), data=positives[i])
            hf.close()

            positives_torch = []
            for i in range(self.num_classes - 1):
                # for j in range(self.iterations):
                positives_torch.append(torch.tensor(positives[i].reshape(positives[i].shape[0], positives[i].shape[1]), device='cuda'))

        return positives_torch
