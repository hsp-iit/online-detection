import sys
import os

basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir)))
sys.path.append(os.path.abspath(os.path.join(basedir, '..', '..')))

import PositivesSelectorAbstract as psA
import h5py
import numpy as np
from py_od_utils import loadFeature, getFeatPath
import yaml
import torch


class RPNPositivesSelector(psA.PositivesSelectorAbstract):
    def __init__(self, cfg_path):
        cfg = yaml.load(open(cfg_path), Loader=yaml.FullLoader)
        self.iterations = cfg['ONLINE_REGION_CLASSIFIER']['MINIBOOTSTRAP']['ITERATIONS']
        self.batch_size = cfg['ONLINE_REGION_CLASSIFIER']['MINIBOOTSTRAP']['BATCH_SIZE']
        self.neg_easy_thresh = cfg['ONLINE_REGION_CLASSIFIER']['MINIBOOTSTRAP']['EASY_THRESH']
        self.neg_hard_thresh = cfg['ONLINE_REGION_CLASSIFIER']['MINIBOOTSTRAP']['HARD_THRESH']
        self.num_classes = cfg['NUM_CLASSES_RPN']
        self.experiment_name = cfg['EXPERIMENT_NAME']
        self.train_imset = cfg['DATASET']['TARGET_TASK']['TRAIN_IMSET']
        self.feature_folder = getFeatPath(cfg)

    def selectPositives(self, feat_type='h5'):
        feat_path = os.path.join(basedir, '..', '..', '..', 'Data', 'feat_cache', self.feature_folder)
        positives_file = os.path.join(feat_path, 'RPN_positives')
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
                for i in range(self.num_classes):
                    positives_torch.append(torch.tensor(np.asfortranarray(np.array(positives_dataset[str(i)]))))
            else:
                print('Unrecognized type of feature file')
                positives_torch = None
        except:
            print('Failed to load positives features. Extracting positives from the dataset..')
            with open(self.train_imset, 'r') as f:
                path_list = f.readlines()
            feat_path = os.path.join(basedir, '..', '..', '..', 'Data', 'feat_cache', self.feature_folder, 'RPN_trainval')
            positives_torch = []
            for i in range(len(path_list)):
                l = loadFeature(feat_path, path_list[i].rstrip(), type='torch')
                for c in range(self.num_classes):
                    if len(positives_torch) < c + 1:
                        positives_torch.append([])  # Initialization for class c-th
                    sel = torch.where((l.get_field('classifier') == c) & (l.get_field('overlap') >= 0.3))[0]

                    if len(sel):
                        if len(positives_torch[c]) == 0:
                            positives_torch[c] = l.get_field('features')[sel, :]
                        else:
                            positives_torch[c] = torch.cat((positives_torch[c], l.get_field('features')[sel, :]), 0)

            hf = h5py.File(positives_file, 'w')
            grp = hf.create_group('list')
            for i in range(self.num_classes):
                if len(positives_torch[i]):
                    grp.create_dataset(str(i), data=np.array(positives_torch[i].to('cpu')))
                else:
                    grp.create_group(str(i))
            hf.close()

        return positives_torch

    def get_num_classes(self):
        return self.num_classes + 1
