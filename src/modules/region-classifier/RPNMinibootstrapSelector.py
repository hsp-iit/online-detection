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
import torch


class RPNMinibootstrapSelector(nsA.NegativeSelectorAbstract):
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
        self.all_negatives = None

    def selectNegatives(self, neg_ovr_thresh=0.3, max_regions=300, feat_type='h5'):
        print('Selecting negatives from the {} dataset.'.format(self.train_imset))
        feat_path = os.path.join(basedir, '..', '..', '..', 'Data', 'feat_cache', self.feature_folder)
        negatives_file = os.path.join(feat_path, 'RPN_negatives{}x{}'.format(self.iterations,
                                                                         self.batch_size))
        try:
            if feat_type == 'mat':
                negatives_file = negatives_file + '.mat'
                print('Trying to load negative samples from {}'.format(negatives_file))
                mat_negatives = h5py.File(negatives_file, 'r')
                X_neg = mat_negatives['X_neg']
                negatives_torch = []
                for i in range(self.num_classes - 1):
                    tmp = []
                    for j in range(self.iterations):
                        tmp.append(torch.tensor(mat_negatives[mat_negatives[X_neg[0, i]][0, j]][()].transpose()))
                    negatives_torch.append(tmp)
            elif feat_type == 'h5':
                negatives_dataset = h5py.File(negatives_file, 'r')['list']
                negatives_torch = []
                for i in range(self.num_classes - 1):
                    tmp = []
                    # for j in range(self.iterations):
                    #     tmp.append(negatives_dataset[i][j])
                    cls_neg = negatives_dataset[str(i)]
                    for j in range(self.iterations):
                        tmp.append(torch.tensor(np.asfortranarray(np.array(cls_neg[str(j)]))))
                    negatives_torch.append(tmp)
                # negatives_dataset.close()
            else:
                print('Unrecognized type of feature file')
                negatives_torch = None
        except:
            print('Loading failed. Starting negatives computation')

            if self.all_negatives is None:
                print("Error: all Negatives is None. Please call the function setAllNegatives before performing"
                      "negatives selection for Minibootstrap")
                sys.exit(0)

            # ############# PERFORM NEGATIVES SELECTION HERE ##############
            negatives = self.all_negatives
            # #############################################################

            negatives_torch = []
            for i in range(self.num_classes - 1):
                for j in range(self.iterations):
                    negatives_torch.append(torch.tensor(negatives[i][j], device='cuda'))

            hf = h5py.File(negatives_file, 'w')
            grp = hf.create_group('list')
            for i in range(self.num_classes-1):
                grpp = grp.create_group(str(i))
                for j in range(len(negatives[i])):
                    grpp.create_dataset(str(j), data=negatives[i][j])
            hf.close()

        return negatives_torch

    def setAllNegatives(self, all_negatives) -> None:
        if type(all_negatives) == str:
            self.all_negatives = torch.load(all_negatives)
        else:
            self.all_negatives = all_negatives

    def setIterations(self, iterations) -> None:
        self.iterations = iterations

    def setBatchSize(self, batch_size) -> None:
        self.batch_size = batch_size
