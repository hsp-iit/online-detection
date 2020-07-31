import pickle
from scipy.io import loadmat
#import h5py
import numpy as np
import os
import time
#import warnings
from scipy import linalg
import torch
from utils import list_features, features_to_COXY

basedir = os.path.dirname(__file__)
from py_od_utils import getFeatPath

class RegionPredictor():
    def __init__(self, cfg, models=None, boxes=None):
        self.cfg = cfg
        self.features_format = self.cfg['FEATURE_INFO']['FORMAT']
        feature_folder = getFeatPath(self.cfg)
        if 'val' in self.cfg['DATASET']['TARGET_TASK']['TEST_IMSET']:
            feat_path = os.path.join(basedir, '..', '..', '..', '..', 'Data', 'feat_cache', feature_folder, 'train_val')
        else:
            feat_path = os.path.join(basedir, '..', '..', '..', '..', 'Data', 'feat_cache', feature_folder, 'test')
        self.path_to_features = feat_path + '/%s' + self.features_format
        self.path_to_imgset_test = self.cfg['DATASET']['TARGET_TASK']['TRAIN_IMSET']
        self.features_dictionary_test = list_features(self.path_to_imgset_test)
        if models is not None:
            self.models = models
        else:
            try:
                self.models = torch.load(self.cfg['REGION_REFINER']['MODELS_PATH'])
            except:
                print('Failed to load model')
        if boxes is not None:
            self.boxes = boxes
        else:
            try:
                self.boxes = torch.load(self.cfg['REGION_REFINER']['BOXES_PATH'])
            except:
                print('Failed to load boxes')
        try:
            self.stats = torch.load(os.path.join(basedir, '..', '..', '..', '..', 'Data', 'feat_cache', feature_folder) + '/stats')
            for key in self.stats.keys():
                self.stats[key] = self.stats[key].to('cuda')
        except:
            self.stats = None
        self.normalize_features = True


    def __call__(self):
        pred_boxes = self.predict()
        return pred_boxes

    def predict(self):
        chosen_classes = self.cfg['CHOSEN_CLASSES']
        opts = self.cfg['REGION_REFINER']['opts']

        # cache_dir = 'bbox_reg/'
        # if not os.path.exists(cache_dir):
        #    os.mkdir(cache_dir)

        num_clss = len(chosen_classes)
        bbox_model_suffix = '_first_test'

        img_size = self.boxes[0].size
        img_width = img_size[0]
        img_height = img_size[1]

        l = 1
        # Loop on the list of boxlists
        for box_list in self.boxes:
            # TODO evaluate whether to add tic_toc_print
            pth = self.path_to_features % box_list.get_field('name_file')
            #print(pth)
            if '.pkl' in pth:
                with open(pth, 'rb') as f:
                    feat = pickle.load(f)
            elif '.mat' in pth:
                feat = loadmat(pth)



            #refined_boxes = torch.empty((0, len(chosen_classes), 4))
            num_gt = np.sum(feat['class'] > 0)
            feat = torch.tensor(feat['feat'][num_gt:]).to('cuda')
            if self.normalize_features:
                feat = feat - self.stats['mean']
                feat = feat * (20 / self.stats['mean_norm'].item())
            ex_box = box_list.bbox.to('cuda')
            num_boxes = ex_box.size()[0]
            refined_boxes = ex_box
            for j in range(1, len(chosen_classes)):
                weights = self.models[j-1]['Beta']['0']['weights'].view(1,2049)
                
                for k in range(1, 4):
                    weights = torch.cat((weights, self.models[j-1]['Beta'][str(k)]['weights'].view(1,2049)))
                weights = torch.t(weights)
                Y = torch.matmul(feat, weights[:-1])
                #print(Y, Y.size())
                Y += weights[-1]
                #print(Y, Y.size())
                #quit()
                Y = torch.matmul(Y, self.models[j-1]['T_inv'])
                Y += self.models[j-1]['mu']


                dst_ctr_x = Y[:,0]
                dst_ctr_y = Y[:,1]
                dst_scl_x = Y[:,2]
                dst_scl_y = Y[:,3]
    
                src_w = ex_box[:,2] - ex_box[:,0] + np.spacing(1)
                src_h = ex_box[:,3] - ex_box[:,1] + np.spacing(1)
                src_ctr_x = ex_box[:,0] + 0.5 * src_w
                src_ctr_y = ex_box[:,1] + 0.5 * src_h
                pred_ctr_x = (dst_ctr_x * src_w) + src_ctr_x
                pred_ctr_y = (dst_ctr_y * src_h) + src_ctr_y
                pred_w = torch.exp(dst_scl_x) * src_w
                pred_h = torch.exp(dst_scl_y) * src_h
                pred_boxes = torch.cat(((pred_ctr_x - 0.5 * pred_w).view(num_boxes,1), (pred_ctr_y - 0.5 * pred_h).view(num_boxes,1)), dim=1)
                pred_boxes = torch.cat((pred_boxes, (pred_ctr_x + 0.5 * pred_w).view(num_boxes,1)),dim=1)
                pred_boxes = torch.cat((pred_boxes, (pred_ctr_y + 0.5 * pred_h).view(num_boxes,1)),dim=1)


                if '.pkl' in pth:
                    pred_boxes[:, 0] = torch.max(pred_boxes[:, 0], torch.zeros(pred_boxes[:,0].size(), device='cuda'))
                    pred_boxes[:, 1] = torch.max(pred_boxes[:, 1], torch.zeros(pred_boxes[:,1].size(), device='cuda'))
                    pred_boxes[:, 2] = torch.min(pred_boxes[:, 2], torch.full(pred_boxes[:,2].size(), img_width - 1, device='cuda'))
                    pred_boxes[:, 3] = torch.min(pred_boxes[:, 3], torch.full(pred_boxes[:,3].size(), img_height - 1, device='cuda'))
                elif '.mat' in pth:
                    pred_boxes[:, 0] = torch.max(pred_boxes[:, 0], torch.ones(pred_boxes[:,0].size(), device='cuda'))
                    pred_boxes[:, 1] = torch.max(pred_boxes[:, 1], torch.ones(pred_boxes[:,1].size(), device='cuda'))
                    pred_boxes[:, 2] = torch.min(pred_boxes[:, 2], torch.full(pred_boxes[:,2].size(), img_width, device='cuda'))
                    pred_boxes[:, 3] = torch.min(pred_boxes[:, 3], torch.full(pred_boxes[:,3].size(), img_height, device='cuda'))
                refined_boxes = torch.cat((refined_boxes, pred_boxes), dim=1)

            refined_boxes = refined_boxes.view((num_boxes, len(chosen_classes), 4))

            box_list.bbox = refined_boxes

        return self.boxes
