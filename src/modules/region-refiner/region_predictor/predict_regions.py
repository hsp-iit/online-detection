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

class RegionPredictor():
    def __init__(self, cfg, models, boxes):
        self.cfg = cfg
        self.features_format = self.cfg['FEATURES_FORMAT']
        self.path_to_features = self.cfg['PATHS']['FEATURES_PATH_TEST']+'/%s'+self.features_format
        self.path_to_imgset_test = self.cfg['PATHS']['IMAGESET_TRAIN']
        self.features_dictionary_test = list_features(self.path_to_imgset_test)
        self.models = models
        self.boxes = boxes
        return

    def __call__(self):
        pred_boxes = self.predict()
        return pred_boxes

    def predict(self):
        chosen_classes = ("__background__",
            "flower2", "flower5", "flower7",
            "mug1", "mug3", "mug4",
            "wallet6", "wallet7", "wallet10",
            "sodabottle2", "sodabottle3", "sodabottle4",
            "book4", "book6", "book9",
            "ringbinder4", "ringbinder5", "ringbinder6",
            "bodylotion2", "bodylotion5", "bodylotion8",
            "sprayer6", "sprayer8", "sprayer9",
            "pencilcase3", "pencilcase5", "pencilcase6",
            "hairclip2", "hairclip6", "hairclip8"
            )
        imdb = {
            'classes': chosen_classes
        }

        opts = self.cfg['opts']

        # cache_dir = 'bbox_reg/'
        # if not os.path.exists(cache_dir):
        #    os.mkdir(cache_dir)
        clss = imdb['classes']
        num_clss = len(clss)
        bbox_model_suffix = '_first_test'

        models = np.empty((0))
        #print(models)
        for key in self.features_dictionary_test:
            # TODO evaluate whether to add tic_toc_print
            pth = self.path_to_features % self.features_dictionary_test[key]
            print(pth)
            if '.pkl' in pth:
                with open(pth, 'rb') as f:
                    feat = pickle.load(f)
            elif '.mat' in pth:
                feat = loadmat(pth)


            start_time = time.time()
            for i in range(1, len(imdb['classes'])):
                if feat['class'][0] != i:
                    continue
                print("Predicting bounding boxes for class %s" %str(i))
                ex_box = feat['boxes'][0]

                Y=torch.empty((0)).to('cuda')
                for j in range(0, 4):
                    Y_j = torch.matmul(self.models[i-1]['Beta'][str(j)]['weights'][:-1], torch.tensor(feat['feat'][0]).to('cuda'))
                    Y_j += self.models[i-1]['Beta'][str(j)]['weights'][-1]
                    Y = torch.cat((Y,Y_j.view((1,1))), dim=1)

                Y = torch.matmul(torch.squeeze(Y), self.models[i-1]['T_inv'])
                Y += self.models[i-1]['mu']

                dst_ctr_x = Y[0]
                dst_ctr_y = Y[1]
                dst_scl_x = Y[2]
                dst_scl_y = Y[3]

                src_w = ex_box[2] - ex_box[0] + np.spacing(1)
                src_h = ex_box[3] - ex_box[1] + np.spacing(1)
                src_ctr_x = ex_box[0] + 0.5 * src_w
                src_ctr_y = ex_box[1] + 0.5 * src_h

                pred_ctr_x = (dst_ctr_x * src_w) + src_ctr_x
                pred_ctr_y = (dst_ctr_y * src_h) + src_ctr_y
                pred_w = torch.exp(dst_scl_x) * src_w
                pred_h = torch.exp(dst_scl_y) * src_h
                pred_boxes = torch.tensor([pred_ctr_x - 0.5 * pred_w, pred_ctr_y - 0.5 * pred_h,
                              pred_ctr_x + 0.5 * pred_w, pred_ctr_y + 0.5 * pred_h])
                print(pred_boxes, ex_box)
        return