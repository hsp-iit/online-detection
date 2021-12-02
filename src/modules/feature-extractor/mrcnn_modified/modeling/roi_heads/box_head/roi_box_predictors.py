# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#from maskrcnn_benchmark.modeling import registry
from mrcnn_modified.modeling import registry
from torch import nn
import torch

@registry.ROI_BOX_PREDICTOR.register("OnlineDetectionBOXPredictor")
class FastRCNNPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(FastRCNNPredictor, self).__init__()
        assert in_channels is not None

        num_inputs = in_channels

        num_classes = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.cls_score = nn.Linear(num_inputs, num_classes)
        num_bbox_reg_classes = 2 if config.MODEL.CLS_AGNOSTIC_BBOX_REG else num_classes
        self.bbox_pred = nn.Linear(num_inputs, num_bbox_reg_classes * 4)

        nn.init.normal_(self.cls_score.weight, mean=0, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)

        nn.init.normal_(self.bbox_pred.weight, mean=0, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)

        self.normalize_features_regressors = False

        self.parallel_inference = config.INFERENCE.PARALLEL_FALKON

    def forward(self, x, proposals=None):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # If FALKON classifiers, regressors and stats are defined use online pipeline
        if hasattr(self, 'classifiers'):
            if not self.normalize_features_regressors:
                # Refine boxes
                if self.parallel_inference:
                    if self.regressors.shape[0] > 0:
                        bbox_pred = self.refine_boxes_parallel(x)
                    # If the regressors are empty, use the sequential inference
                    else:
                        bbox_pred = self.refine_boxes(x)
                else:
                    bbox_pred = self.refine_boxes(x)
            if self.stats:
                # Normalize features
                x = x - self.stats['mean']
                x = x * (20 / self.stats['mean_norm'])
            if self.normalize_features_regressors:
                # Refine boxes
                if self.parallel_inference:
                    if self.regressors.shape[0] > 0:
                        bbox_pred = self.refine_boxes_parallel(x)
                    # If the regressors are empty, use the sequential inference
                    else:
                        bbox_pred = self.refine_boxes(x)
                else:
                    bbox_pred = self.refine_boxes(x)
            # Compute scores with FALKON
            if self.parallel_inference:
                if len(self.classifiers) > 1:
                    cls_scores = self.predict_clss_FALKON_parallel(x)
                # If the list of classifier is empty, use the sequential inference
                else:
                    cls_scores = self.predict_clss_FALKON(x)
            else:
                cls_scores = self.predict_clss_FALKON(x)
            return cls_scores, bbox_pred
        # Else use pretrained weights
        else:
            cls_logit = self.cls_score(x)
            bbox_pred = self.bbox_pred(x)
            return cls_logit, bbox_pred

    def refine_boxes(self, features):
        refined_boxes = torch.zeros((features.size()[0], 4), device='cuda')
        for j in range(len(self.regressors)):
            # Refine boxes with RLS regressors
            if self.regressors[j]['Beta'] is not None:
                weights = self.regressors[j]['Beta'][str(0)]['weights'].view(1, -1)
                for k in range(1, 4):
                    weights = torch.cat((weights, self.regressors[j]['Beta'][str(k)]['weights'].view(1, -1)))

                weights = torch.t(weights)
                Y = torch.matmul(features, weights[:-1])
                Y += weights[-1]
                Y = torch.matmul(Y, self.regressors[j]['T_inv'])
                Y += self.regressors[j]['mu']
            # If the regressor is not available, do not refine the boxes
            else:
                Y = torch.zeros((features.size()[0], 4), device='cuda')
            refined_boxes = torch.cat((refined_boxes, Y), dim=1)
        return refined_boxes

    def refine_boxes_parallel(self, features):
        if not hasattr(self, 'regressors_parallel'):
            weights_parallel = torch.zeros((features.size()[1] + 1, 4), device='cuda')
            T_inv_parallel = torch.zeros(((len(self.regressors)+1)*4, (len(self.regressors)+1)*4), device='cuda')
            mu_parallel = torch.zeros((1, 4), device='cuda')
            for j in range(len(self.regressors)):
                # Refine boxes with RLS regressors
                if self.regressors[j]['Beta'] is not None:
                    weights = torch.empty((0, features.size()[1] + 1), device='cuda')
                    for k in range(0, 4):
                        weights = torch.cat((weights, self.regressors[j]['Beta'][str(k)]['weights'].view(1, features.size()[1] + 1)))
                    # T_inv_parallel is a block diagonal matrix
                    T_inv_parallel[(j+1)*4:(j+2)*4, (j+1)*4:(j+2)*4] = self.regressors[j]['T_inv']
                    mu_parallel = torch.cat((mu_parallel, self.regressors[j]['mu'].view(1,4)), dim=1)
                else:
                    weights = torch.zeros((4, features.size()[1] + 1), device='cuda')
                    mu_parallel = torch.cat((mu_parallel, torch.zeros((1, 4), device='cuda')), dim=1)

                weights_parallel = torch.cat((weights_parallel, torch.t(weights)), dim=1)
            self.regressors_parallel = {'weights_parallel': weights_parallel,
                                        'T_inv_parallel': T_inv_parallel,
                                        'mu_parallel': mu_parallel}

        Y_parallel = torch.matmul(features, self.regressors_parallel['weights_parallel'][:-1])
        Y_parallel += self.regressors_parallel['weights_parallel'][-1]
        Y_parallel = torch.matmul(Y_parallel, self.regressors_parallel['T_inv_parallel'])
        Y_parallel += self.regressors_parallel['mu_parallel']
        return Y_parallel


    def predict_clss_FALKON(self, features):
        # Set background class to the default negative value -2
        objectness_scores = torch.full((features.size()[0], 1), -2, device='cuda')
        for classifier in self.classifiers:
            # If the classifier is not available, set the objectness to the default value -2 (which is smaller than all the other proposed values by trained FALKON classifiers)
            if classifier is None:
                predictions = torch.full((features.size()[0], 1), -2, device='cuda')
            # Compute objectness with falkon classifier
            else:
                predictions = classifier.predict(features)
            objectness_scores = torch.cat((objectness_scores, predictions), dim=1)
        return objectness_scores

    def predict_clss_FALKON_parallel(self, features):
        if not hasattr(self, 'nystrom_parallel'):
            self.kernel = None
            self.max_nystrom_centers = 0
            self.total_nystrom_centers = torch.sum(torch.tensor([self.classifiers[i].M if self.classifiers[i] else 0 for i in range(len(self.classifiers))]))
            self.alpha_parallel = torch.zeros((self.total_nystrom_centers, len(self.classifiers)), device='cuda')
            for i in range(len(self.classifiers)):
                if self.classifiers[i]:
                    self.max_nystrom_centers = max(self.max_nystrom_centers, self.classifiers[i].M)
            current_nystrom_center_row = 0
            for i in range(len(self.classifiers)):
                if self.classifiers[i] and not self.kernel:
                    self.kernel = self.classifiers[i].kernel
                if self.classifiers[i]:
                    self.alpha_parallel[current_nystrom_center_row:current_nystrom_center_row+self.classifiers[i].M, i] = self.classifiers[i].alpha_.squeeze()
                    current_nystrom_center_row += self.classifiers[i].M
            self.nystrom_parallel = torch.cat([self.classifiers[i].ny_points_ if self.classifiers[i] else torch.empty((0, self.feat_size), device='cuda') for i in range(len(self.classifiers))])

        self.scores = self.kernel.mmv(features, self.nystrom_parallel, self.alpha_parallel)
        scores = torch.cat((torch.full((features.size()[0], 1), -2, device='cuda'), self.scores), dim=1)
        return scores


@registry.ROI_BOX_PREDICTOR.register("FPNPredictor")
class FPNPredictor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(FPNPredictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        representation_size = in_channels

        self.cls_score = nn.Linear(representation_size, num_classes)
        num_bbox_reg_classes = 2 if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG else num_classes
        self.bbox_pred = nn.Linear(representation_size, num_bbox_reg_classes * 4)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

    def forward(self, x):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
            x = x.view(x.size(0), -1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas


def make_roi_box_predictor(cfg, in_channels):
    func = registry.ROI_BOX_PREDICTOR[cfg.MODEL.ROI_BOX_HEAD.PREDICTOR]
    return func(cfg, in_channels)
