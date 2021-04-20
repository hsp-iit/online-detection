# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.layers import Conv2d
from maskrcnn_benchmark.layers import ConvTranspose2d
from maskrcnn_benchmark.modeling import registry

import torch

import falkon
from falkon.mmv_ops import batch_mmv

@registry.ROI_MASK_PREDICTOR.register("MaskRCNNC4Predictor")
class MaskRCNNC4Predictor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(MaskRCNNC4Predictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        dim_reduced = cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS[-1]
        num_inputs = in_channels

        self.conv5_mask = ConvTranspose2d(num_inputs, dim_reduced, 2, 2, 0)
        self.mask_fcn_logits = Conv2d(dim_reduced, num_classes, 1, 1, 0)

        # TODO decide how to set this param
        self.parallel_inference = True

        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                # Caffe2 implementation uses MSRAFill, which in fact
                # corresponds to kaiming_normal_ in PyTorch
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = F.relu(self.conv5_mask(x))
        feat_width = x.size()[2]
        if hasattr(self, 'classifiers'):
            # Normalize features
            x = x.permute(0,2,3,1).reshape(-1,x.size()[1])
            x = x - self.stats['mean']
            x = x * (20 / self.stats['mean_norm'])
            if self.parallel_inference:
                return self.predict_pixel_FALKON_parallel(x, feat_width)
            else:
                return self.predict_pixel_FALKON(x, feat_width)
        else:
            return self.mask_fcn_logits(x)

    def predict_pixel_FALKON(self, features, feat_width):
        # Set background class to the default negative value -2
        pixels_scores = torch.full((features.size()[0], 1), -2, device='cuda')
        for classifier in self.classifiers:
            # If the classifier is not available, set the pixel value to the default value -2 (which is smaller than all the other proposed values by trained FALKON classifiers)
            if classifier is None:
                predictions = torch.full((features.size()[0], 1), -2, device='cuda')
            # Compute pixel predictions with falkon classifier
            else:
                predictions = classifier.predict(features)
            pixels_scores = torch.cat((pixels_scores, predictions), dim=1)

        to_return = torch.empty((0, len(self.classifiers)+1, feat_width, feat_width), device='cuda')    #TODO maybe the loop can be optimized
        for i in range(int((len(predictions))/(feat_width**2))):
            to_return = torch.cat((to_return, pixels_scores[i*(feat_width**2):(i+1)*(feat_width**2)].T.reshape(1, len(self.classifiers)+1, feat_width, feat_width)))
        return to_return

    def predict_pixel_FALKON_parallel(self, features, feat_width):
        if not hasattr(self, 'nystrom_parallel'):
            self.kernel = None
            self.max_nystrom_centers = 0
            for i in range(len(self.classifiers)):
                if self.classifiers[i]:
                    self.max_nystrom_centers = max(self.max_nystrom_centers, self.classifiers[i].M)
            for i in range(len(self.classifiers)):
                if self.classifiers[i] and not self.kernel:
                    self.kernel = self.classifiers[i].kernel
            self.alpha_parallel = torch.cat([torch.nn.functional.pad(self.classifiers[i].alpha_, (0, 0, 0, self.max_nystrom_centers - len(self.classifiers[i].alpha_))).unsqueeze(0) if self.classifiers[i] else torch.zeros(
                (1, self.max_nystrom_centers, 1), device='cuda') for i in range(len(self.classifiers))])
            self.nystrom_parallel = torch.cat([torch.nn.functional.pad(self.classifiers[i].ny_points_, (0, 0, 0, self.max_nystrom_centers - len(self.classifiers[i].ny_points_))).unsqueeze(0) if self.classifiers[i] else torch.zeros(
                (1, self.max_nystrom_centers, features.size()[1]), device='cuda') for i in range(len(self.classifiers))])

        features = features.repeat((len(self.classifiers), 1, 1))
        pixels_scores = batch_mmv.batch_fmmv_incore(features, self.nystrom_parallel, self.alpha_parallel, self.kernel)
        pixels_scores = torch.cat((torch.full((features.size()[1], 1), -2, device='cuda'), torch.t(pixels_scores.squeeze())), dim=1)

        to_return = torch.empty((0, len(self.classifiers)+1, feat_width, feat_width), device='cuda')    #TODO maybe the loop can be optimized
        for i in range(int((len(pixels_scores))/(feat_width**2))):
            to_return = torch.cat((to_return, pixels_scores[i*(feat_width**2):(i+1)*(feat_width**2)].T.reshape(1, len(self.classifiers)+1, feat_width, feat_width)))
        return to_return


@registry.ROI_MASK_PREDICTOR.register("MaskRCNNConv1x1Predictor")
class MaskRCNNConv1x1Predictor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(MaskRCNNConv1x1Predictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        num_inputs = in_channels

        self.mask_fcn_logits = Conv2d(num_inputs, num_classes, 1, 1, 0)

        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                # Caffe2 implementation uses MSRAFill, which in fact
                # corresponds to kaiming_normal_ in PyTorch
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        return self.mask_fcn_logits(x)


def make_roi_mask_predictor(cfg, in_channels):
    func = registry.ROI_MASK_PREDICTOR[cfg.MODEL.ROI_MASK_HEAD.PREDICTOR]
    return func(cfg, in_channels)
