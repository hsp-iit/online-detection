# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from .roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_box_predictors import make_roi_box_predictor
from .inference import make_roi_box_post_processor
from .loss import make_roi_box_loss_evaluator
import OnlineDetectionPostProcessor as odp


class ROIBoxHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg, in_channels):
        super(ROIBoxHead, self).__init__()

        self.score_thresh = cfg['MODEL']['ROI_HEADS']['SCORE_THRESH']
        self.nms = cfg['MODEL']['ROI_HEADS']['NMS']
        self.detections_per_img = cfg['TEST']['DETECTIONS_PER_IMG']
        self.num_classes = cfg['MINIBOOTSTRAP']['DETECTOR']['NUM_CLASSES'] + 1

        self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_box_predictor(
            cfg, self.feature_extractor.out_channels)

        self.post_processor = None
        self.loss_evaluator = make_roi_box_loss_evaluator(cfg)

        self.cfg = cfg

    def forward(self, features, proposals, gt_bbox=None, gt_label=None, img_size=None, gt_labels_list=None, is_train=True, result_dir=None, targets=None):

        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        x = self.feature_extractor(features, proposals)

        if hasattr(self.predictor, 'classifiers'):
            if self.post_processor is None:
                self.post_processor = odp.OnlineDetectionPostProcessor(score_thresh=self.score_thresh, nms=self.nms,
                                                                       detections_per_img=self.detections_per_img)
            # final classifier that converts the features into predictions
            cls_scores, bbox_pred = self.predictor(x)
            result = self.post_processor((cls_scores, bbox_pred), proposals, len(self.predictor.classifiers) + 1, img_size)
            return features, result, {}

        else:
            if self.post_processor is None:
                self.post_processor = make_roi_box_post_processor(self.cfg)
            # final classifier that converts the features into predictions
            class_logits, box_regression = self.predictor(x)
            result = self.post_processor((class_logits, box_regression), proposals)
            return features, result, {}


def build_roi_box_head(cfg, in_channels):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIBoxHead(cfg, in_channels)
