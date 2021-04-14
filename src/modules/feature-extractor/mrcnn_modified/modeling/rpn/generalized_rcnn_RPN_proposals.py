# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

from maskrcnn_benchmark.modeling.backbone import build_backbone
from .rpn_getProposals import build_rpn

class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)

    def forward(self, images, gt_bbox = None, gt_label = None, img_size = [0,0], compute_average_recall_RPN=False, gt_labels_list = None, is_train = True, result_dir = None, extract_features_segmentation=False, img_name=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        images = to_image_list(images)
        features = self.backbone(images.tensors)
        proposals, proposal_losses, average_recall_RPN = self.rpn(images, features, gt_bbox=gt_bbox, img_size=img_size, compute_average_recall_RPN=compute_average_recall_RPN, is_train = is_train, result_dir = result_dir)
        return average_recall_RPN

