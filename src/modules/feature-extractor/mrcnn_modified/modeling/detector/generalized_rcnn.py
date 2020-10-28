# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

from maskrcnn_benchmark.modeling.backbone import build_backbone
from mrcnn_modified.modeling.rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads

import time

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
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)

    def forward(self, images, gt_bbox = None, gt_label = None, img_size = [0,0], compute_average_recall_RPN=False, gt_labels_list = None, is_train = True, result_dir = None, evaluate_segmentation=True, eval_segm_with_gt_bboxes=False):
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
        if gt_bbox is not None:
            proposals, proposal_losses, average_recall_RPN = self.rpn(images, features, gt_bbox.resize((images.image_sizes[0][1], images.image_sizes[0][0])), compute_average_recall_RPN=compute_average_recall_RPN)
        else:
            proposals, proposal_losses, average_recall_RPN = self.rpn(images, features, gt_bbox, compute_average_recall_RPN=compute_average_recall_RPN)
        if gt_bbox is not None and is_train:
            # Resize the ground truth boxes to the correct format
            width, height = proposals[0].size
            gt_bbox = gt_bbox.resize((width, height))
            # Add the ground truth proposals to the proposal vector
            proposals[0].bbox = torch.cat((gt_bbox.bbox, proposals[0].bbox), 0)
            proposals[0].extra_fields['objectness'] = torch.cat((1.0 * torch.ones(gt_bbox.bbox.size()[0], device="cuda"), proposals[0].extra_fields['objectness']), 0)

        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, gt_bbox = gt_bbox, gt_label= gt_label, img_size=img_size, gt_labels_list = gt_labels_list, is_train = is_train, result_dir = result_dir, evaluate_segmentation=evaluate_segmentation, eval_segm_with_gt_bboxes=eval_segm_with_gt_bboxes)

        return average_recall_RPN, result

